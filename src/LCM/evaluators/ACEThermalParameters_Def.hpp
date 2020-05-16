// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <fstream>

#include "Albany_Macros.hpp"
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "ACEcommon.hpp" 

namespace LCM {

template <typename EvalT, typename Traits>
ACEThermalParameters<EvalT, Traits>::ACEThermalParameters(Teuchos::ParameterList& p, 
    const Teuchos::RCP<Albany::Layouts>& dl)
    : thermal_conductivity_(p.get<std::string> ("ACE Thermal Conductivity QP Variable Name"), 
		            dl->qp_scalar),
      thermal_inertia_(p.get<std::string> ("ACE Thermal Inertia QP Variable Name"), 
		            dl->qp_scalar),
      bluff_salinity_(p.get<std::string> ("ACE Bluff Salinity QP Variable Name"), 
		            dl->qp_scalar),
      ice_saturation_(p.get<std::string> ("ACE Ice Saturation QP Variable Name"), 
		            dl->qp_scalar),
      density_(p.get<std::string> ("ACE Density QP Variable Name"), 
		            dl->qp_scalar),
      heat_capacity_(p.get<std::string> ("ACE Density QP Variable Name"), 
		            dl->qp_scalar),
      water_saturation_(p.get<std::string> ("ACE Water Saturation QP Variable Name"), 
		            dl->qp_scalar),
      porosity_(p.get<std::string> ("ACE Porosity QP Variable Name"), 
		            dl->qp_scalar)
{
  Teuchos::ParameterList* cond_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<Teuchos::ParameterList const> reflist =
      this->getValidThermalCondParameters();

  // Check the parameters contained in the input file. Do not check the defaults
  // set programmatically
  cond_list->validateParameters(
      *reflist,
      0,
      Teuchos::VALIDATE_USED_ENABLED,
      Teuchos::VALIDATE_DEFAULTS_DISABLED);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  coord_vec_ = decltype(coord_vec_)(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  num_qps_  = dims[1];
  num_dims_ = dims[2];

  // We have a multiple material problem and need to map element blocks to
  // material data
 
  eb_names_ = p.get<Teuchos::ArrayRCP<std::string>>("Element Block Names", {});

  if (p.isType<Teuchos::RCP<Albany::MaterialDatabase>>("MaterialDB")) {
    material_db_ = p.get<Teuchos::RCP<Albany::MaterialDatabase>>("MaterialDB");
  } 
  else {
    ALBANY_ABORT("\nError! Must specify a material database for thermal parameters.\n"); 
  }
 
  this->createElementBlockParameterMaps(); 

  this->addDependentField(coord_vec_);
  this->addEvaluatedField(thermal_conductivity_);
  this->addEvaluatedField(thermal_inertia_);
  this->addEvaluatedField(bluff_salinity_);
  this->addEvaluatedField(ice_saturation_);
  this->addEvaluatedField(density_);
  this->addEvaluatedField(heat_capacity_);
  this->addEvaluatedField(water_saturation_);
  this->addEvaluatedField(porosity_);

  this->setName("ACE Thermal Parameters");
}


// **********************************************************************
template <typename EvalT, typename Traits>
void
ACEThermalParameters<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(thermal_conductivity_, fm);
  this->utils.setFieldData(thermal_inertia_, fm);
  this->utils.setFieldData(bluff_salinity_, fm);
  this->utils.setFieldData(ice_saturation_, fm);
  this->utils.setFieldData(density_, fm);
  this->utils.setFieldData(heat_capacity_, fm);
  this->utils.setFieldData(water_saturation_, fm);
  this->utils.setFieldData(porosity_, fm);
  this->utils.setFieldData(coord_vec_, fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
ACEThermalParameters<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  std::string eb_name = workset.EBName; 
  Teuchos::ParameterList& sublist 
	  = material_db_->getElementBlockSublist(eb_name, "ACE Thermal Parameters"); 
  ScalarT thermal_conduct_eb  = sublist.get("ACE Thermal Conductivity Value", 1.0);
  ScalarT thermal_inertia_eb  = sublist.get("ACE Thermal Inertia Value", 1.0);
  ScalarT sat_mod_eb = this->queryElementBlockParameterMap(eb_name, sat_mod_map_);
  //IKT FIXME: add similar calls to above, as needed  
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t qp = 0; qp < num_qps_; ++qp) {
      auto const height = Sacado::Value<ScalarT>::eval(coord_vec_(cell, qp, 2));
      ScalarT sal_eb = this->queryElementBlockParameterMap(eb_name, salinity_base_map_);
      auto const salinity_eb = this->queryElementBlockParameterMap(eb_name, salinity_map_); 
      auto const z_above_mean_sea_level_eb = this->queryElementBlockParameterMap(eb_name, 
		                                        z_above_mean_sea_level_map_); 
      if (salinity_eb.size() > 0) {
        sal_eb = interpolateVectors(z_above_mean_sea_level_eb, salinity_eb, height);
      }
      bluff_salinity_(cell, qp) = sal_eb;
      thermal_conductivity_(cell, qp) = thermal_conduct_eb; 
      thermal_inertia_(cell, qp) = thermal_inertia_eb; 
    }
  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename ACEThermalParameters<EvalT, Traits>::ScalarT&
ACEThermalParameters<EvalT, Traits>::getValue(std::string const& n)
{
  ALBANY_ABORT("\nError! Logic error in getting parameter " << n
      << " in ACE Thermal Parameters::getValue()!\n");
  return constant_value_;
}

// **********************************************************************
template <typename EvalT, typename Traits>
Teuchos::RCP<Teuchos::ParameterList const>
ACEThermalParameters<EvalT, Traits>::getValidThermalCondParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> valid_pl =
      rcp(new Teuchos::ParameterList("Valid ACE Thermal Parameters"));
  valid_pl->set<double>("ACE Thermal Conductivity Value", 1.0,
      "Constant thermal conductivity value across element block");
  valid_pl->set<double>("ACE Thermal Inertia Value", 1.0,
      "Constant thermal inertia value across element block");
  valid_pl->set<double>("Saturation Modulus", 0.0, 
      "Constant value of saturation modulus in element block");
  valid_pl->set<double>("Saturation Exponent", 0.0, 
      "Constant value of saturation exponent in element block");
  valid_pl->set<double>("ACE Ice Density", 0.0, 
      "Constant value of ice density in element block");
  valid_pl->set<double>("ACE Water Density", 0.0, 
      "Constant value of water density in element block");
  valid_pl->set<double>("ACE Sediment Density", 0.0, 
      "Constant value of sediment density in element block");
  valid_pl->set<double>("ACE Ice Thermal Conductivity", 0.0, 
      "Constant value of ice thermal conductivity in element block");
  valid_pl->set<double>("ACE Water Thermal Conductivity", 0.0, 
      "Constant value of water thermal conductivity in element block");
  valid_pl->set<double>("ACE Sediment Thermal Conductivity", 0.0, 
      "Constant value of sediment thermal conductivity in element block");
  valid_pl->set<double>("ACE Ice Heat Capacity", 0.0, 
      "Constant value of ice heat capacity in element block");
  valid_pl->set<double>("ACE Water Heat Capacity", 0.0, 
      "Constant value of water heat capacity in element block");
  valid_pl->set<double>("ACE Sediment Heat Capacity", 0.0, 
      "Constant value of sediment heat capacity in element block");
  valid_pl->set<double>("ACE Ice Initial Saturation", 0.0, 
      "Constant value of ice initial saturation in element block");
  valid_pl->set<double>("ACE Ice Maximum Saturation", 0.0, 
      "Constant value of ice maximum saturation in element block");
  valid_pl->set<double>("ACE Water Minimum Saturation", 0.0, 
      "Constant value of water minimum saturation in element block");
  valid_pl->set<double>("ACE Base Salinity", 0.0, 
      "Constant value of base salinity in element block");
  valid_pl->set<double>("ACE Salt Enhanced D", 0.0, 
      "Constant value of salt enhanced D in element block");
  valid_pl->set<double>("ACE Freezing Curve Shift", 0.25, 
      "Value of freezing curve shift in element block");
  valid_pl->set<double>("ACE Latent Heat", 0.0, 
      "Constant value latent heat in element block");
  valid_pl->set<double>("ACE Surface Porosity", 0.0, 
      "Constant value surface porosity in element block");
  return valid_pl;
}

// **********************************************************************
template <typename EvalT, typename Traits>
void 
ACEThermalParameters<EvalT, Traits>::createElementBlockParameterMaps() 
{
  for (int i=0; i<eb_names_.size(); i++) {
    Teuchos::ParameterList& sublist
          = material_db_->getElementBlockSublist(eb_names_[i], "ACE Thermal Parameters");
    //IKT, FIXME: may not need sat_mod and sat_exp - may be mechanics only; if so, remove  
    sat_mod_map_[eb_names_[i]] = sublist.get("Saturation Modulus", 0.0);
    sat_exp_map_[eb_names_[i]] = sublist.get("Saturation Exponent", 0.0);
    ice_density_map_[eb_names_[i]] = sublist.get("ACE Ice Density", 0.0);
    water_density_map_[eb_names_[i]] = sublist.get("ACE Water Density", 0.0);
    soil_density_map_[eb_names_[i]] = sublist.get("ACE Sediment Density", 0.0);
    ice_thermal_cond_map_[eb_names_[i]] = sublist.get("ACE Ice Thermal Conductivity", 0.0);
    water_thermal_cond_map_[eb_names_[i]] = sublist.get("ACE Water Thermal Conductivity", 0.0);
    soil_thermal_cond_map_[eb_names_[i]] = sublist.get("ACE Sediment Thermal Conductivity", 0.0);
    ice_heat_capacity_map_[eb_names_[i]] = sublist.get("ACE Ice Heat Capacity", 0.0);
    water_heat_capacity_map_[eb_names_[i]] = sublist.get("ACE Water Heat Capacity", 0.0);
    soil_heat_capacity_map_[eb_names_[i]] = sublist.get("ACE Sediment Heat Capacity", 0.0);
    ice_saturation_init_map_[eb_names_[i]] = sublist.get("ACE Ice Initial Saturation", 0.0);
    ice_saturation_max_map_[eb_names_[i]] = sublist.get("ACE Ice Maximum Saturation", 0.0);
    water_saturation_min_map_[eb_names_[i]] = sublist.get("ACE Water Minimum Saturation", 0.0);
    salinity_base_map_[eb_names_[i]] = sublist.get("ACE Base Salinity", 0.0);
    salinity_enhanced_D_map_[eb_names_[i]] = sublist.get("ACE Salt Enhanced D", 0.0);
    f_shift_map_[eb_names_[i]] = sublist.get("ACE Freezing Curve Shift", 0.25);
    latent_heat_map_[eb_names_[i]] = sublist.get("ACE Latent Heat", 0.0);
    porosity0_map_[eb_names_[i]] = sublist.get("ACE Surface Porosity", 0.0);

    if (sublist.isParameter("ACE Time File") == true) {
      const std::string filename = sublist.get<std::string>("ACE Time File");
      time_map_[eb_names_[i]]  = vectorFromFile(filename);
    }
    if (sublist.isParameter("ACE Sea Level File") == true) {
      const std::string filename = sublist.get<std::string>("ACE Sea Level File");
      sea_level_map_[eb_names_[i]] = vectorFromFile(filename);
    }
    if (sublist.isParameter("ACE Z Depth File") == true) {
      const std::string filename = sublist.get<std::string>("ACE Z Depth File");
      z_above_mean_sea_level_map_[eb_names_[i]] = vectorFromFile(filename); 
    }
    if (sublist.isParameter("ACE Salinity File") == true) {
      const std::string filename = sublist.get<std::string>("ACE Salinity File");
      salinity_map_[eb_names_[i]] = vectorFromFile(filename);
      ALBANY_ASSERT(
          z_above_mean_sea_level_map_[eb_names_[i]].size() == salinity_map_[eb_names_[i]].size(),
          "*** ERROR: Number of z values and number of salinity values in ACE "
          "Salinity File must match.");
    }
    if (sublist.isParameter("ACE Ocean Salinity File") == true) {
      const std::string filename = sublist.get<std::string>("ACE Ocean Salinity File");
      ocean_salinity_map_[eb_names_[i]] = vectorFromFile(filename);
      ALBANY_ASSERT(
          time_map_[eb_names_[i]].size() == ocean_salinity_map_[eb_names_[i]].size(),
          "*** ERROR: Number of time values and number of ocean salinity values "
          "in "
          "ACE Ocean Salinity File must match.");
    }
    if (sublist.isParameter("ACE Porosity File") == true) {
      const std::string filename = sublist.get<std::string>("ACE Porosity File");
      porosity_from_file_map_[eb_names_[i]] = vectorFromFile(filename);
      ALBANY_ASSERT(
          z_above_mean_sea_level_map_[eb_names_[i]].size() == porosity_from_file_map_[eb_names_[i]].size(),
          "*** ERROR: Number of z values and number of porosity values in "
          "ACE Porosity File must match.");
    }
    if (sublist.isParameter("ACE Sand File") == true) {
      const std::string filename = sublist.get<std::string>("ACE Sand File");
      sand_from_file_map_[eb_names_[i]] = vectorFromFile(filename);
      ALBANY_ASSERT(
          z_above_mean_sea_level_map_[eb_names_[i]].size() == sand_from_file_map_[eb_names_[i]].size(),
          "*** ERROR: Number of z values and number of sand values in "
          "ACE Sand File must match.");
    }
    if (sublist.isParameter("ACE Clay File") == true) {
      const std::string filename = sublist.get<std::string>("ACE Clay File");
      clay_from_file_map_[eb_names_[i]] = vectorFromFile(filename);
      ALBANY_ASSERT(
          z_above_mean_sea_level_map_[eb_names_[i]].size() == clay_from_file_map_[eb_names_[i]].size(),
          "*** ERROR: Number of z values and number of clay values in "
          "ACE Clay File must match.");
    }
    if (sublist.isParameter("ACE Silt File") == true) {
      const std::string filename = sublist.get<std::string>("ACE Silt File");
      silt_from_file_map_[eb_names_[i]] = vectorFromFile(filename);
      ALBANY_ASSERT(
          z_above_mean_sea_level_map_[eb_names_[i]].size() == silt_from_file_map_[eb_names_[i]].size(),
          "*** ERROR: Number of z values and number of silt values in "
          "ACE Silt File must match.");
    }
    if (sublist.isParameter("ACE Peat File") == true) {
      const std::string filename = sublist.get<std::string>("ACE Peat File");
      peat_from_file_map_[eb_names_[i]] = vectorFromFile(filename);
      ALBANY_ASSERT(
          z_above_mean_sea_level_map_[eb_names_[i]].size() == peat_from_file_map_[eb_names_[i]].size(),
          "*** ERROR: Number of z values and number of peat values in "
          "ACE Peat File must match.");
    }

    ALBANY_ASSERT(
        time_map_[eb_names_[i]].size() == sea_level_map_[eb_names_[i]].size(),
        "*** ERROR: Number of times and number of sea level values must match.");

  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
RealType
ACEThermalParameters<EvalT, Traits>::queryElementBlockParameterMap(const std::string eb_name, 
		                                                   const std::map<std::string, RealType> map)
{
  typename std::map<std::string, RealType>::const_iterator it; 
  it = map.find(eb_name); 
  if (it == map.end()) {
    ALBANY_ABORT("\nError! Element block = " << eb_name << " was not found in map!\n");
  } 
  return it->second; 
}
// **********************************************************************

template <typename EvalT, typename Traits>
std::vector<RealType>
ACEThermalParameters<EvalT, Traits>::queryElementBlockParameterMap(const std::string eb_name, 
		                                                   const std::map<std::string, std::vector<RealType>> map)
{
  typename std::map<std::string, std::vector<RealType>>::const_iterator it; 
  it = map.find(eb_name); 
  if (it == map.end()) {
    //Element block is not found in map - return std::vector of length 0
    std::vector<RealType> vec;  
    vec.resize(0); 
    return vec; 
  } 
  return it->second; 
}
// **********************************************************************
}  // namespace PHAL
