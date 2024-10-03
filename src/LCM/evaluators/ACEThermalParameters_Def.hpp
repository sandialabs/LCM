// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <MiniTensor.h>

#include <fstream>

#include "ACEcommon.hpp"
#include "Albany_Macros.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
ACEThermalParameters<EvalT, Traits>::ACEThermalParameters(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
    : thermal_conductivity_(p.get<std::string>("ACE_Therm_Cond QP Variable Name"), dl->qp_scalar),
      thermal_cond_grad_at_nodes_(p.get<std::string>("ACE_Therm_Cond Gradient Node Variable Name"), dl->node_vector),
      thermal_cond_grad_at_qps_(p.get<std::string>("ACE_Therm_Cond Gradient QP Variable Name"), dl->qp_vector),
      wgradbf_(p.get<std::string>("Weighted Gradient BF Name"), dl->node_qp_vector),
      bf_(p.get<std::string>("BF Name"), dl->node_qp_scalar),
      thermal_inertia_(p.get<std::string>("ACE_Thermal_Inertia QP Variable Name"), dl->qp_scalar),
      bluff_salinity_(p.get<std::string>("ACE_Bluff_Salinity QP Variable Name"), dl->qp_scalar),
      bluff_salinity_read_(p.get<std::string>("ACE_Bluff_SalinityRead QP Variable Name"), dl->qp_scalar),
      ice_saturation_(p.get<std::string>("ACE_Ice_Saturation QP Variable Name"), dl->qp_scalar),
      density_(p.get<std::string>("ACE_Density QP Variable Name"), dl->qp_scalar),
      heat_capacity_(p.get<std::string>("ACE_Heat_Capacity QP Variable Name"), dl->qp_scalar),
      water_saturation_(p.get<std::string>("ACE_Water_Saturation QP Variable Name"), dl->qp_scalar),
      porosity_(p.get<std::string>("ACE_Porosity QP Variable Name"), dl->qp_scalar),
      temperature_(p.get<std::string>("ACE Temperature QP Variable Name"), dl->qp_scalar),
      time_(p.get<double>("Current Time"))
{
  if (time_ == 0.0) is_initial_timestep_ = true;

  Teuchos::ParameterList* cond_list = p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<Teuchos::ParameterList const> reflist = this->getValidThermalCondParameters();

  // Check the parameters contained in the input file. Do not check the defaults
  // set programmatically
  cond_list->validateParameters(*reflist, 0, Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  Teuchos::RCP<PHX::DataLayout> vector_dl                   = p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  coord_vec_                                                = decltype(coord_vec_)(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
  Teuchos::RCP<PHX::DataLayout>           node_qp_vector_dl = p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  node_qp_vector_dl->dimensions(dims);
  workset_size_ = dims[0];
  num_nodes_    = dims[1];
  num_qps_      = dims[2];
  num_dims_     = dims[3];

  // We have a multiple material problem and need to map element blocks to
  // material data

  eb_names_ = p.get<Teuchos::ArrayRCP<std::string>>("Element Block Names", {});

  if (p.isType<Teuchos::RCP<Albany::MaterialDatabase>>("MaterialDB")) {
    material_db_ = p.get<Teuchos::RCP<Albany::MaterialDatabase>>("MaterialDB");
  } else {
    ALBANY_ABORT("\nError! Must specify a material database for thermal parameters.\n");
  }

  this->createElementBlockParameterMaps();

  this->addDependentField(coord_vec_.fieldTag());
  this->addDependentField(temperature_.fieldTag());
  this->addDependentField(wgradbf_.fieldTag());
  this->addDependentField(bf_.fieldTag());
  this->addDependentField(bluff_salinity_read_.fieldTag());
  this->addEvaluatedField(thermal_conductivity_);
  this->addEvaluatedField(thermal_inertia_);
  this->addEvaluatedField(bluff_salinity_);
  this->addEvaluatedField(ice_saturation_);
  this->addEvaluatedField(density_);
  this->addEvaluatedField(heat_capacity_);
  this->addEvaluatedField(water_saturation_);
  this->addEvaluatedField(porosity_);
  this->addEvaluatedField(thermal_cond_grad_at_nodes_);
  this->addEvaluatedField(thermal_cond_grad_at_qps_);

  this->setName("ACE Thermal Parameters");
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
ACEThermalParameters<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(thermal_conductivity_, fm);
  this->utils.setFieldData(thermal_inertia_, fm);
  this->utils.setFieldData(bluff_salinity_, fm);
  this->utils.setFieldData(bluff_salinity_read_, fm);
  this->utils.setFieldData(ice_saturation_, fm);
  this->utils.setFieldData(density_, fm);
  this->utils.setFieldData(heat_capacity_, fm);
  this->utils.setFieldData(water_saturation_, fm);
  this->utils.setFieldData(porosity_, fm);
  this->utils.setFieldData(coord_vec_, fm);
  this->utils.setFieldData(temperature_, fm);
  this->utils.setFieldData(thermal_cond_grad_at_nodes_, fm);
  this->utils.setFieldData(thermal_cond_grad_at_qps_, fm);
  this->utils.setFieldData(wgradbf_, fm);
  this->utils.setFieldData(bf_, fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
ACEThermalParameters<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  std::string const eb_name            = workset.EBName;
  auto const        num_cells          = workset.numCells;
  ScalarT           thermal_conduct_eb = this->queryElementBlockParameterMap(eb_name, const_thermal_conduct_map_);
  ScalarT           thermal_inertia_eb = this->queryElementBlockParameterMap(eb_name, const_thermal_inertia_map_);
  // Initialize thermal_cond_grad_at_nodes to zero
  for (std::size_t cell = 0; cell < num_cells; ++cell) {
    for (std::size_t node = 0; node < num_nodes_; ++node) {
      for (std::size_t ndim = 0; ndim < num_dims_; ++ndim) {
        thermal_cond_grad_at_nodes_(cell, node, ndim) = 0.0;
      }
    }
  }
  // Initialize thermal_cond_grad_at_qps to zero
  for (std::size_t cell = 0; cell < num_cells; ++cell) {
    for (std::size_t qp = 0; qp < num_qps_; ++qp) {
      for (std::size_t ndim = 0; ndim < num_dims_; ++ndim) {
        thermal_cond_grad_at_qps_(cell, qp, ndim) = 0.0;
      }
    }
  }

  // Set thermal conductivity, thermal inertia and other fields
  if ((thermal_conduct_eb >= 0) || (thermal_inertia_eb >= 0)) {
    for (std::size_t cell = 0; cell < num_cells; ++cell) {
      for (std::size_t qp = 0; qp < num_qps_; ++qp) {
        thermal_conductivity_(cell, qp) = thermal_conduct_eb;
        thermal_inertia_(cell, qp)      = thermal_inertia_eb;
      }
    }
    return;
  }

  double current_time = workset.current_time;
  double delta_time   = workset.time_step;

  Albany::AbstractDiscretization&    disc        = *workset.disc;
  Albany::STKDiscretization&         stk_disc    = dynamic_cast<Albany::STKDiscretization&>(disc);
  Albany::AbstractSTKMeshStruct&     mesh_struct = *(stk_disc.getSTKMeshStruct());
  Albany::AbstractSTKFieldContainer& field_cont  = *(mesh_struct.getFieldContainer());
  have_cell_boundary_indicator_                  = field_cont.hasCellBoundaryIndicatorField();

  if (have_cell_boundary_indicator_ == true) {
    cell_boundary_indicator_ = workset.cell_boundary_indicator;
    ALBANY_ASSERT(cell_boundary_indicator_.is_null() == false);
  }

  std::vector<RealType> const salinity_eb               = this->queryElementBlockParameterMap(eb_name, salinity_map_);
  std::vector<RealType> const z_above_mean_sea_level_eb = this->queryElementBlockParameterMap(eb_name, z_above_mean_sea_level_map_);
  std::vector<RealType> const time_eb                   = this->queryElementBlockParameterMap(eb_name, time_map_);
  std::vector<RealType> const sea_level_eb              = this->queryElementBlockParameterMap(eb_name, sea_level_map_);

  std::vector<RealType> porosity_from_file_eb = this->queryElementBlockParameterMap(eb_name, porosity_from_file_map_);
  std::vector<RealType> ocean_salinity_eb     = this->queryElementBlockParameterMap(eb_name, ocean_salinity_map_);
  std::vector<RealType> snow_depth_eb         = this->queryElementBlockParameterMap(eb_name, snow_depth_map_);
  std::vector<RealType> sand_from_file_eb     = this->queryElementBlockParameterMap(eb_name, sand_from_file_map_);
  std::vector<RealType> clay_from_file_eb     = this->queryElementBlockParameterMap(eb_name, clay_from_file_map_);
  std::vector<RealType> silt_from_file_eb     = this->queryElementBlockParameterMap(eb_name, silt_from_file_map_);
  std::vector<RealType> peat_from_file_eb     = this->queryElementBlockParameterMap(eb_name, peat_from_file_map_);
  // The following is for specifying snow for ACI/NH
  std::vector<RealType> air_from_file_eb = this->queryElementBlockParameterMap(eb_name, air_from_file_map_);

  ScalarT ice_density_eb         = this->queryElementBlockParameterMap(eb_name, ice_density_map_);
  ScalarT water_density_eb       = this->queryElementBlockParameterMap(eb_name, water_density_map_);
  ScalarT soil_density_eb        = this->queryElementBlockParameterMap(eb_name, soil_density_map_);
  ScalarT ice_heat_capacity_eb   = this->queryElementBlockParameterMap(eb_name, ice_heat_capacity_map_);
  ScalarT water_heat_capacity_eb = this->queryElementBlockParameterMap(eb_name, water_heat_capacity_map_);
  ScalarT soil_heat_capacity_eb  = this->queryElementBlockParameterMap(eb_name, soil_heat_capacity_map_);
  ScalarT ice_thermal_cond_eb    = this->queryElementBlockParameterMap(eb_name, ice_thermal_cond_map_);
  ScalarT water_thermal_cond_eb  = this->queryElementBlockParameterMap(eb_name, water_thermal_cond_map_);
  ScalarT soil_thermal_cond_eb   = this->queryElementBlockParameterMap(eb_name, soil_thermal_cond_map_);
  ScalarT thermal_factor_eb      = this->queryElementBlockParameterMap(eb_name, thermal_factor_map_);
  ScalarT latent_heat_eb         = this->queryElementBlockParameterMap(eb_name, latent_heat_map_);

  ScalarT const salinity_base_eb   = this->queryElementBlockParameterMap(eb_name, salinity_base_map_);
  ScalarT const porosity_bulk_eb   = this->queryElementBlockParameterMap(eb_name, porosity_bulk_map_);
  ScalarT const element_size_eb    = this->queryElementBlockParameterMap(eb_name, element_size_map_);
  ScalarT const salt_enhanced_D_eb = this->queryElementBlockParameterMap(eb_name, salt_enhanced_D_map_);

  ScalarT const cell_half_width    = 0.5 * element_size_eb;
  ScalarT const cell_exposed_area  = element_size_eb * element_size_eb;
  ScalarT const cell_volume        = cell_exposed_area * element_size_eb;
  ScalarT const per_exposed_length = 1.0 / element_size_eb;
  ScalarT const factor             = per_exposed_length * salt_enhanced_D_eb;

  for (std::size_t cell = 0; cell < num_cells; ++cell) {
    double const cell_bi     = have_cell_boundary_indicator_ == true ? *(cell_boundary_indicator_[cell]) : 0.0;
    bool const   is_erodible = cell_bi == 2.0;
    for (std::size_t qp = 0; qp < num_qps_; ++qp) {
      RealType const height = Sacado::Value<ScalarT>::eval(coord_vec_(cell, qp, 2));
      ScalarT        sal_eb = salinity_base_eb;
      if (salinity_eb.size() > 0) {
        sal_eb = interpolateVectors(z_above_mean_sea_level_eb, salinity_eb, height);
      }
      // IKT 11/4/2022: if we are in the initial timestep, set bluff_salinity from sal_eb
      if (is_initial_timestep_ == true) {
        bluff_salinity_(cell, qp) = sal_eb;
      }
      // IKT 11/4/2022: if we are not in the initial timestep, set bluff_salinity from bluff_salinity_read_ field
      else {
        bluff_salinity_(cell, qp) = bluff_salinity_read_(cell, qp);
      }
      const ScalarT sea_level = sea_level_eb.size() > 0 ? interpolateVectors(time_eb, sea_level_eb, current_time) : -999.0;
      // const ScalarT sea_level = sea_level_eb.size() > 0 ? (interpolateVectors(time_eb, sea_level_eb, current_time) * 2.0) : -999.0;

      // Thermal calculation
      // Calculate the depth-dependent porosity
      // NOTE: The porosity does not change in time so this calculation only
      // needs
      //       to be done once, at the beginning of the simulation.
      ScalarT porosity_eb = porosity_bulk_eb;
      if (porosity_from_file_eb.size() > 0) {
        porosity_eb = interpolateVectors(z_above_mean_sea_level_eb, porosity_from_file_eb, height);
      }
      porosity_(cell, qp) = porosity_eb;

      // IKT 2/23/2024: the following was added for the snow_depth field.
      // TODO Jenn: use this field to incorporate snow into mixture model
      ScalarT snow_depth(0.0);
      bool    snow_given{false};
      if (snow_depth_eb.size() > 0) {
        snow_depth = interpolateVectors(time_eb, snow_depth_eb, current_time);
        snow_given = true;
      }
      // std::cout << "IKT snow_depth = " << snow_depth << "\n";

      // Calculate the salinity of the grid cell
      if ((is_erodible == true) && (height <= sea_level)) {
        ScalarT       ocean_sal = salinity_base_eb;
        ScalarT const sal_curr  = bluff_salinity_(cell, qp);
        // IKT, FIXME?: ocean_salinity is not block-dependent, so we may want to
        // make it just a std::vector, to avoid creating and querying a map.
        ScalarT const zero_sal(0.0);
        if (ocean_salinity_eb.size() > 0) {
          ocean_sal = interpolateVectors(time_eb, ocean_salinity_eb, current_time);
        }
        // --- elyce begin commenting out (8-26-24) ---- 
        // Note: below is being commented out because re: email thread with Jenn, it was decided to actually just take the ocean salinity at the bluff face
        // the below code was *likely* overriden the subsequent line of code, but not necessarily... so I am commenting it out just to be safe
        // ScalarT const sal_diff   = ocean_sal - sal_curr;
        // ScalarT const sal_grad   = sal_diff / cell_half_width;
        // ScalarT const sal_update = sal_grad * delta_time * factor;
        // ScalarT       sal_trial  = sal_curr + sal_update;
        // if (sal_trial < zero_sal) sal_trial = zero_sal;
        // if (sal_trial > ocean_sal) sal_trial = ocean_sal;
        // bluff_salinity_(cell, qp) = std::min(sal_trial, ocean_sal);  // ensures the salinity doesn't exceed ocean salinity
        // OVERRIDES EVERYTHING ABOVE:

        // ---- end commented out
        bluff_salinity_(cell, qp) = std::max(ocean_sal, bluff_salinity_(cell, qp));
      }

      ScalarT const sal = bluff_salinity_(cell, qp);

      // Calculate melting temperature
      ScalarT sal15(0.0);
      if (sal > 0.0) {
        sal15 = std::sqrt(sal * sal * sal);
      }
      ScalarT const pressure_fixed = 1.0;
      // Tmelt is in Kelvin
      ScalarT const Tmelt = -0.057 * sal + 0.00170523 * sal15 - 0.0002154996 * sal * sal - 0.000753 / 10000.0 * pressure_fixed + 273.15;

      // Set current temperature
      ScalarT const& Tcurr = temperature_(cell, qp);

      // Check if sediment fractions were provided
      bool sediment_given{false};
      if ((sand_from_file_eb.size() > 0) && (clay_from_file_eb.size() > 0) && (silt_from_file_eb.size() > 0) && (peat_from_file_eb.size() > 0)) {
        sediment_given = true;
      }

      // Check if air fraction was provided
      bool air_given{false};
      if (air_from_file_eb.size() > 0) {
        air_given = true;
      }
      ScalarT  Tshift;
      ScalarT  Tdiff;
      RealType v = 0.1;

      if (sediment_given == true) {
        auto sand_frac = interpolateVectors(z_above_mean_sea_level_eb, sand_from_file_eb, height);
        auto clay_frac = interpolateVectors(z_above_mean_sea_level_eb, clay_from_file_eb, height);
        auto silt_frac = interpolateVectors(z_above_mean_sea_level_eb, silt_from_file_eb, height);
        auto peat_frac = interpolateVectors(z_above_mean_sea_level_eb, peat_from_file_eb, height);
        v              = (peat_frac * 0.1) + (sand_frac * 1.0) + (silt_frac * 15.0) + (clay_frac * 50.0);
        Tshift         = (peat_frac * 0.1) + (sand_frac * 0.3) + (silt_frac * 0.6) + (clay_frac * 1.0);
      } else {
        Tshift = 0.1;
      }

      // IKT 2/17/2024: code to use air frac goes here for Jenn to fill in.
      // Might want to move elsewhere in this function...
      if (air_given == true) {
        auto air_frac = interpolateVectors(z_above_mean_sea_level_eb, air_from_file_eb, height);
        // std::cout << "IKT air_frac = " << air_frac << "\n";
      }

      // Use freezing curve to get icurr and dfdT
      ScalarT icurr{1.0};
      ScalarT dfdT{0.0};

      RealType const A = 0.0;
      RealType const G = 1.0;
      RealType const C = 1.0;
      RealType const Q = 0.001;
      RealType const B = 10.0;

      Tdiff = Tcurr - (Tmelt + Tshift);

      ScalarT const r_max  = 1.7e308;  // This is the max double that can be represented in C++ w/o overflow
      ScalarT const tol_bt = 709.0;
      // IKT 2/4/2025: Please see ace_thermal_param_upper_bound_to_prevent_nans.pdf in
      // the arctic_coastal_erosion repo for a derivation of the following tolerance.
      // It is calcutated such that d^2f/dT^2 does not overflow.  This value comes up in the derivatives
      // of qebt, which are used to calculate the stiffness matrix for the problem.
      ScalarT const d        = 1.0 / 0.1 + 2.0;
      ScalarT const tol_qebt = pow(r_max, 1.0 / d) - C;
      // std::cout << "IKT r_max, v, d, 1/d, tol_qebt = " << r_max << ", " << v << ", " << d << ", "
      //           << 1.0/d << ", " << tol_qebt << "\n";
      ScalarT const bt = -B * Tdiff;

      if (bt < -tol_bt) {
        dfdT  = 0.0;
        icurr = 0.0;
      } else if (bt > tol_bt) {
        dfdT  = 0.0;
        icurr = 1.0;
      } else {
        ScalarT const qebt = Q * std::exp(bt);
        auto const    eps  = minitensor::machine_epsilon<RealType>();
        if (qebt < eps) {  // (C + qebt) ~ C :: occurs when totally melted
          dfdT  = 0.0;
          icurr = 0.0;
        } else if (qebt > tol_qebt) {  // (C + qebt) ~ qebt :: occurs in deep frozen state
          dfdT  = 0.0;
          icurr = 1.0;
        } else {  // occurs when near melting temperature
          dfdT  = -1.0 * ((B * Q * (G - A)) * pow(C + qebt, -1.0 / v) * (qebt / Q)) / (v * (C + qebt));
          icurr = 1.0 - (A + ((G - A) / (pow(C + qebt, 1.0 / v))));
        }
      }

      if (snow_given == true) {
        dfdT  = 0.0;
        icurr = 1.0;
      }

      std::min(icurr, 1.0);
      std::max(icurr, 0.0);

      // Update the water saturation
      ScalarT wcurr = 1.0 - icurr;

      ScalarT calc_soil_heat_capacity;
      ScalarT calc_soil_thermal_cond;
      ScalarT calc_soil_density;
      if (sediment_given == true) {
        ScalarT sand_frac = interpolateVectors(z_above_mean_sea_level_eb, sand_from_file_eb, height);
        ScalarT clay_frac = interpolateVectors(z_above_mean_sea_level_eb, clay_from_file_eb, height);
        ScalarT silt_frac = interpolateVectors(z_above_mean_sea_level_eb, silt_from_file_eb, height);
        ScalarT peat_frac = interpolateVectors(z_above_mean_sea_level_eb, peat_from_file_eb, height);

        // THERMAL PROPERTIES OF ROCKS, E.C. Robertson, U.S. Geological Survey
        // Open-File Report 88-441 (1988).
        // AGU presentation (2019) --> peat K value
        // Gnatowski, Tomasz (2016) Thermal properties of degraded lowland
        // peat-moorsh soils, EGU General Assembly 2016, held 17-22 April, 2016
        // in Vienna Austria, id. EPSC2016-8105 --> peat Cp value Cp values in
        // [J/kg/K]
        calc_soil_heat_capacity = (0.7e3 * sand_frac) + (0.6e3 * clay_frac) + (0.7e3 * silt_frac) + (1.93e3 * peat_frac);
        // K values in [W/K/m]
        calc_soil_thermal_cond = (8.0 * sand_frac) + (0.4 * clay_frac) + (4.9 * silt_frac) + (0.40 * peat_frac);
        // calc_soil_thermal_cond = (8.0 * sand_frac) + (0.4 * clay_frac) + (4.9 * silt_frac) + (0.08 * peat_frac);
        // calc_soil_thermal_cond = pow(8.0,sand_frac) * pow(0.4,clay_frac) * pow(4.9,silt_frac) * pow(0.08,peat_frac);
        //  Rho values in [kg/m3]
        //  Peat density from Emily Bristol
        calc_soil_density = (2600.0 * sand_frac) + (2350.0 * clay_frac) + (2500.0 * silt_frac) + (250.0 * peat_frac);
        // Update the effective material density
        density_(cell, qp) = (porosity_eb * ((ice_density_eb * icurr) + (water_density_eb * wcurr))) + ((1.0 - porosity_eb) * calc_soil_density);
      } else {
        density_(cell, qp) = (porosity_eb * ((ice_density_eb * icurr) + (water_density_eb * wcurr))) + ((1.0 - porosity_eb) * soil_density_eb);
      }

      if (snow_given == true) {
        auto const SWE     = 0.10;
        density_(cell, qp) = SWE * water_density_eb;
      }

      // Update the effective material heat capacity
      if (sediment_given == true) {
        heat_capacity_(cell, qp) =
            (porosity_eb * ((ice_heat_capacity_eb * icurr) + (water_heat_capacity_eb * wcurr))) + ((1.0 - porosity_eb) * calc_soil_heat_capacity);
      } else {
        heat_capacity_(cell, qp) =
            (porosity_eb * ((ice_heat_capacity_eb * icurr) + (water_heat_capacity_eb * wcurr))) + ((1.0 - porosity_eb) * soil_heat_capacity_eb);
      }
      // HACK!! HACK!! HACK!!
      // HACK!! HACK!! HACK!!
      heat_capacity_(cell, qp) = (1.0) * heat_capacity_(cell, qp);

      if (snow_given == true) {
        heat_capacity_(cell, qp) = 2090.0;  // [J/kg/K]
      }

      // Update the effective material thermal conductivity
      if (sediment_given == true) {
        thermal_conductivity_(cell, qp) =
            (porosity_eb * ((ice_thermal_cond_eb * icurr) + (water_thermal_cond_eb * wcurr))) + ((1.0 - porosity_eb) * calc_soil_thermal_cond);
        // thermal_conductivity_(cell, qp) =
        //     pow(ice_thermal_cond_eb,(icurr*porosity_eb)) * pow(water_thermal_cond_eb,(wcurr*porosity_eb)) * pow(calc_soil_thermal_cond,(1.0 - porosity_eb));
      } else {
        thermal_conductivity_(cell, qp) =
            (porosity_eb * ((ice_thermal_cond_eb * icurr) + (water_thermal_cond_eb * wcurr))) + ((1.0 - porosity_eb) * soil_thermal_cond_eb);
      }
      // HACK!! HACK!! HACK!!
      // HACK!! HACK!! HACK!!
      thermal_conductivity_(cell, qp) = (1.0) * thermal_conductivity_(cell, qp);

      if (snow_given == true) {
        ScalarT       snow_K = 0.1;              // [W/K/m]
        ScalarT const dZ     = element_size_eb;  // [m]

        // Elyce 5/23/24 Put up safeguard against K going to negative infty for snow_depth values very small but less than 0.:
        // As long as snow_depth is nonzero, it'll get caught by the min condition not to let it go above 15
        if (snow_depth < 0.0) {
          snow_depth = 0.0;
        }

        snow_K                          = snow_K * (dZ / snow_depth);
        snow_K                          = std::min(snow_K, 15.0);  // [W/K/m] // don't let it go above 15
        thermal_conductivity_(cell, qp) = snow_K;                  // std::min(snow_K, 15.0);  // [W/K/m]

        // Elyce debugging:
        // std::cout << "\n snow_depth = " << snow_depth << " , dZ = " << dZ << "snow_K = " << snow_K << ", thermal_cond = " << thermal_conductivity_(cell, qp);
      }

      // Jenn's sub-grid scale model to calibrate niche formation follows.
      // By default, thermal_factor = 1.0, so that no scaling occurs.
      if ((is_erodible == true) && (height <= sea_level)) {
        thermal_conductivity_(cell, qp) = thermal_conductivity_(cell, qp) * thermal_factor_eb;
        heat_capacity_(cell, qp)        = heat_capacity_(cell, qp) / thermal_factor_eb;
      }

      // Update the material thermal inertia term
      thermal_inertia_(cell, qp) = (density_(cell, qp) * heat_capacity_(cell, qp)) - (ice_density_eb * latent_heat_eb * dfdT);
      // Return values
      ice_saturation_(cell, qp)   = icurr;
      water_saturation_(cell, qp) = wcurr;
    }
  }
  // Calculate thermal conductivity gradient at nodes using thermal conductivity and wgradbf
  for (std::size_t cell = 0; cell < num_cells; ++cell) {
    for (std::size_t node = 0; node < num_nodes_; ++node) {
      for (std::size_t qp = 0; qp < num_qps_; ++qp) {
        for (std::size_t ndim = 0; ndim < num_dims_; ++ndim) {
          thermal_cond_grad_at_nodes_(cell, node, ndim) += thermal_conductivity_(cell, qp) * wgradbf_(cell, node, qp, ndim);
        }
      }
    }
  }
  // Calculate thermal conductivity gradient at qps using thermal conductivity gradient at
  // nodes and bf_
  for (std::size_t cell = 0; cell < num_cells; ++cell) {
    for (int qp = 0; qp < num_qps_; ++qp) {
      for (std::size_t ndim = 0; ndim < num_dims_; ++ndim) {
        for (std::size_t node = 0; node < num_nodes_; ++node) {
          thermal_cond_grad_at_qps_(cell, qp, ndim) += thermal_cond_grad_at_nodes_(cell, node, ndim) * bf_(cell, node, qp);
        }
      }
    }
  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
Teuchos::RCP<Teuchos::ParameterList const>
ACEThermalParameters<EvalT, Traits>::getValidThermalCondParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> valid_pl = rcp(new Teuchos::ParameterList("Valid ACE Thermal Parameters"));
  valid_pl->set<double>("ACE_Therm_Cond Value", 1.0, "Constant thermal conductivity value across element block");
  valid_pl->set<double>("ACE_Thermal_Inertia Value", 1.0, "Constant thermal inertia value across element block");
  valid_pl->set<double>("ACE Ice Density", 920.0, "Constant value of ice density in element block");
  valid_pl->set<double>("ACE Water Density", 1000.0, "Constant value of water density in element block");
  valid_pl->set<double>("ACE Sediment Density", 2650.0, "Constant value of sediment density in element block");
  valid_pl->set<double>("ACE Ice Thermal Conductivity", 2.3, "Constant value of ice thermal conductivity in element block");
  valid_pl->set<double>("ACE Water Thermal Conductivity", 0.6, "Constant value of water thermal conductivity in element block");
  valid_pl->set<double>("ACE Sediment Thermal Conductivity", 4.3, "Constant value of sediment thermal conductivity in element block");
  valid_pl->set<double>("ACE Ice Heat Capacity", 2.0e+03, "Constant value of ice heat capacity in element block");
  valid_pl->set<double>("ACE Water Heat Capacity", 4.0e+03, "Constant value of water heat capacity in element block");
  valid_pl->set<double>("ACE Sediment Heat Capacity", 0.7e+03, "Constant value of sediment heat capacity in element block");
  valid_pl->set<double>("ACE Base Salinity", 0.0, "Constant value of base salinity in element block");
  valid_pl->set<double>("ACE Salt Enhanced D", 0.0, "Constant value of salt enhanced D in element block");
  valid_pl->set<double>("ACE Latent Heat", 334.0, "Constant value latent heat in element block");
  valid_pl->set<double>("ACE Bulk Porosity", 0.60, "Constant value bulk porosity in element block");
  valid_pl->set<double>("ACE Element Size", 1.0, "Constant value of element size in element block");
  return valid_pl;
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
ACEThermalParameters<EvalT, Traits>::createElementBlockParameterMaps()
{
  for (int i = 0; i < eb_names_.size(); i++) {
    std::string eb_name                 = eb_names_[i];
    const_thermal_conduct_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE_Therm_Cond Value", -1.0);
    if (const_thermal_conduct_map_[eb_name] != -1.0) {
      ALBANY_ASSERT((const_thermal_conduct_map_[eb_name] > 0.0), "*** ERROR: ACE_Therm_Cond Value must be positive!");
    }
    const_thermal_inertia_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE_Thermal_Inertia Value", -1.0);
    if (const_thermal_inertia_map_[eb_name] != -1.0) {
      ALBANY_ASSERT((const_thermal_inertia_map_[eb_name] > 0.0), "*** ERROR: ACE_Thermal_Inertia Value must be positive!");
    }
    ice_density_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Ice Density", 920.0);
    ALBANY_ASSERT((ice_density_map_[eb_name] >= 0.0), "*** ERROR: ACE Ice Density must be non-negative!");
    water_density_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Water Density", 1000.0);
    ALBANY_ASSERT((water_density_map_[eb_name] >= 0.0), "*** ERROR: ACE Water Density must be non-negative!");
    soil_density_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Sediment Density", 2650.0);
    ALBANY_ASSERT((soil_density_map_[eb_name] >= 0.0), "*** ERROR: ACE Soil Density must be non-negative!");
    ice_thermal_cond_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Ice Thermal Conductivity", 2.3);
    ALBANY_ASSERT((ice_thermal_cond_map_[eb_name] >= 0.0), "*** ERROR: ACE Ice Thermal Conductivity must be non-negative!");
    water_thermal_cond_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Water Thermal Conductivity", 0.6);
    ALBANY_ASSERT((water_thermal_cond_map_[eb_name] >= 0.0), "*** ERROR: ACE Water Thermal Conductivity must be non-negative!");
    soil_thermal_cond_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Sediment Thermal Conductivity", 4.3);
    ALBANY_ASSERT((soil_thermal_cond_map_[eb_name] >= 0.0), "*** ERROR: ACE Sediment Thermal Conductivity must be non-negative!");
    ice_heat_capacity_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Ice Heat Capacity", 2.0e+03);
    ALBANY_ASSERT((ice_heat_capacity_map_[eb_name] >= 0.0), "*** ERROR: ACE Ice Heat Capacity must be non-negative!");
    water_heat_capacity_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Water Heat Capacity", 4.0e+03);
    ALBANY_ASSERT((water_heat_capacity_map_[eb_name] >= 0.0), "*** ERROR: ACE Water Heat Capacity must be non-negative!");
    soil_heat_capacity_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Sediment Heat Capacity", 0.7e+03);
    ALBANY_ASSERT((soil_heat_capacity_map_[eb_name] >= 0.0), "*** ERROR: ACE Sediment Heat Capacity must be non-negative!");
    salinity_base_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Base Salinity", 0.0);
    ALBANY_ASSERT((salinity_base_map_[eb_name] >= 0.0), "*** ERROR: ACE Base Salinity must be non-negative!");
    salt_enhanced_D_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Salt Enhanced D", 0.0);
    ALBANY_ASSERT((salt_enhanced_D_map_[eb_name] >= 0.0), "*** ERROR: ACE Salt Enhanced D must be non-negative!");
    latent_heat_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Latent Heat", 334.0);
    ALBANY_ASSERT((latent_heat_map_[eb_name] >= 0.0), "*** ERROR: ACE Latent Heat must be non-negative!");
    porosity_bulk_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Bulk Porosity", 0.60);
    ALBANY_ASSERT((porosity_bulk_map_[eb_name] >= 0.0), "*** ERROR: ACE Bulk Porosity must be non-negative!");
    element_size_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Element Size", 1.0);
    ALBANY_ASSERT((element_size_map_[eb_name] >= 0.0), "*** ERROR: ACE Element Size must be non-negative!");
    thermal_factor_map_[eb_name] = material_db_->getElementBlockParam<RealType>(eb_name, "ACE Thermal Erosion Factor", 1.0);
    ALBANY_ASSERT((thermal_factor_map_[eb_name] >= 1.0), "*** ERROR: ACE Salt Enhanced D must be greater than or equal to 1!");

    if (material_db_->isElementBlockParam(eb_name, "ACE Time File") == true) {
      std::string const filename = material_db_->getElementBlockParam<std::string>(eb_name, "ACE Time File");
      time_map_[eb_name]         = vectorFromFile(filename);
    }
    if (material_db_->isElementBlockParam(eb_name, "ACE Sea Level File") == true) {
      std::string const filename = material_db_->getElementBlockParam<std::string>(eb_name, "ACE Sea Level File");
      sea_level_map_[eb_name]    = vectorFromFile(filename);
    }
    if (material_db_->isElementBlockParam(eb_name, "ACE Z Depth File") == true) {
      std::string const filename           = material_db_->getElementBlockParam<std::string>(eb_name, "ACE Z Depth File");
      z_above_mean_sea_level_map_[eb_name] = vectorFromFile(filename);
    }
    if (material_db_->isElementBlockParam(eb_name, "ACE Salinity File") == true) {
      std::string const filename = material_db_->getElementBlockParam<std::string>(eb_name, "ACE Salinity File");
      salinity_map_[eb_name]     = vectorFromFile(filename);
      ALBANY_ASSERT(
          z_above_mean_sea_level_map_[eb_name].size() == salinity_map_[eb_name].size(),
          "*** ERROR: Number of z values and number of salinity values in ACE "
          "Salinity File must match. \n"
          "Hint: Did you provide the 'ACE Z Depth File'?");
    }
    if (material_db_->isElementBlockParam(eb_name, "ACE Ocean Salinity File") == true) {
      std::string const filename   = material_db_->getElementBlockParam<std::string>(eb_name, "ACE Ocean Salinity File");
      ocean_salinity_map_[eb_name] = vectorFromFile(filename);
      ALBANY_ASSERT(
          time_map_[eb_name].size() == ocean_salinity_map_[eb_name].size(),
          "*** ERROR: Number of time values and number of ocean salinity "
          "values in "
          "ACE Ocean Salinity File must match.");
    }
    if (material_db_->isElementBlockParam(eb_name, "ACE Snow Depth File") == true) {
      std::string const filename = material_db_->getElementBlockParam<std::string>(eb_name, "ACE Snow Depth File");
      snow_depth_map_[eb_name]   = vectorFromFile(filename);
      ALBANY_ASSERT(
          time_map_[eb_name].size() == snow_depth_map_[eb_name].size(),
          "*** ERROR: Number of time values and number of snow depth "
          "values in "
          "ACE Snow Depth File must match.");
    }
    if (material_db_->isElementBlockParam(eb_name, "ACE_Porosity File") == true) {
      std::string const filename       = material_db_->getElementBlockParam<std::string>(eb_name, "ACE_Porosity File");
      porosity_from_file_map_[eb_name] = vectorFromFile(filename);
      ALBANY_ASSERT(
          z_above_mean_sea_level_map_[eb_name].size() == porosity_from_file_map_[eb_name].size(),
          "*** ERROR: Number of z values and number of porosity values in "
          "ACE_Porosity File must match. \n"
          "Hint: Did you provide the 'ACE Z Depth File'?");
    }
    if (material_db_->isElementBlockParam(eb_name, "ACE Sand File") == true) {
      std::string const filename   = material_db_->getElementBlockParam<std::string>(eb_name, "ACE Sand File");
      sand_from_file_map_[eb_name] = vectorFromFile(filename);
      ALBANY_ASSERT(
          z_above_mean_sea_level_map_[eb_name].size() == sand_from_file_map_[eb_name].size(),
          "*** ERROR: Number of z values and number of sand values in "
          "ACE Sand File must match. \n"
          "Hint: Did you provide the 'ACE Z Depth File'?");
    }
    if (material_db_->isElementBlockParam(eb_name, "ACE Clay File") == true) {
      std::string const filename   = material_db_->getElementBlockParam<std::string>(eb_name, "ACE Clay File");
      clay_from_file_map_[eb_name] = vectorFromFile(filename);
      ALBANY_ASSERT(
          z_above_mean_sea_level_map_[eb_name].size() == clay_from_file_map_[eb_name].size(),
          "*** ERROR: Number of z values and number of clay values in "
          "ACE Clay File must match. \n"
          "Hint: Did you provide the 'ACE Z Depth File'?");
    }
    if (material_db_->isElementBlockParam(eb_name, "ACE Silt File") == true) {
      std::string const filename   = material_db_->getElementBlockParam<std::string>(eb_name, "ACE Silt File");
      silt_from_file_map_[eb_name] = vectorFromFile(filename);
      ALBANY_ASSERT(
          z_above_mean_sea_level_map_[eb_name].size() == silt_from_file_map_[eb_name].size(),
          "*** ERROR: Number of z values and number of silt values in "
          "ACE Silt File must match. \n"
          "Hint: Did you provide the 'ACE Z Depth File'?");
    }
    if (material_db_->isElementBlockParam(eb_name, "ACE Peat File") == true) {
      std::string const filename   = material_db_->getElementBlockParam<std::string>(eb_name, "ACE Peat File");
      peat_from_file_map_[eb_name] = vectorFromFile(filename);
      ALBANY_ASSERT(
          z_above_mean_sea_level_map_[eb_name].size() == peat_from_file_map_[eb_name].size(),
          "*** ERROR: Number of z values and number of peat values in "
          "ACE Peat File must match. \n"
          "Hint: Did you provide the 'ACE Z Depth File'?");
    }
    if (material_db_->isElementBlockParam(eb_name, "ACE Air File") == true) {
      std::string const filename  = material_db_->getElementBlockParam<std::string>(eb_name, "ACE Air File");
      air_from_file_map_[eb_name] = vectorFromFile(filename);
      // IKT 2/17/2024: I am not sure if the following assert makes sense for the air.
      // TODO: check with Jenn.
      ALBANY_ASSERT(
          z_above_mean_sea_level_map_[eb_name].size() == air_from_file_map_[eb_name].size(),
          "*** ERROR: Number of z values and number of air values in "
          "ACE Air File must match. \n"
          "Hint: Did you provide the 'ACE Z Depth File'?");
    }

    ALBANY_ASSERT(
        time_map_[eb_name].size() == sea_level_map_[eb_name].size(),
        "*** ERROR: Number of times and number of sea level values must "
        "match.");
  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename EvalT::ScalarT
ACEThermalParameters<EvalT, Traits>::queryElementBlockParameterMap(std::string const eb_name, const std::map<std::string, RealType> map)
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
ACEThermalParameters<EvalT, Traits>::queryElementBlockParameterMap(std::string const eb_name, const std::map<std::string, std::vector<RealType>> map)
{
  typename std::map<std::string, std::vector<RealType>>::const_iterator it;
  it = map.find(eb_name);
  if (it == map.end()) {
    // Element block is not found in map - return std::vector of length 0
    std::vector<RealType> vec;
    vec.resize(0);
    return vec;
  }
  return it->second;
}
// **********************************************************************
}  // namespace LCM
