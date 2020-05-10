// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <fstream>

#include "Albany_Macros.hpp"
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
ACEThermalConductivity<EvalT, Traits>::ACEThermalConductivity(Teuchos::ParameterList& p)
    : thermal_conductivity_(p.get<std::string>("QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
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
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  num_qps_  = dims[1];
  num_dims_ = dims[2];

  std::string eb_name = p.get<std::string>("Element Block Name", "Missing");

  type_ = cond_list->get("ACE Thermal Conductivity Type", "Constant");
  if (type_ == "Constant") {
    ScalarT value = cond_list->get("Value", 1.0);
    init_constant(value, p);

  }

  else if (type_ == "Block Dependent") {
    // We have a multiple material problem and need to map element blocks to
    // material data

    if (p.isType<Teuchos::RCP<Albany::MaterialDatabase>>("MaterialDB")) {
      material_db_ = p.get<Teuchos::RCP<Albany::MaterialDatabase>>("MaterialDB");
    } else {
      ALBANY_ABORT(
          std::endl
          << "Error! Must specify a material database if using block "
             "dependent "
          << "thermal conductivity" << std::endl);
    }

    // Get the sublist for thermal conductivity for the element block in the mat
    // DB (the material in the elem block eb_name.

    Teuchos::ParameterList& subList =
        material_db_->getElementBlockSublist(eb_name, "ACE Thermal Conductivity");

    std::string typ = subList.get("ACE Thermal Conductivity Type", "Constant");

    if (typ == "Constant") {
      ScalarT value = subList.get("Value", 1.0);
      init_constant(value, p);
    }
  }  // Block dependent

  else {
    ALBANY_ABORT("Invalid thermal conductivity type " << type_);
  }

  this->addEvaluatedField(thermal_conductivity_);
  this->setName("ACE Thermal Conductivity");
}

template <typename EvalT, typename Traits>
void
ACEThermalConductivity<EvalT, Traits>::init_constant(
    ScalarT                 value,
    Teuchos::ParameterList& p)
{
  is_constant = true;

  constant_value_ = value;

  // Add thermal conductivity as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> param_lib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  this->registerSacadoParameter("ACE Thermal Conductivity", param_lib);

}  // init_constant

// **********************************************************************
template <typename EvalT, typename Traits>
void
ACEThermalConductivity<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(thermal_conductivity_, fm);
  if (!is_constant) this->utils.setFieldData(coord_vec_, fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
ACEThermalConductivity<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  if (is_constant) {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < num_qps_; ++qp) {
        thermal_conductivity_(cell, qp) = constant_value_;
      }
    }
  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename ACEThermalConductivity<EvalT, Traits>::ScalarT&
ACEThermalConductivity<EvalT, Traits>::getValue(std::string const& n)
{
  if (is_constant) { return constant_value_; }
  ALBANY_ABORT(
      std::endl
      << "Error! Logic error in getting parameter " << n
      << " in ACE Thermal Conductivity::getValue()" << std::endl);
  return constant_value_;
}

// **********************************************************************
template <typename EvalT, typename Traits>
Teuchos::RCP<Teuchos::ParameterList const>
ACEThermalConductivity<EvalT, Traits>::getValidThermalCondParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> valid_pl =
      rcp(new Teuchos::ParameterList("Valid ACE Thermal Conductivity Params"));
  ;

  valid_pl->set<std::string>("ACE Thermal Conductivity Type", "Constant",
      "Constant thermal conductivity across the entire domain");
  valid_pl->set<double>("Value", 1.0, "Constant thermal conductivity value");

  return valid_pl;
}

// **********************************************************************
// **********************************************************************
}  // namespace PHAL
