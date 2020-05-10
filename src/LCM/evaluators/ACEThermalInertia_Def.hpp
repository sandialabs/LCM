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
ACEThermalInertia<EvalT, Traits>::ACEThermalInertia(Teuchos::ParameterList& p)
    : thermal_inertia(p.get<std::string>("QP Variable Name"),
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
  numQPs  = dims[1];
  numDims = dims[2];

  std::string ebName = p.get<std::string>("Element Block Name", "Missing");

  type = cond_list->get("ACEThermalInertia Type", "Constant");
  if (type == "Constant") {
    ScalarT value = cond_list->get("Value", 1.0);
    init_constant(value, p);

  }

  else if (type == "Block Dependent") {
    // We have a multiple material problem and need to map element blocks to
    // material data

    if (p.isType<Teuchos::RCP<Albany::MaterialDatabase>>("MaterialDB")) {
      materialDB = p.get<Teuchos::RCP<Albany::MaterialDatabase>>("MaterialDB");
    } else {
      ALBANY_ABORT(
          std::endl
          << "Error! Must specify a material database if using block "
             "dependent "
          << "thermal inertia" << std::endl);
    }

    // Get the sublist for thermal inertia for the element block in the mat
    // DB (the material in the elem block ebName.

    Teuchos::ParameterList& subList =
        materialDB->getElementBlockSublist(ebName, "ACEThermalInertia");

    std::string typ = subList.get("ACEThermalInertia Type", "Constant");

    if (typ == "Constant") {
      ScalarT value = subList.get("Value", 1.0);
      init_constant(value, p);
    }
  }  // Block dependent

  else {
    ALBANY_ABORT("Invalid thermal inertia type " << type);
  }

  this->addEvaluatedField(thermal_inertia);
  this->setName("ACEThermalInertia");
}

template <typename EvalT, typename Traits>
void
ACEThermalInertia<EvalT, Traits>::init_constant(
    ScalarT                 value,
    Teuchos::ParameterList& p)
{
  is_constant = true;

  constant_value = value;

  // Add thermal inertia as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  this->registerSacadoParameter("ACEThermalInertia", paramLib);

}  // init_constant

// **********************************************************************
template <typename EvalT, typename Traits>
void
ACEThermalInertia<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(thermal_inertia, fm);
  if (!is_constant) this->utils.setFieldData(coordVec, fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
ACEThermalInertia<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  if (is_constant) {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        thermal_inertia(cell, qp) = constant_value;
      }
    }
  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename ACEThermalInertia<EvalT, Traits>::ScalarT&
ACEThermalInertia<EvalT, Traits>::getValue(std::string const& n)
{
  if (is_constant) { return constant_value; }
  ALBANY_ABORT(
      std::endl
      << "Error! Logic error in getting parameter " << n
      << " in ACEThermalInertia::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
template <typename EvalT, typename Traits>
Teuchos::RCP<Teuchos::ParameterList const>
ACEThermalInertia<EvalT, Traits>::getValidThermalCondParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      rcp(new Teuchos::ParameterList("Valid ACEThermalInertia Params"));
  ;

  validPL->set<std::string>("ACEThermalInertia Type", "Constant",
      "Constant thermal inertia across the entire domain");
  validPL->set<double>("Value", 1.0, "Constant thermal inertia value");

  return validPL;
}

// **********************************************************************
// **********************************************************************
}  // namespace PHAL
