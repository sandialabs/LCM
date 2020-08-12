// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <fstream>

#include "Albany_Macros.hpp"
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

// Radom field types
/*
enum SG_RF {CONSTANT, UNIFORM, LOGNORMAL};
int const num_sg_rf = 3;
const SG_RF sg_rf_values[] = {CONSTANT, UNIFORM, LOGNORMAL};
char const *sg_rf_names[] = {"Constant", "Uniform", "Log-Normal"};

SG_RF randField = CONSTANT;
*/

namespace PHAL {

template <typename EvalT, typename Traits>
ThermalConductivity<EvalT, Traits>::ThermalConductivity(Teuchos::ParameterList& p)
    : thermalCond(p.get<std::string>("QP Variable Name"), p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  randField = CONSTANT;

  Teuchos::ParameterList* cond_list = p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<Teuchos::ParameterList const> reflist = this->getValidThermalCondParameters();

  // Check the parameters contained in the input file. Do not check the defaults
  // set programmatically
  cond_list->validateParameters(*reflist, 0, Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  Teuchos::RCP<PHX::DataLayout>           vector_dl = p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  std::string ebName = p.get<std::string>("Element Block Name", "Missing");

  type = cond_list->get("ThermalConductivity Type", "Constant");

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
          << "thermal conductivity" << std::endl);
    }

    // Get the sublist for thermal conductivity for the element block in the mat
    // DB (the material in the elem block ebName.

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "ThermalConductivity");

    std::string typ = subList.get("ThermalConductivity Type", "Constant");

    if (typ == "Constant") {
      ScalarT value = subList.get("Value", 1.0);
      init_constant(value, p);
    }
  }  // Block dependent

  else {
    ALBANY_ABORT("Invalid thermal conductivity type " << type);
  }

  this->addEvaluatedField(thermalCond);
  this->setName("ThermalConductivity");
}

template <typename EvalT, typename Traits>
void
ThermalConductivity<EvalT, Traits>::init_constant(ScalarT value, Teuchos::ParameterList& p)
{
  is_constant = true;
  randField   = CONSTANT;

  constant_value = value;

  // Add thermal conductivity as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  this->registerSacadoParameter("ThermalConductivity", paramLib);

}  // init_constant

// **********************************************************************
template <typename EvalT, typename Traits>
void
ThermalConductivity<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(thermalCond, fm);
  if (!is_constant) this->utils.setFieldData(coordVec, fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
ThermalConductivity<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  if (is_constant) {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        thermalCond(cell, qp) = constant_value;
      }
    }
  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename ThermalConductivity<EvalT, Traits>::ScalarT&
ThermalConductivity<EvalT, Traits>::getValue(std::string const& n)
{
  if (is_constant) {
    return constant_value;
  }
  ALBANY_ABORT(
      std::endl
      << "Error! Logic error in getting paramter " << n << " in ThermalConductivity::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
template <typename EvalT, typename Traits>
Teuchos::RCP<Teuchos::ParameterList const>
ThermalConductivity<EvalT, Traits>::getValidThermalCondParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("Valid ThermalConductivity Params"));
  ;

  validPL->set<std::string>(
      "ThermalConductivity Type", "Constant", "Constant thermal conductivity across the entire domain");
  validPL->set<double>("Value", 1.0, "Constant thermal conductivity value");

  // Truncated KL parameters

  validPL->set<int>("Number of KL Terms", 2, "");
  validPL->set<double>("Mean", 0.2, "");
  validPL->set<double>("Standard Deviation", 0.1, "");
  validPL->set<std::string>("Domain Lower Bounds", "{0.0 0.0}", "");
  validPL->set<std::string>("Domain Upper Bounds", "{1.0 1.0}", "");
  validPL->set<std::string>("Correlation Lengths", "{1.0 1.0}", "");
  return validPL;
}

// **********************************************************************
// **********************************************************************
}  // namespace PHAL
