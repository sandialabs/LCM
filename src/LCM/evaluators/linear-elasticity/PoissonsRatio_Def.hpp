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
PoissonsRatio<EvalT, Traits>::PoissonsRatio(Teuchos::ParameterList& p)
    : poissonsRatio(
          p.get<std::string>("QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* pr_list = p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  Teuchos::RCP<PHX::DataLayout>           vector_dl = p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  std::string type = pr_list->get("Poissons Ratio Type", "Constant");
  if (type == "Constant") {
    is_constant    = true;
    constant_value = pr_list->get<double>("Value");

    // Add Poissons Ratio as a Sacado-ized parameter
    this->registerSacadoParameter("Poissons Ratio", paramLib);
  } else {
    ALBANY_ABORT("Invalid Poissons ratio type " << type);
  }

  // Optional dependence on Temperature (nu = nu_ + dnudT * T)
  // Switched ON by sending Temperature field in p

  if (p.isType<std::string>("QP Temperature Name")) {
    Teuchos::RCP<PHX::DataLayout> scalar_dl = p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
    Temperature = decltype(Temperature)(p.get<std::string>("QP Temperature Name"), scalar_dl);
    this->addDependentField(Temperature);
    isThermoElastic = true;
    dnudT_value     = pr_list->get("dnudT Value", 0.0);
    refTemp         = p.get<RealType>("Reference Temperature", 0.0);
    this->registerSacadoParameter("dnudT Value", paramLib);
  } else {
    isThermoElastic = false;
    dnudT_value     = 0.0;
  }

  this->addEvaluatedField(poissonsRatio);
  this->setName("Poissons Ratio" + PHX::print<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
PoissonsRatio<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(poissonsRatio, fm);
  if (!is_constant) this->utils.setFieldData(coordVec, fm);
  if (isThermoElastic) this->utils.setFieldData(Temperature, fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
PoissonsRatio<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  if (is_constant) {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        poissonsRatio(cell, qp) = constant_value;
      }
    }
  }
  if (isThermoElastic) {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        poissonsRatio(cell, qp) += dnudT_value * (Temperature(cell, qp) - refTemp);
      }
    }
  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename PoissonsRatio<EvalT, Traits>::ScalarT&
PoissonsRatio<EvalT, Traits>::getValue(std::string const& n)
{
  if (n == "Poissons Ratio")
    return constant_value;
  else if (n == "dnudT Value")
    return dnudT_value;
  ALBANY_ABORT(
      std::endl
      << "Error! Logic error in getting paramter " << n << " in PoissonsRatio::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}  // namespace LCM
