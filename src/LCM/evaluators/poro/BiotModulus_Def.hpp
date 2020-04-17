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
BiotModulus<EvalT, Traits>::BiotModulus(Teuchos::ParameterList& p)
    : biotModulus(
          p.get<std::string>("Biot Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* elmd_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  std::string type = elmd_list->get("Biot Modulus Type", "Constant");
  if (type == "Constant") {
    is_constant    = true;
    constant_value = elmd_list->get("Value", 1.0);

    // Add Biot Modulus as a Sacado-ized parameter
    this->registerSacadoParameter("Biot Modulus", paramLib);
  } else {
    ALBANY_ABORT("Invalid Biot modulus type " << type);
  }

  if (p.isType<std::string>("Porosity Name")) {
    Teuchos::RCP<PHX::DataLayout> scalar_dl =
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
    porosity =
        decltype(porosity)(p.get<std::string>("Porosity Name"), scalar_dl);
    this->addDependentField(porosity);
    isPoroElastic    = true;
    FluidBulkModulus = elmd_list->get("Fluid Bulk Modulus Value", 10.0e9);
    this->registerSacadoParameter(
        "Skeleton Bulk Modulus Parameter Value", paramLib);
    GrainBulkModulus = elmd_list->get(
        "Grain Bulk Modulus Value", 10.0e12);  // typically Kgrain >> Kskeleton
    this->registerSacadoParameter("Grain Bulk Modulus Value", paramLib);
  } else {
    isPoroElastic    = false;
    FluidBulkModulus = 10.0e9;   // temp value..need to change
    GrainBulkModulus = 10.0e12;  // temp value need to change
  }

  if (p.isType<std::string>("Biot Coefficient Name")) {
    Teuchos::RCP<PHX::DataLayout> scalar_dl =
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
    biotCoefficient = decltype(biotCoefficient)(
        p.get<std::string>("Biot Coefficient Name"), scalar_dl);
    this->addDependentField(biotCoefficient);
  }

  this->addEvaluatedField(biotModulus);
  this->setName("Biot Modulus" + PHX::print<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
BiotModulus<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(biotModulus, fm);
  if (!is_constant) this->utils.setFieldData(coordVec, fm);
  //  if (isThermoElastic) this->utils.setFieldData(Temperature,fm);
  if (isPoroElastic) this->utils.setFieldData(porosity, fm);
  if (isPoroElastic) this->utils.setFieldData(biotCoefficient, fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
BiotModulus<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  if (is_constant) {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        biotModulus(cell, qp) = constant_value;
      }
    }
  }
  if (isPoroElastic) {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        // 1/M = (B-phi)/Ks + phi/Kf
        biotModulus(cell, qp) =
            1 / ((biotCoefficient(cell, qp) - porosity(cell, qp)) /
                     GrainBulkModulus +
                 porosity(cell, qp) / FluidBulkModulus);
      }
    }
  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename BiotModulus<EvalT, Traits>::ScalarT&
BiotModulus<EvalT, Traits>::getValue(std::string const& n)
{
  if (n == "Biot Modulus")
    return constant_value;
  else if (n == "Fluid Bulk Modulus Value")
    return FluidBulkModulus;
  else if (n == "Grain Bulk Modulus Value")
    return GrainBulkModulus;
  ALBANY_ABORT(
      std::endl
      << "Error! Logic error in getting paramter " << n
      << " in BiotModulus::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}  // namespace LCM
