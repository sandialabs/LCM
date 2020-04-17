// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <string>
#include <vector>

#include "Albany_Macros.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

namespace PHAL {

template <typename EvalT, typename Traits>
SharedParameter<EvalT, Traits>::SharedParameter(Teuchos::ParameterList const& p)
{
  paramName  = p.get<std::string>("Parameter Name");
  paramValue = p.get<double>("Parameter Value");

  Teuchos::RCP<PHX::DataLayout> layout =
      p.get<Teuchos::RCP<PHX::DataLayout>>("Data Layout");

  //! Initialize field with same name as parameter
  PHX::MDField<ScalarT, Dim> f(paramName, layout);
  paramAsField = f;

  // Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib>>(
      "Parameter Library");  //, Teuchos::null ANDY - why a compiler error with
                             // this?
  this->registerSacadoParameter(paramName, paramLib);

  this->addEvaluatedField(paramAsField);
  this->setName("Shared Parameter");
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
SharedParameter<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(paramAsField, fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
SharedParameter<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  paramAsField(0) = paramValue;
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename SharedParameter<EvalT, Traits>::ScalarT&
SharedParameter<EvalT, Traits>::getValue(std::string const& n)
{
  ALBANY_PANIC(n != paramName);
  return paramValue;
}

// **********************************************************************

template <typename EvalT, typename Traits>
SharedParameterVec<EvalT, Traits>::SharedParameterVec(
    Teuchos::ParameterList const& p)
{
  Teuchos::RCP<PHX::DataLayout> layout =
      p.get<Teuchos::RCP<PHX::DataLayout>>("Data Layout");
  Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib>>(
      "Parameter Library");  //, Teuchos::null ANDY - why a compiler error with
                             // this?

  numParams = layout->extent(1);

  paramNames.resize(numParams);
  paramValues.resize(numParams);
  paramNames  = p.get<Teuchos::Array<std::string>>("Parameters Names");
  paramValues = p.get<Teuchos::Array<ScalarT>>("Parameters Values");

  ALBANY_PANIC(
      paramNames.size() == numParams,
      "Error! The array of names' size does not match the layout first "
      "dimension.\n");
  ALBANY_PANIC(
      paramValues.size() == numParams,
      "Error! The array of values' size does not match the layout first "
      "dimension.\n");

  std::string paramVecName = p.get<std::string>("Parameter Vector Name");

  this->registerSacadoParameter(paramVecName, paramLib);

  this->addEvaluatedField(paramAsField);

  this->setName("Shared Parameter Vector");
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
SharedParameterVec<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(paramAsField, fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
SharedParameterVec<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  for (int i = 0; i < numParams; ++i) paramAsField(i) = paramValues[i];
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename SharedParameterVec<EvalT, Traits>::ScalarT&
SharedParameterVec<EvalT, Traits>::getValue(std::string const& n)
{
  for (int i = 0; i < numParams; ++i)
    if (n == paramNames[i]) return paramValues[i];

  ALBANY_ABORT("Error! Parameter name not found.\n");

  // To avoid warnings
  static ScalarT dummy;
  return dummy;
}

// **********************************************************************
}  // namespace PHAL
