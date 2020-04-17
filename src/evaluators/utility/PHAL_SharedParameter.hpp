// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_SHARED_PARAMETER_HPP
#define PHAL_SHARED_PARAMETER_HPP

#include "Albany_Types.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace PHAL {
/** \brief SharedParameter

*/

template <typename EvalT, typename Traits>
class SharedParameter : public PHX::EvaluatorWithBaseImpl<Traits>,
                        public PHX::EvaluatorDerived<EvalT, Traits>,
                        public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;

  SharedParameter(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

  ScalarT&
  getValue(std::string const& n);

 private:
  std::string                paramName;
  ScalarT                    paramValue;
  PHX::MDField<ScalarT, Dim> paramAsField;
};

template <typename EvalT, typename Traits>
class SharedParameterVec : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>,
                           public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;

  SharedParameterVec(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

  ScalarT&
  getValue(std::string const& n);

 private:
  int                         numParams;
  Teuchos::Array<std::string> paramNames;
  Teuchos::Array<ScalarT>     paramValues;
  PHX::MDField<ScalarT, Dim>  paramAsField;
};

}  // namespace PHAL

#endif  // PHAL_SHARED_PARAMETER_HPP
