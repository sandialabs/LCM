// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef TIME_HPP
#define TIME_HPP

#include "Albany_Types.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
/**
 * \brief Evaluates Time and the time step
 */

template <typename EvalT, typename Traits>
class Time : public PHX::EvaluatorWithBaseImpl<Traits>,
             public PHX::EvaluatorDerived<EvalT, Traits>,
             public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  Time(Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

  ScalarT&
  getValue(std::string const& n);

 private:
  //! Constant value
  PHX::MDField<ScalarT, Dummy> time;
  PHX::MDField<ScalarT, Dummy> deltaTime;
  ScalarT                      timeValue;

  bool        enableTransient;
  std::string timeName;
};
}  // namespace LCM

#endif
