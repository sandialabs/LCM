// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_DUMMY_RESIDUAL_HPP
#define PHAL_DUMMY_RESIDUAL_HPP

#include "Albany_Layouts.hpp"
#include "Albany_Types.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace PHAL {

template <typename EvalT, typename Traits>
class DummyResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;

  DummyResidual(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& fm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  // Input:
  PHX::MDField<ScalarT const, Cell, Node> solution;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> residual;
};

}  // Namespace PHAL

#endif  // PHAL_DUMMY_RESIDUAL_HPP
