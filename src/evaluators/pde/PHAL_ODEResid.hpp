// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_ODERESID_HPP
#define PHAL_ODERESID_HPP

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Teuchos_ParameterList.hpp"

namespace PHAL {

template <typename EvalT, typename Traits>
class ODEResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                 public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ODEResid(Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData ud);

 private:
  typedef typename EvalT::ScalarT ScalarT;

  // Input:
  PHX::MDField<ScalarT const, Cell, Node> X;
  PHX::MDField<ScalarT const, Cell, Node> X_dot;
  PHX::MDField<ScalarT const, Cell, Node> Y;
  PHX::MDField<ScalarT const, Cell, Node> Y_dot;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> Xoderesid;
  PHX::MDField<ScalarT, Cell, Node> Yoderesid;
};
}  // namespace PHAL

#endif
