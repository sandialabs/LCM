// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_NSPERMEABILITYTERM_HPP
#define PHAL_NSPERMEABILITYTERM_HPP

#include "PHAL_Dimension.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/
namespace PHAL {

template <typename EvalT, typename Traits>
class NSPermeabilityTerm : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;

  NSPermeabilityTerm(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim> V;
  PHX::MDField<ScalarT const, Cell, QuadPoint>      mu;
  PHX::MDField<ScalarT const, Cell, QuadPoint>      phi;
  PHX::MDField<ScalarT const, Cell, QuadPoint>      K;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim> permTerm;

  unsigned int numQPs, numDims, numNodes;
  bool         enableTransient;
  bool         haveHeat;
};
}  // namespace PHAL

#endif
