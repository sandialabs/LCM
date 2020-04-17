// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef DEFGRAD_HPP
#define DEFGRAD_HPP

#include "Albany_Types.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
/** \brief Deformation Gradient

 This evaluator computes the deformation gradient

 */

template <typename EvalT, typename Traits>
class DefGrad : public PHX::EvaluatorWithBaseImpl<Traits>,
                public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  DefGrad(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& fm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  // Input:
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim, Dim> GradU;
  PHX::MDField<const MeshScalarT, Cell, QuadPoint>       weights;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defgrad;
  PHX::MDField<ScalarT, Cell, QuadPoint>           J;

  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  //! flag to compute the weighted average of J
  bool weightedAverage;

  //! stabilization parameter for the weighted average
  ScalarT alpha;
};
}  // namespace LCM
#endif
