// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef SCALAR_L2_PROJECTION_RESIDUAL_HPP
#define SCALAR_L2_PROJECTION_RESIDUAL_HPP

#include "Albany_Types.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
/** \brief Finite Element Interpolation Evaluator

    This evaluator computes residual of a scalar projection from Gauss points to
   nodes.

*/

template <typename EvalT, typename Traits>
class ScalarL2ProjectionResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                                   public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ScalarL2ProjectionResidual(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint>      wBF;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wGradBF;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim, Dim>      DefGrad;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                projectedStress;

  // Input for hydro-static stress effect
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim, Dim> Pstress;

  bool enableTransient;

  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  Kokkos::DynRankView<ScalarT, PHX::Device> tauH;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> TResidual;
};
}  // namespace LCM

#endif
