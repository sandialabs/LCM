// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_NSMOMENTUMRESID_HPP
#define PHAL_NSMOMENTUMRESID_HPP

#include "PHAL_Dimension.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template <typename EvalT, typename Traits>
class NSMomentumResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                        public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;

  NSMomentumResid(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint>      wBF;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wGradBF;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>           pGrad;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim, Dim>      VGrad;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>           V;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                P;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>           Rm;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                TauM;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                mu;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                rho;

  // Output:
  PHX::MDField<ScalarT, Cell, Node, VecDim> MResidual;

  unsigned int numQPs, numDims, numNodes;
  bool         enableTransient;
  bool         haveSUPG;
};
}  // namespace PHAL

#endif
