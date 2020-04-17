// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_ADVDIFFRESID_HPP
#define PHAL_ADVDIFFRESID_HPP

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
class AdvDiffResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                     public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  AdvDiffResid(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint>      wBF;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wGradBF;

  PHX::MDField<ScalarT const, Cell, QuadPoint, VecDim>      U;
  PHX::MDField<ScalarT const, Cell, QuadPoint, VecDim, Dim> UGrad;
  PHX::MDField<ScalarT const, Cell, QuadPoint, VecDim>      UDot;

  double mu;          // viscosity coefficient
  double a;           // advection coefficient
  double b;           // advection coefficient
  bool   useAugForm;  // use augmented form?
  int    formType;    // augmented form type

  // Output:
  PHX::MDField<ScalarT, Cell, Node, VecDim> Residual;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDim;
  bool        enableTransient;
};
}  // namespace PHAL

#endif
