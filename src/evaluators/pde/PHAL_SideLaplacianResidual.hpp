// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license
// detailed in the file license.txt in the top-level Albany directory.

#ifndef PHAL_SIDE_LAPLACIAN_RESIDUAL_HPP
#define PHAL_SIDE_LAPLACIAN_RESIDUAL_HPP 1

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "PHAL_Dimension.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace PHAL {

template <typename EvalT, typename Traits>
class SideLaplacianResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;

  SideLaplacianResidual(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& fm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  void
  evaluateFieldsCell(typename Traits::EvalData d);
  void
  evaluateFieldsSide(typename Traits::EvalData d);

  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<RealType>    BF;
  PHX::MDField<MeshScalarT> GradBF;
  PHX::MDField<MeshScalarT> w_measure;
  PHX::MDField<MeshScalarT, Cell, Side, QuadPoint, Dim, Dim>
      metric;  // Only used in 2D, so we know the layout

  PHX::MDField<ScalarT> u;
  PHX::MDField<ScalarT> grad_u;

  // Output:
  PHX::MDField<ScalarT, Cell, Node>
      residual;  // Always a 3D residual, so we know the layout

  std::string                   sideSetName;
  std::vector<std::vector<int>> sideNodes;

  int spaceDim;
  int gradDim;
  int numNodes;
  int numQPs;

  bool sideSetEquation;
};

}  // Namespace PHAL

#endif  // PHAL_SIDE_LAPLACIAN_RESIDUAL_HPP
