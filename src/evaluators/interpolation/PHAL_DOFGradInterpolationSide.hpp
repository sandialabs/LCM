// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license
// detailed in the file license.txt in the top-level Albany directory.

#ifndef PHAL_DOF_GRAD_INTERPOLATION_SIDE_HPP
#define PHAL_DOF_GRAD_INTERPOLATION_SIDE_HPP 1

#include "Albany_Layouts.hpp"
#include "Albany_SacadoTypes.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Utilities.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace PHAL {
/** \brief Finite Element InterpolationSide Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template <typename EvalT, typename Traits, typename ScalarT>
class DOFGradInterpolationSideBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                     public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  DOFGradInterpolationSideBase(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl_side);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename Albany::StrongestScalarType<ScalarT, MeshScalarT>::type
      OutputScalarT;

  std::string sideSetName;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT const, Cell, Side, Node> val_node;
  //! Basis Functions
  PHX::MDField<const MeshScalarT, Cell, Side, Node, QuadPoint, Dim> gradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<OutputScalarT, Cell, Side, QuadPoint, Dim> grad_qp;

  int numSideNodes;
  int numSideQPs;
  int numDims;
};

// Some shortcut names
template <typename EvalT, typename Traits>
using DOFGradInterpolationSide =
    DOFGradInterpolationSideBase<EvalT, Traits, typename EvalT::ScalarT>;

template <typename EvalT, typename Traits>
using DOFGradInterpolationSideMesh =
    DOFGradInterpolationSideBase<EvalT, Traits, typename EvalT::MeshScalarT>;

template <typename EvalT, typename Traits>
using DOFGradInterpolationSideParam =
    DOFGradInterpolationSideBase<EvalT, Traits, typename EvalT::ParamScalarT>;

}  // Namespace PHAL

#endif  // PHAL_DOF_GRAD_INTERPOLATION_SIDE_HPP
