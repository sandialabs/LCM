// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_DOF_VEC_INTERPOLATION_SIDE_HPP
#define PHAL_DOF_VEC_INTERPOLATION_SIDE_HPP 1

#include "Albany_DataTypes.hpp"
#include "Albany_Layouts.hpp"
#include "PHAL_Dimension.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace PHAL {
/** \brief Finite Element InterpolationSide Evaluator

    This evaluator interpolates nodal DOF vector values to quad points.

*/

template <typename EvalT, typename Traits, typename ScalarT>
class DOFVecInterpolationSideBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                    public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  DOFVecInterpolationSideBase(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef ScalarT ParamScalarT;

  std::string sideSetName;

  // Input:
  //! Values at nodes
  PHX::MDField<const ParamScalarT, Cell, Side, Node, Dim> val_node;
  //! Basis Functions
  PHX::MDField<const RealType, Cell, Side, Node, QuadPoint> BF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ParamScalarT, Side, Cell, QuadPoint, Dim> val_qp;

  int numSideNodes;
  int numSideQPs;
  int vecDim;
};

// Some shortcut names
template <typename EvalT, typename Traits>
using DOFVecInterpolationSide =
    DOFVecInterpolationSideBase<EvalT, Traits, typename EvalT::ScalarT>;

template <typename EvalT, typename Traits>
using DOFVecInterpolationSideMesh =
    DOFVecInterpolationSideBase<EvalT, Traits, typename EvalT::MeshScalarT>;

template <typename EvalT, typename Traits>
using DOFVecInterpolationSideParam =
    DOFVecInterpolationSideBase<EvalT, Traits, typename EvalT::ParamScalarT>;

}  // Namespace PHAL

#endif  // PHAL_DOF_VEC_INTERPOLATION_SIDE_HPP
