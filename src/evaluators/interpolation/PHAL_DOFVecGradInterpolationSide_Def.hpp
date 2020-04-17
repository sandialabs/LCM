// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_Macros.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

//**********************************************************************
template <typename EvalT, typename Traits, typename ScalarT>
DOFVecGradInterpolationSideBase<EvalT, Traits, ScalarT>::
    DOFVecGradInterpolationSideBase(
        Teuchos::ParameterList const&        p,
        const Teuchos::RCP<Albany::Layouts>& dl_side)
    : sideSetName(p.get<std::string>("Side Set Name")),
      val_node(p.get<std::string>("Variable Name"), dl_side->node_vector),
      gradBF(p.get<std::string>("Gradient BF Name"), dl_side->node_qp_gradient),
      grad_qp(
          p.get<std::string>("Gradient Variable Name"),
          dl_side->qp_vecgradient)
{
  ALBANY_PANIC(
      !dl_side->isSideLayouts,
      "Error! The layouts structure does not appear to be that of a side "
      "set.\n");

  this->addDependentField(val_node.fieldTag());
  this->addDependentField(gradBF.fieldTag());
  this->addEvaluatedField(grad_qp);

  this->setName("DOFVecGradInterpolationSideBase");

  numSideNodes = dl_side->node_qp_gradient->extent(2);
  numSideQPs   = dl_side->node_qp_gradient->extent(3);
  numDims      = dl_side->node_qp_gradient->extent(4);
  vecDim       = dl_side->node_vector->extent(3);
}

//**********************************************************************
template <typename EvalT, typename Traits, typename ScalarT>
void
DOFVecGradInterpolationSideBase<EvalT, Traits, ScalarT>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node, fm);
  this->utils.setFieldData(gradBF, fm);
  this->utils.setFieldData(grad_qp, fm);
}

// *********************************************************************************
template <typename EvalT, typename Traits, typename ScalarT>
void
DOFVecGradInterpolationSideBase<EvalT, Traits, ScalarT>::evaluateFields(
    typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName) == workset.sideSets->end()) return;

  std::vector<Albany::SideStruct> const& sideSet =
      workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet) {
    // Get the local data of side and cell
    int const cell = it_side.elem_LID;
    int const side = it_side.side_local_id;

    for (int qp = 0; qp < numSideQPs; ++qp) {
      for (int comp = 0; comp < vecDim; ++comp) {
        for (int dim = 0; dim < numDims; ++dim) {
          grad_qp(cell, side, qp, comp, dim) = 0.;
          for (int node = 0; node < numSideNodes; ++node) {
            grad_qp(cell, side, qp, comp, dim) +=
                val_node(cell, side, node, comp) *
                gradBF(cell, side, node, qp, dim);
          }
        }
      }
    }
  }
}

}  // Namespace PHAL
