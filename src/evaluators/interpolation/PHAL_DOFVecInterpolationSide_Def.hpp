// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_Macros.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

//**********************************************************************
template <typename EvalT, typename Traits, typename Type>
DOFVecInterpolationSideBase<EvalT, Traits, Type>::DOFVecInterpolationSideBase(
    Teuchos::ParameterList const&        p,
    const Teuchos::RCP<Albany::Layouts>& dl_side)
    : sideSetName(p.get<std::string>("Side Set Name")),
      val_node(p.get<std::string>("Variable Name"), dl_side->node_vector),
      BF(p.get<std::string>("BF Name"), dl_side->node_qp_scalar),
      val_qp(p.get<std::string>("Variable Name"), dl_side->qp_vector)
{
  ALBANY_PANIC(
      !dl_side->isSideLayouts,
      "Error! The layouts structure does not appear to be that of a side "
      "set.\n");

  this->addDependentField(val_node.fieldTag());
  this->addDependentField(BF.fieldTag());
  this->addEvaluatedField(val_qp);

  this->setName("DOFVecInterpolationSide" + PHX::print<EvalT>());

  numSideNodes = dl_side->node_qp_scalar->extent(2);
  numSideQPs   = dl_side->node_qp_scalar->extent(3);
  vecDim       = dl_side->qp_vector->extent(3);
}

//**********************************************************************
template <typename EvalT, typename Traits, typename Type>
void
DOFVecInterpolationSideBase<EvalT, Traits, Type>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node, fm);
  this->utils.setFieldData(BF, fm);
  this->utils.setFieldData(val_qp, fm);
  d.fill_field_dependencies(this->dependentFields(), this->evaluatedFields());
}

//**********************************************************************
template <typename EvalT, typename Traits, typename Type>
void
DOFVecInterpolationSideBase<EvalT, Traits, Type>::evaluateFields(
    typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName) == workset.sideSets->end()) return;

  std::vector<Albany::SideStruct> const& sideSet =
      workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet) {
    // Get the local data of side and cell
    int const cell = it_side.elem_LID;
    int const side = it_side.side_local_id;

    for (int dim = 0; dim < vecDim; ++dim) {
      for (int qp = 0; qp < numSideQPs; ++qp) {
        val_qp(cell, side, qp, dim) =
            val_node(cell, side, 0, dim) * BF(cell, side, 0, qp);
        for (int node = 1; node < numSideNodes; ++node) {
          val_qp(cell, side, qp, dim) +=
              val_node(cell, side, node, dim) * BF(cell, side, node, qp);
        }
      }
    }
  }
}

}  // Namespace PHAL
