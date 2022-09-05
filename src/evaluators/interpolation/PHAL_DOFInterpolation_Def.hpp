// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_Macros.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Workset.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

//*****
template <typename EvalT, typename Traits, typename ScalarT>
DOFInterpolationBase<EvalT, Traits, ScalarT>::DOFInterpolationBase(
    Teuchos::ParameterList const&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : val_node(p.get<std::string>("Variable Name"), dl->node_scalar),
      BF(p.get<std::string>("BF Name"), dl->node_qp_scalar),
      val_qp(p.get<std::string>("Variable Name"), dl->qp_scalar)
{
  this->addDependentField(val_node.fieldTag());
  this->addDependentField(BF.fieldTag());
  this->addEvaluatedField(val_qp);

  this->setName("DOFInterpolationBase" + PHX::print<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
}

//*****
template <typename EvalT, typename Traits, typename ScalarT>
void
DOFInterpolationBase<EvalT, Traits, ScalarT>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node, fm);
  this->utils.setFieldData(BF, fm);
  this->utils.setFieldData(val_qp, fm);

  d.fill_field_dependencies(this->dependentFields(), this->evaluatedFields());
}

// *********************************************************************
// Kokkos functor
template <typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION void
DOFInterpolationBase<EvalT, Traits, ScalarT>::operator()(const DOFInterpolationBase_Tag& tag, const int& cell) const
{
  for (int qp = 0; qp < numQPs; ++qp) {
    val_qp(cell, qp) = val_node(cell, 0) * BF(cell, 0, qp);
    for (int node = 1; node < numNodes; ++node) {
      val_qp(cell, qp) += val_node(cell, node) * BF(cell, node, qp);
    }
  }
}

//*****
template <typename EvalT, typename Traits, typename ScalarT>
void
DOFInterpolationBase<EvalT, Traits, ScalarT>::evaluateFields(typename Traits::EvalData workset)
{
  Kokkos::parallel_for(DOFInterpolationBase_Policy(0, workset.numCells), *this);
}

}  // namespace PHAL
