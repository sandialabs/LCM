// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"
#include "PHAL_GatherScalarNodalParameter.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template <typename EvalT, typename Traits>
GatherScalarNodalParameterBase<EvalT, Traits>::GatherScalarNodalParameterBase(
    Teuchos::ParameterList const&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : numNodes(dl->node_scalar->extent(1)),
      param_name(p.get<std::string>("Parameter Name"))
{
  std::string field_name = p.isParameter("Field Name") ?
                               p.get<std::string>("Field Name") :
                               param_name;
  val = PHX::MDField<ParamScalarT, Cell, Node>(field_name, dl->node_scalar);

  this->addEvaluatedField(val);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
GatherScalarNodalParameterBase<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val, fm);
  d.fill_field_dependencies(
      this->dependentFields(),
      this->evaluatedFields(),
      d.memoizer_for_params_active());
}

// **********************************************************************
template <typename EvalT, typename Traits>
GatherScalarNodalParameter<EvalT, Traits>::GatherScalarNodalParameter(
    Teuchos::ParameterList const&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : GatherScalarNodalParameterBase<EvalT, Traits>(p, dl)
{
  this->setName(
      "GatherNodalParameter(" + this->param_name + ")" + PHX::print<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
GatherScalarNodalParameter<EvalT, Traits>::GatherScalarNodalParameter(
    Teuchos::ParameterList const& p)
    : GatherScalarNodalParameterBase<EvalT, Traits>(
          p,
          p.get<Teuchos::RCP<Albany::Layouts>>("Layouts Struct"))
{
  this->setName(
      "GatherNodalParameter(" + this->param_name + ")" + PHX::print<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
GatherScalarNodalParameter<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  Teuchos::RCP<Thyra_Vector const> pvec =
      workset.distParamLib->get(this->param_name)->overlapped_vector();
  Teuchos::ArrayRCP<const ST> pvec_constView = Albany::getLocalData(pvec);

  const Albany::IDArray& wsElDofs = workset.distParamLib->get(this->param_name)
                                        ->workset_elem_dofs()[workset.wsIndex];

  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const LO lid            = wsElDofs((int)cell, (int)node, 0);
      (this->val)(cell, node) = (lid >= 0) ? pvec_constView[lid] : 0;
    }
  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
GatherScalarExtruded2DNodalParameter<EvalT, Traits>::
    GatherScalarExtruded2DNodalParameter(
        Teuchos::ParameterList const&        p,
        const Teuchos::RCP<Albany::Layouts>& dl)
    : GatherScalarNodalParameterBase<EvalT, Traits>(p, dl),
      fieldLevel(p.get<int>("Field Level"))
{
  this->setName(
      "GatherScalarExtruded2DNodalParameter(" + this->param_name + ")" +
      PHX::print<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
GatherScalarExtruded2DNodalParameter<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // TODO: find a way to abstract away from the map concept. Perhaps using
  // Panzer::ConnManager?
  Teuchos::RCP<Thyra_Vector const> pvec =
      workset.distParamLib->get(this->param_name)->overlapped_vector();
  Teuchos::ArrayRCP<const ST> pvec_constView = Albany::getLocalData(pvec);

  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering =
      *workset.disc->getLayeredMeshNumbering();

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>& wsElNodeID =
      workset.disc->getWsElNodeID()[workset.wsIndex];

  auto overlapNodeVS   = workset.disc->getOverlapNodeVectorSpace();
  auto ov_node_indexer = Albany::createGlobalLocalIndexer(overlapNodeVS);
  auto pspace_indexer  = Albany::createGlobalLocalIndexer(pvec->space());
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const LO lnodeId = ov_node_indexer->getLocalElement(elNodeID[node]);
      LO       base_id, ilayer;
      layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
      const LO inode          = layeredMeshNumbering.getId(base_id, fieldLevel);
      const GO ginode         = ov_node_indexer->getGlobalElement(inode);
      const LO p_lid          = pspace_indexer->getLocalElement(ginode);
      (this->val)(cell, node) = (p_lid >= 0) ? pvec_constView[p_lid] : 0;
    }
  }
}

}  // namespace PHAL
