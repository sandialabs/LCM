// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"
#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "PHAL_Utilities.hpp"
#include "Phalanx_DataLayout.hpp"

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace PHAL {

template <typename EvalT, typename Traits>
SeparableScatterScalarResponseBase<EvalT, Traits>::SeparableScatterScalarResponseBase(Teuchos::ParameterList const& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  setup(p, dl);
}

template <typename EvalT, typename Traits>
void
SeparableScatterScalarResponseBase<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData /* d */, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(local_response, fm);
  if (!this->stand_alone) {
    this->utils.setFieldData(local_response_eval, fm);
  }
}

template <typename EvalT, typename Traits>
void
SeparableScatterScalarResponseBase<EvalT, Traits>::setup(Teuchos::ParameterList const& p, const Teuchos::RCP<Albany::Layouts>& /* dl */)
{
  this->stand_alone = p.get<bool>("Stand-alone Evaluator");

  // Setup fields we require
  auto local_response_tag = p.get<PHX::Tag<ScalarT>>("Local Response Field Tag");
  local_response          = decltype(local_response)(local_response_tag);
  if (this->stand_alone) {
    this->addDependentField(local_response);
  } else {
    local_response_eval = decltype(local_response_eval)(local_response_tag);
    this->addEvaluatedField(local_response_eval);
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template <typename Traits>
SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian, Traits>::SeparableScatterScalarResponse(
    Teuchos::ParameterList const&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setup(p, dl);
}

template <typename Traits>
void
SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian, Traits>::preEvaluate(typename Traits::PreEvalData workset)
{
  // Initialize derivatives
  Teuchos::RCP<Thyra_MultiVector> dgdx            = workset.dgdx;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdx = workset.overlapped_dgdx;
  if (dgdx != Teuchos::null) {
    dgdx->assign(0.0);
    overlapped_dgdx->assign(0.0);
  }

  Teuchos::RCP<Thyra_MultiVector> dgdxdot            = workset.dgdxdot;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdxdot = workset.overlapped_dgdxdot;
  if (dgdxdot != Teuchos::null) {
    dgdxdot->assign(0.0);
    overlapped_dgdxdot->assign(0.0);
  }
}

template <typename Traits>
void
SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  // Here we scatter the *local* response derivative
  auto                            nodeID  = workset.wsElNodeEqID;
  Teuchos::RCP<Thyra_MultiVector> dgdx    = workset.overlapped_dgdx;
  Teuchos::RCP<Thyra_MultiVector> dgdxdot = workset.overlapped_dgdxdot;
  Teuchos::RCP<Thyra_MultiVector> dg;
  if (dgdx != Teuchos::null) {
    dg = dgdx;
  } else {
    dg = dgdxdot;
  }

  auto dg_data = Albany::getNonconstLocalData(dg);

  // Loop over cells in workset
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    // Loop over responses

    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      auto val = this->local_response(cell, res);

      // Loop over nodes in cell
      for (unsigned int node_dof = 0; node_dof < numNodes; node_dof++) {
        int neq = nodeID.extent(2);

        // Loop over equations per node
        for (unsigned int eq_dof = 0; eq_dof < neq; eq_dof++) {
          // local derivative component
          int deriv = neq * node_dof + eq_dof;

          // local DOF
          int dof = nodeID(cell, node_dof, eq_dof);

          // Set dg/dx
          // NOTE: mv local data is in column major
          dg_data[res][dof] += val.dx(deriv);

        }  // column equations
      }  // column nodes
    }  // response

  }  // cell
}

template <typename Traits>
void
SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian, Traits>::evaluate2DFieldsDerivativesDueToExtrudedSolution(
    typename Traits::EvalData            workset,
    std::string&                         sideset,
    Teuchos::RCP<const CellTopologyData> cellTopo)
{
  // Here we scatter the *local* response derivative
  Teuchos::RCP<Thyra_MultiVector> dgdx    = workset.overlapped_dgdx;
  Teuchos::RCP<Thyra_MultiVector> dgdxdot = workset.overlapped_dgdxdot;
  Teuchos::RCP<Thyra_MultiVector> dg;
  if (dgdx != Teuchos::null) {
    dg = dgdx;
  } else {
    dg = dgdxdot;
  }
  auto dg_data = Albany::getNonconstLocalData(dg);

  int const                                       neq                  = workset.wsElNodeEqID.extent(2);
  const Albany::NodalDOFManager&                  solDOFManager        = workset.disc->getOverlapDOFManager("ordinary_solution");
  const Albany::LayeredMeshNumbering<LO>&         layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  int                                             numLayers            = layeredMeshNumbering.numLayers;
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>& wsElNodeID           = workset.disc->getWsElNodeID()[workset.wsIndex];

  if (workset.sideSets == Teuchos::null) ALBANY_ABORT("Side sets not properly specified on the mesh" << std::endl);

  const Albany::SideSetList&          ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it     = ssList.find(sideset);

  if (it != ssList.end()) {
    std::vector<Albany::SideStruct> const& sideSet = it->second;

    auto overlapNodeVS   = workset.disc->getOverlapNodeVectorSpace();
    auto ov_node_indexer = Albany::createGlobalLocalIndexer(overlapNodeVS);

    for (std::size_t iSide = 0; iSide < sideSet.size(); ++iSide) {  // loop over the sides on this ws and name
      // Get the data that corresponds to the side
      int const                       elem_LID     = sideSet[iSide].elem_LID;
      int const                       elem_side    = sideSet[iSide].side_local_id;
      const CellTopologyData_Subcell& side         = cellTopo->side[elem_side];
      int                             numSideNodes = side.topology->node_count;

      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];
      for (std::size_t res = 0; res < this->global_response.size(); res++) {
        auto val = this->local_response(elem_LID, res);
        LO   base_id, ilayer;
        for (int i = 0; i < numSideNodes; ++i) {
          std::size_t node    = side.node[i];
          const LO    lnodeId = ov_node_indexer->getLocalElement(elNodeID[node]);
          layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
          for (unsigned int il_col = 0; il_col < numLayers + 1; il_col++) {
            const LO inode = layeredMeshNumbering.getId(base_id, il_col);
            for (unsigned int eq_col = 0; eq_col < neq; eq_col++) {
              const LO dof   = solDOFManager.getLocalDOF(inode, eq_col);
              int      deriv = neq * this->numNodes + il_col * neq * numSideNodes + neq * i + eq_col;
              dg_data[res][dof] += val.dx(deriv);
            }
          }
        }
      }
    }
  }
}

template <typename Traits>
void
SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian, Traits>::postEvaluate(typename Traits::PostEvalData workset)
{
  // Here we scatter the *global* response
  Teuchos::RCP<Thyra_Vector> g = workset.g;
  if (g != Teuchos::null) {
    Teuchos::ArrayRCP<ST> g_nonconstView = Albany::getNonconstLocalData(g);
    for (PHAL::MDFieldIterator<ScalarT const> gr(this->global_response); !gr.done(); ++gr) g_nonconstView[gr.idx()] = gr.ref().val();
  }

  // Here we scatter the *global* response derivatives
  Teuchos::RCP<Thyra_MultiVector> dgdx            = workset.dgdx;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdx = workset.overlapped_dgdx;
  if (dgdx != Teuchos::null) {
    workset.x_cas_manager->combine(overlapped_dgdx, dgdx, Albany::CombineMode::ADD);
  }

  Teuchos::RCP<Thyra_MultiVector> dgdxdot            = workset.dgdxdot;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdxdot = workset.overlapped_dgdxdot;
  if (dgdxdot != Teuchos::null) {
    workset.x_cas_manager->combine(overlapped_dgdxdot, dgdxdot, Albany::CombineMode::ADD);
  }
}

}  // namespace PHAL
