// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Albany_Macros.hpp"
#include "PHAL_SaveStateField.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"

namespace PHAL {

template <typename EvalT, typename Traits>
SaveStateField<EvalT, Traits>::SaveStateField(
    Teuchos::ParameterList const& /* p */)
{
  // States Not Saved for Generic Type, only Specializations
  this->setName("Save State Field");
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
SaveStateField<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData /* d */,
    PHX::FieldManager<Traits>& /* fm */)
{
  // States Not Saved for Generic Type, only Specializations
}

// **********************************************************************
template <typename EvalT, typename Traits>
void SaveStateField<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData /* workset */)
{
  // States Not Saved for Generic Type, only Specializations
}
// **********************************************************************
// **********************************************************************
template <typename Traits>
SaveStateField<PHAL::AlbanyTraits::Residual, Traits>::SaveStateField(
    Teuchos::ParameterList const& p)
{
  fieldName = p.get<std::string>("Field Name");
  stateName = p.get<std::string>("State Name");

  Teuchos::RCP<PHX::DataLayout> layout =
      p.get<Teuchos::RCP<PHX::DataLayout>>("State Field Layout");
  field = decltype(field)(fieldName, layout);

  if (layout->name(0) != "Cell" && layout->name(0) != "Node") {
    worksetState = true;
    nodalState   = false;
  } else {
    worksetState = false;
    nodalState =
        p.isParameter("Nodal State") ? p.get<bool>("Nodal State") : false;
  }

  Teuchos::RCP<PHX::DataLayout> dummy =
      Teuchos::rcp(new PHX::MDALayout<Dummy>(0));
  savestate_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dummy));

  this->addDependentField(field.fieldTag());
  this->addEvaluatedField(*savestate_operation);

  this->setName(
      "Save Field " + fieldName + " to State " + stateName + "Residual");
}

// **********************************************************************
template <typename Traits>
void
SaveStateField<PHAL::AlbanyTraits::Residual, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field, fm);

  if (nodalState) {
    ALBANY_PANIC(
        field.fieldTag().dataLayout().size() < 2,
        "Error! To save a nodal state, pass the cell-based version of it "
        "(<Cell,Node,...>).\n");
    ALBANY_PANIC(
        field.fieldTag().dataLayout().name(1) != "Node",
        "Error! To save a nodal state, the second tag of the layout MUST be "
        "'Node'.\n");
  }
  d.fill_field_dependencies(this->dependentFields(), this->evaluatedFields());
}
// **********************************************************************
template <typename Traits>
void
SaveStateField<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  if (this->nodalState)
    saveNodeState(workset);
  else if (this->worksetState)
    saveWorksetState(workset);
  else
    saveElemState(workset);
}

template <typename Traits>
void
SaveStateField<PHAL::AlbanyTraits::Residual, Traits>::saveElemState(
    typename Traits::EvalData workset)
{
  // Get shards Array (from STK) for this state
  // Need to check if we can just copy full size -- can assume same ordering?
  Albany::StateArray::const_iterator it;
  it = workset.stateArrayPtr->find(stateName);

  ALBANY_PANIC(
      (it == workset.stateArrayPtr->end()),
      std::endl
          << "Error: cannot locate " << stateName
          << " in PHAL_SaveStateField_Def" << std::endl);

  Albany::MDArray                         sta = it->second;
  std::vector<PHX::DataLayout::size_type> dims;
  sta.dimensions(dims);
  int size = dims.size();

  switch (size) {
    case 1:
      for (int cell = 0; cell < workset.numCells; ++cell)
        sta(cell) = field(cell);
      break;
    case 2:
      for (int cell = 0; cell < workset.numCells; ++cell)
        for (int qp = 0; qp < dims[1]; ++qp) sta(cell, qp) = field(cell, qp);
      ;
      break;
    case 3:
      for (int cell = 0; cell < workset.numCells; ++cell)
        for (int qp = 0; qp < dims[1]; ++qp)
          for (int i = 0; i < dims[2]; ++i)
            sta(cell, qp, i) = field(cell, qp, i);
      break;
    case 4:
      for (int cell = 0; cell < workset.numCells; ++cell)
        for (int qp = 0; qp < dims[1]; ++qp)
          for (int i = 0; i < dims[2]; ++i)
            for (int j = 0; j < dims[3]; ++j)
              sta(cell, qp, i, j) = field(cell, qp, i, j);
      break;
    case 5:
      for (int cell = 0; cell < workset.numCells; ++cell)
        for (int qp = 0; qp < dims[1]; ++qp)
          for (int i = 0; i < dims[2]; ++i)
            for (int j = 0; j < dims[3]; ++j)
              for (int k = 0; k < dims[4]; ++k)
                sta(cell, qp, i, j, k) = field(cell, qp, i, j, k);
      break;
    default:
      ALBANY_PANIC(
          size < 1 || size > 5,
          "Unexpected Array dimensions in SaveStateField: " << size);
  }
}

template <typename Traits>
void
SaveStateField<PHAL::AlbanyTraits::Residual, Traits>::saveWorksetState(
    typename Traits::EvalData workset)
{
  // Get shards Array (from STK) for this state
  // Need to check if we can just copy full size -- can assume same ordering?
  Albany::StateArray::const_iterator it;
  it = workset.stateArrayPtr->find(stateName);

  ALBANY_PANIC(
      (it == workset.stateArrayPtr->end()),
      std::endl
          << "Error: cannot locate " << stateName
          << " in PHAL_SaveStateField_Def" << std::endl);

  Albany::MDArray                         sta = it->second;
  std::vector<PHX::DataLayout::size_type> dims;
  sta.dimensions(dims);
  int size = dims.size();

  switch (size) {
    case 1:
      for (int cell = 0; cell < dims[0]; ++cell) sta(cell) = field(cell);
      break;
    case 2:
      for (int cell = 0; cell < dims[0]; ++cell)
        for (int qp = 0; qp < dims[1]; ++qp) sta(cell, qp) = field(cell, qp);
      ;
      break;
    default:
      ALBANY_PANIC(
          size < 1 || size > 5,
          "Unexpected (workset) Array dimensions in SaveStateField: " << size);
  }
}

template <typename Traits>
void
SaveStateField<PHAL::AlbanyTraits::Residual, Traits>::saveNodeState(
    typename Traits::EvalData workset)
{
  // Note: to save nodal fields, we need to open up the mesh, and work directly
  // on it.
  //       For this reason, we assume the mesh is stk. Moreover, since the cell
  //       buckets and node buckets do not coincide (elem-bucket 12 does not
  //       necessarily contain all nodes in node-bucket 12), we need to work
  //       with stk fields. To do this we must extract entities from the bulk
  //       data and use them to access the values of the stk field.

  Teuchos::RCP<Albany::AbstractDiscretization> disc = workset.disc;
  ALBANY_PANIC(
      disc == Teuchos::null,
      "Error! Discretization is needed to save nodal state.\n");

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> mesh =
      Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(
          disc->getMeshStruct());
  ALBANY_PANIC(
      mesh == Teuchos::null,
      "Error! Save nodal states available only for stk meshes.\n");

  stk::mesh::MetaData& metaData = *mesh->metaData;
  stk::mesh::BulkData& bulkData = *mesh->bulkData;

  const auto& wsElNodeID = disc->getWsElNodeID();

  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType SFT;
  typedef Albany::AbstractSTKFieldContainer::VectorFieldType VFT;

  SFT* scalar_field;
  VFT* vector_field;

  std::vector<PHX::DataLayout::size_type> dims;
  field.dimensions(dims);

  GO                nodeId;
  double*           values;
  stk::mesh::Entity e;
  switch (dims.size()) {
    case 2:  // node_scalar
      scalar_field =
          metaData.get_field<SFT>(stk::topology::NODE_RANK, stateName);
      ALBANY_PANIC(scalar_field == 0, "Error! Field not found.\n");
      for (int cell = 0; cell < workset.numCells; ++cell)
        for (int node = 0; node < dims[1]; ++node) {
          nodeId    = wsElNodeID[workset.wsIndex][cell][node];
          e         = bulkData.get_entity(stk::topology::NODE_RANK, nodeId + 1);
          values    = stk::mesh::field_data(*scalar_field, e);
          values[0] = field(cell, node);
        }
      break;
    case 3:  // node_vector
      vector_field =
          metaData.get_field<VFT>(stk::topology::NODE_RANK, stateName);
      ALBANY_PANIC(vector_field == 0, "Error! Field not found.\n");
      for (int cell = 0; cell < workset.numCells; ++cell)
        for (int node = 0; node < dims[1]; ++node) {
          nodeId = wsElNodeID[workset.wsIndex][cell][node];
          e      = bulkData.get_entity(stk::topology::NODE_RANK, nodeId + 1);
          values = stk::mesh::field_data(*vector_field, e);
          for (int i = 0; i < dims[2]; ++i) values[i] = field(cell, node, i);
        }
      break;
    default:  // error!
      ALBANY_ABORT(
          "Error! Unexpected field dimension (only node_scalar/node_vector for "
          "now).\n");
  }
}

}  // namespace PHAL
