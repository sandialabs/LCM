// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Albany_Macros.hpp"
#include "Albany_SideSetSTKMeshStruct.hpp"
#include "PHAL_SaveSideSetStateField.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template <typename EvalT, typename Traits>
SaveSideSetStateField<EvalT, Traits>::SaveSideSetStateField(Teuchos::ParameterList const& /* p */, const Teuchos::RCP<Albany::Layouts>& /* dl */)
{
  // States Not Saved for Generic Type, only Specializations
  this->setName("Save Side Set State Field" + PHX::print<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
SaveSideSetStateField<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData /* d */, PHX::FieldManager<Traits>& /* fm */)
{
  // States Not Saved for Generic Type, only Specializations
}

// **********************************************************************
template <typename EvalT, typename Traits>
void SaveSideSetStateField<EvalT, Traits>::evaluateFields(typename Traits::EvalData /* workset */)
{
  // States Not Saved for Generic Type, only Specializations
}
// **********************************************************************

// **********************************************************************
template <typename Traits>
SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::SaveSideSetStateField(Teuchos::ParameterList const& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  sideSetName = p.get<std::string>("Side Set Name");

  fieldName = p.get<std::string>("Field Name");
  stateName = p.get<std::string>("State Name");

  field = decltype(field)(fieldName, p.get<Teuchos::RCP<PHX::DataLayout>>("Field Layout"));

  nodalState = p.isParameter("Nodal State") ? p.get<bool>("Nodal State") : false;

  savestate_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));

  this->addDependentField(field.fieldTag());
  this->addEvaluatedField(*savestate_operation);

  if (nodalState) {
    ALBANY_PANIC(
        field.fieldTag().dataLayout().size() < 3,
        "Error! To save a side-set nodal state, pass the cell-side-based "
        "version of it (<Cell,Side,Node,...>).\n");
    ALBANY_PANIC(
        field.fieldTag().dataLayout().name(2) != "Node",
        "Error! To save a side-set nodal state, the third tag of the layout "
        "MUST be 'Node'.\n");

    Teuchos::RCP<shards::CellTopology> cellType;
    cellType = p.get<Teuchos::RCP<shards::CellTopology>>("Cell Type");

    int numSides = dl->cell_gradient->extent(1);
    int sideDim  = cellType->getDimension() - 1;
    sideNodes.resize(numSides);
    for (int side = 0; side < numSides; ++side) {
      // Need to get the subcell exact count, since different sides may have
      // different number of nodes (e.g., Wedge)
      int thisSideNodes = cellType->getNodeCount(sideDim, side);
      sideNodes[side].resize(thisSideNodes);
      for (int node = 0; node < thisSideNodes; ++node) {
        sideNodes[side][node] = cellType->getNodeMap(sideDim, side, node);
      }
    }
  }

  this->setName("Save Side Set Field " + fieldName + " to Side Set State " + stateName + " <Residual>");
}

// **********************************************************************
template <typename Traits>
void
SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::postRegistrationSetup(typename Traits::SetupData /* d */, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field, fm);
}
// **********************************************************************
template <typename Traits>
void
SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  if (this->nodalState)
    saveNodeState(workset);
  else
    saveElemState(workset);
}

template <typename Traits>
void
SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::saveElemState(typename Traits::EvalData workset)
{
  ALBANY_PANIC(workset.sideSets == Teuchos::null, "Error! The mesh does not store any side set.\n");

  if (workset.sideSets->find(sideSetName) == workset.sideSets->end()) return;  // Side set not present in this workset

  ALBANY_PANIC(workset.disc == Teuchos::null, "Error! The workset must store a valid discretization pointer.\n");

  const Albany::AbstractDiscretization::SideSetDiscretizationsType& ssDiscs = workset.disc->getSideSetDiscretizations();

  ALBANY_PANIC(ssDiscs.size() == 0, "Error! The discretization must store side set discretizations.\n");

  ALBANY_PANIC(ssDiscs.find(sideSetName) == ssDiscs.end(), "Error! No discretization found for side set " << sideSetName << ".\n");

  Teuchos::RCP<Albany::AbstractDiscretization> ss_disc = ssDiscs.at(sideSetName);

  ALBANY_PANIC(ss_disc == Teuchos::null, "Error! Side discretization is invalid for side set " << sideSetName << ".\n");

  std::map<std::string, std::map<GO, GO>> const& ss_maps = workset.disc->getSideToSideSetCellMap();

  ALBANY_PANIC(
      ss_maps.find(sideSetName) == ss_maps.end(),
      "Error! Something is off: the mesh has side discretization but no "
      "sideId-to-sideSetElemId map.\n");

  std::map<GO, GO> const& ss_map = ss_maps.at(sideSetName);

  // Get states from STK mesh
  Albany::StateArrays&   state_arrays = ss_disc->getStateArrays();
  Albany::StateArrayVec& esa          = state_arrays.elemStateArrays;
  Albany::WsLIDList&     elemGIDws3D  = workset.disc->getElemGIDws();
  Albany::WsLIDList&     elemGIDws2D  = ss_disc->getElemGIDws();

  // Get side_node->side_set_cell_node map from discretization
  ALBANY_PANIC(
      workset.disc->getSideNodeNumerationMap().find(sideSetName) == workset.disc->getSideNodeNumerationMap().end(),
      "Error! Sideset " << sideSetName << " has no sideNodeNumeration map.\n");
  std::map<GO, std::vector<int>> const& sideNodeNumerationMap = workset.disc->getSideNodeNumerationMap().at(sideSetName);

  // Establishing the kind of field layout
  std::vector<PHX::DataLayout::size_type> dims;
  field.dimensions(dims);
  std::string const& tag2 = dims.size() > 2 ? field.fieldTag().dataLayout().name(2) : "";
  ALBANY_PANIC(dims.size() > 2 && tag2 != "Node" && tag2 != "Dim" && tag2 != "VecDim", "Error! Invalid field layout in SaveSideSetStateField.\n");

  // Loop on the sides of this sideSet that are in this workset
  std::vector<Albany::SideStruct> const& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet) {
    // Get the data that corresponds to the side
    int const side_GID = it_side.side_GID;
    int const cell     = it_side.elem_LID;
    int const side     = it_side.side_local_id;

    // Not sure if this is even possible, but just for debug pourposes
    ALBANY_PANIC(
        elemGIDws3D[it_side.elem_GID].ws != workset.wsIndex,
        "Error! This workset has a side that belongs to an element not in the "
        "workset.\n");

    // We know the side ID, so we can fetch two things:
    //    1) the 2D-wsIndex where the 2D element lies
    //    2) the LID of the 2D element

    ALBANY_PANIC(
        ss_map.find(side_GID) == ss_map.end(),
        "Error! The sideId-to-sideSetElemId map does not store this side GID. "
        "Weird, should never happen.\n");

    int ss_cell_GID = ss_map.at(side_GID);
    int wsIndex2D   = elemGIDws2D[ss_cell_GID].ws;
    int ss_cell     = elemGIDws2D[ss_cell_GID].LID;

    // Then, after a safety check, we extract the StateArray of the desired
    // state in the right 2D-ws
    ALBANY_PANIC(esa[wsIndex2D].find(stateName) == esa[wsIndex2D].end(), "Error! Cannot locate " << stateName << " in PHAL_SaveSideSetStateField_Def.\n");
    Albany::MDArray state = esa[wsIndex2D].at(stateName);

    std::vector<int> const& nodeMap = sideNodeNumerationMap.at(side_GID);

    // Now we have the two arrays: 3D and 2D. We need to take the part we need
    // from the 3D and put it in the 2D one

    field.dimensions(dims);
    int size = dims.size();

    switch (size) {
      case 2:
        // side set cell scalar
        state(ss_cell) = field(cell, side);
        break;

      case 3:
        if (tag2 == "Node") {
          // side set node scalar
          for (int node = 0; node < dims[2]; ++node) {
            state(ss_cell, nodeMap[node]) = field(cell, side, node);
          }
        } else {
          // side set cell vector/gradient
          for (int idim = 0; idim < dims[2]; ++idim) {
            state(ss_cell, idim) = field(cell, side, idim);
          }
        }
        break;

      case 4:
        if (tag2 == "Node") {
          // side set node vector/gradient
          for (int node = 0; node < dims[2]; ++node) {
            for (int dim = 0; dim < dims[3]; ++dim) state(ss_cell, nodeMap[node], dim) = field(cell, side, node, dim);
          }
        } else {
          // side set cell tensor
          for (int idim = 0; idim < dims[2]; ++idim) {
            for (int jdim = 0; jdim < dims[3]; ++jdim) state(ss_cell, idim, jdim) = field(cell, side, idim, jdim);
          }
        }
        break;

      default: ALBANY_ABORT("Error! Unexpected array dimensions in SaveSideSetStateField: " << size << ".\n");
    }
  }
}

template <typename Traits>
void
SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::saveNodeState(typename Traits::EvalData workset)
{
  ALBANY_PANIC(workset.sideSets == Teuchos::null, "Error! The mesh does not store any side set.\n");

  if (workset.sideSets->find(sideSetName) == workset.sideSets->end()) return;  // Side set not present in this workset

  // Note: to save nodal fields, we need to open up the mesh, and work directly
  // on it.
  //       For this reason, we assume the mesh is stk. Moreover, since the cell
  //       buckets and node buckets do not coincide (elem-bucket 12 does not
  //       necessarily contain all nodes in node-bucket 12), we need to work
  //       with stk fields. To do this we must extract entities from the bulk
  //       data and use them to access the values of the stk field.

  ALBANY_PANIC(workset.disc == Teuchos::null, "Error! The workset must store a valid discretization pointer.\n");
  Teuchos::RCP<Albany::AbstractDiscretization> disc = workset.disc;

  Teuchos::RCP<Albany::LayeredMeshNumbering<LO>> layeredMeshNumbering = disc->getLayeredMeshNumbering();

  const Albany::AbstractDiscretization::SideSetDiscretizationsType& ssDiscs = disc->getSideSetDiscretizations();

  ALBANY_PANIC(ssDiscs.size() == 0, "Error! The discretization must store side set discretizations.\n");

  ALBANY_PANIC(ssDiscs.find(sideSetName) == ssDiscs.end(), "Error! No discretization found for side set " << sideSetName << ".\n");

  Teuchos::RCP<Albany::AbstractDiscretization> ss_disc = ssDiscs.at(sideSetName);

  std::map<std::string, std::map<GO, GO>> const& ss_maps = disc->getSideToSideSetCellMap();

  ALBANY_PANIC(
      ss_maps.find(sideSetName) == ss_maps.end(),
      "Error! Something is off: the mesh has side discretization but no "
      "sideId-to-sideSetElemId map.\n");

  // Get side_node->side_set_cell_node map from discretization
  ALBANY_PANIC(
      disc->getSideNodeNumerationMap().find(sideSetName) == disc->getSideNodeNumerationMap().end(),
      "Error! Sideset " << sideSetName << " has no sideNodeNumeration map.\n");

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> mesh = Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(ss_disc->getMeshStruct());
  ALBANY_PANIC(mesh == Teuchos::null, "Error! Save nodal states available only for stk meshes.\n");

  stk::mesh::MetaData& metaData = *mesh->metaData;
  stk::mesh::BulkData& bulkData = *mesh->bulkData;

  const auto& ElNodeID = disc->getWsElNodeID()[workset.wsIndex];

  using SFT = Albany::AbstractSTKFieldContainer::STKFieldType;
  SFT* stk_field = metaData.get_field<double> (stk::topology::NODE_RANK, stateName);
  ALBANY_PANIC(stk_field==nullptr, "Error! Field not found.\n");

  std::vector<PHX::DataLayout::size_type> dims;
  field.dimensions(dims);

  GO                nodeId3d;
  double*           values;
  stk::mesh::Entity e;

  // Notice: in the following, we retrieve the id of the stk node using the 3d
  // mesh, since it's easier.
  //         However, the node id in the 3d mesh is guaranteed to coincide with
  //         the node id in the 2d mesh for SideSetSTKMeshStruct. If the ss mesh
  //         type is not SideSetSTKMeshStruct, then the side mesh was not
  //         extracted from the 3d one; it's either the basal mesh of an
  //         extruded mesh, or the basal and volume meshes were loaded
  //         separately. Either way, we check if the mesh is layered (it will be
  //         for extruded meshes): if not, 2d and 3d ids will coincide; if yes,
  //         they will coincide if the ordering is LAYER, and won't coincide if
  //         the ordering is COLUMN. In the latter case, we need to use the
  //         LayeredMeshNumbering structure to determine the id of the 2d node
  //         given that of the 3d node.

  if (Teuchos::rcp_dynamic_cast<Albany::SideSetSTKMeshStruct>(mesh) != Teuchos::null || layeredMeshNumbering == Teuchos::null ||
      layeredMeshNumbering->ordering == Albany::LayeredMeshOrdering::LAYER) {
    // Either not a layered mesh or layered but not column-wise. Either way, the
    // GID of the side-set's nodes will coincide with the GID of the 3D mesh
    // nodes.

    // Loop on the sides of this sideSet that are in this workset
    std::vector<Albany::SideStruct> const& sideSet = workset.sideSets->at(sideSetName);
    for (auto const& it_side : sideSet) {
      // Get the data that corresponds to the side
      int const cell = it_side.elem_LID;
      int const side = it_side.side_local_id;

      // Notice: in the following, we retrieve the id of the stk node using the
      // 3d mesh.
      //         This is because the id of entities is the same (please don't
      //         change that) and it is easier to retrieve the id from the 3d
      //         discretization. Then, we use the id to extract the node from
      //         the 2d mesh.
      switch (dims.size()) {
        case 3:  // node_scalar
          for (int node = 0; node < dims[2]; ++node) {
            nodeId3d = ElNodeID[cell][sideNodes[side][node]];
            stk::mesh::EntityKey key(stk::topology::NODE_RANK, nodeId3d + 1);
            e         = bulkData.get_entity(key);
            values    = stk::mesh::field_data(*stk_field, e);
            values[0] = field(cell, side, node);
          }
          break;
        case 4:  // node_vector
          for (int node = 0; node < dims[2]; ++node) {
            nodeId3d = ElNodeID[cell][sideNodes[side][node]];
            e        = bulkData.get_entity(stk::topology::NODE_RANK, nodeId3d + 1);
            values   = stk::mesh::field_data(*stk_field, e);
            for (int i = 0; i < dims[3]; ++i) values[i] = field(cell, side, node, i);
          }
          break;
        default:  // error!
          ALBANY_ABORT(
              "Error! Unexpected field dimension (only node_scalar/node_vector "
              "for now).\n");
      }
    }
  } else {
    // It is a layered mesh, with column-wise ordering. This means that the GID
    // of the node in the 2D mesh will not coincide with the GID of the node in
    // the 3D mesh.

    LO nodeId2d, layer_id;

    // Loop on the sides of this sideSet that are in this workset
    std::vector<Albany::SideStruct> const& sideSet = workset.sideSets->at(sideSetName);
    for (auto const& it_side : sideSet) {
      // Get the data that corresponds to the side
      int const cell = it_side.elem_LID;
      int const side = it_side.side_local_id;

      switch (dims.size()) {
        case 3:  // node_scalar
          for (int node = 0; node < dims[2]; ++node) {
            nodeId3d = ElNodeID[cell][sideNodes[side][node]];
            layeredMeshNumbering->getIndices(nodeId3d, nodeId2d, layer_id);
            stk::mesh::EntityKey key(stk::topology::NODE_RANK, nodeId2d + 1);
            e         = bulkData.get_entity(key);
            values    = stk::mesh::field_data(*stk_field, e);
            values[0] = field(cell, side, node);
          }
          break;
        case 4:  // node_vector
          for (int node = 0; node < dims[2]; ++node) {
            nodeId3d = ElNodeID[cell][sideNodes[side][node]];
            layeredMeshNumbering->getIndices(nodeId3d, nodeId2d, layer_id);
            stk::mesh::EntityKey key(stk::topology::NODE_RANK, nodeId2d + 1);
            e      = bulkData.get_entity(key);
            values = stk::mesh::field_data(*stk_field, e);
            for (int i = 0; i < dims[3]; ++i) values[i] = field(cell, side, node, i);
          }
          break;
        default:  // error!
          ALBANY_ABORT(
              "Error! Unexpected field dimension (only node_scalar/node_vector "
              "for now).\n");
      }
    }
  }
}

}  // Namespace PHAL
