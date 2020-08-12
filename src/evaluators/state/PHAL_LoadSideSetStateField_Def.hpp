// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_Macros.hpp"
#include "PHAL_LoadSideSetStateField.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_VerboseObject.hpp"

namespace PHAL {

template <typename EvalT, typename Traits, typename ScalarType>
LoadSideSetStateFieldBase<EvalT, Traits, ScalarType>::LoadSideSetStateFieldBase(Teuchos::ParameterList const& p)
{
  sideSetName = p.get<std::string>("Side Set Name");

  fieldName = p.get<std::string>("Field Name");
  stateName = p.get<std::string>("State Name");

  field = PHX::MDField<ScalarType>(fieldName, p.get<Teuchos::RCP<PHX::DataLayout>>("Field Layout"));

  this->addEvaluatedField(field);

  this->setName("Load Side Set Field " + fieldName + " from Side Set State " + stateName + PHX::print<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits, typename ScalarType>
void
LoadSideSetStateFieldBase<EvalT, Traits, ScalarType>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field, fm);

  d.fill_field_dependencies(this->dependentFields(), this->evaluatedFields());
}

template <typename EvalT, typename Traits, typename ScalarType>
void
LoadSideSetStateFieldBase<EvalT, Traits, ScalarType>::evaluateFields(typename Traits::EvalData workset)
{
  ALBANY_PANIC(workset.sideSets == Teuchos::null, "Error! The mesh does not store any side set.\n");

  if (workset.sideSets->find(sideSetName) == workset.sideSets->end()) return;  // Side set not present in this workset

  ALBANY_PANIC(workset.disc == Teuchos::null, "Error! The workset must store a valid discretization pointer.\n");

  const Albany::AbstractDiscretization::SideSetDiscretizationsType& ssDiscs = workset.disc->getSideSetDiscretizations();

  ALBANY_PANIC(ssDiscs.size() == 0, "Error! The discretization must store side set discretizations.\n");

  ALBANY_PANIC(
      ssDiscs.find(sideSetName) == ssDiscs.end(),
      "Error! No discretization found for side set " << sideSetName << ".\n");

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
  std::map<GO, std::vector<int>> const& sideNodeNumerationMap =
      workset.disc->getSideNodeNumerationMap().at(sideSetName);

  // Establishing the kind of field layout
  std::vector<PHX::DataLayout::size_type> dims;
  field.dimensions(dims);
  int                size = dims.size();
  std::string const& tag2 = size > 2 ? field.fieldTag().dataLayout().name(2) : "";
  ALBANY_PANIC(
      size > 2 && tag2 != "Node" && tag2 != "Dim" && tag2 != "VecDim",
      "Error! Invalid field layout in LoadSideSetStateField.\n");

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
    ALBANY_PANIC(
        esa[wsIndex2D].find(stateName) == esa[wsIndex2D].end(),
        "Error! Cannot locate " << stateName << " in PHAL_LoadSideSetStateField_Def.\n");
    Albany::MDArray state = esa[wsIndex2D].at(stateName);

    std::vector<int> const& nodeMap = sideNodeNumerationMap.at(side_GID);

    // Now we have the two arrays: 3D and 2D. We need to take the 2D one
    // and put it at the right place in the 3D one

    switch (size) {
      case 2:
        // side set cell scalar
        field(cell, side) = state(ss_cell);
        break;

      case 3:
        if (tag2 == "Node") {
          // side set node scalar
          for (int node = 0; node < dims[2]; ++node) {
            field(cell, side, node) = state(ss_cell, nodeMap[node]);
          }
        } else {
          // side set cell vector/gradient
          for (int idim = 0; idim < dims[2]; ++idim) {
            field(cell, side, idim) = state(ss_cell, idim);
          }
        }
        break;

      case 4:
        if (tag2 == "Node") {
          // side set node vector/gradient
          for (int node = 0; node < dims[2]; ++node) {
            for (int dim = 0; dim < dims[3]; ++dim) field(cell, side, node, dim) = state(ss_cell, nodeMap[node], dim);
          }
        } else {
          // side set cell tensor
          for (int idim = 0; idim < dims[2]; ++idim) {
            for (int jdim = 0; jdim < dims[3]; ++jdim) field(cell, side, idim, jdim) = state(ss_cell, idim, jdim);
          }
        }
        break;

      default: ALBANY_ABORT("Error! Unexpected array dimensions in LoadSideSetStateField: " << size << ".\n");
    }
  }
}

}  // Namespace PHAL
