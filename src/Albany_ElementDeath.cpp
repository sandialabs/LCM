// @HEADER
// *****************************************************************************
//                           Albany Package
//
// Copyright 2016 NTESS and the Albany contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Albany_ElementDeath.hpp"

#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/baseImpl/elementGraph/BulkDataIdMapper.hpp>
#include <stk_mesh/baseImpl/elementGraph/ElemElemGraph.hpp>
#include <stk_mesh/baseImpl/elementGraph/ParallelInfoForGraph.hpp>
#include <stk_mesh/baseImpl/elementGraph/ProcessKilledElements.hpp>
#include <stk_mesh/base/SkinBoundary.hpp>

namespace Albany {

bool
applyElementDeath(
    stk::mesh::BulkData&            bulkData,
    stk::mesh::Part&                activePart,
    stk::mesh::Part&                deadCellsPart,
    const stk::mesh::EntityVector&  killed,
    const stk::mesh::PartVector&    sideSetParts,
    const stk::mesh::PartVector&    boundarySideSetParts)
{
  // PHASE 1 SKELETON: delegate to stk::mesh::process_killed_elements so
  // the env-flag wiring can be validated end-to-end before the real
  // clone-before-disconnect algorithm lands. This makes the new code
  // path bit-comparable to the existing default path for single-rank
  // kills; subsequent phases will replace this body. See
  // doc/element_death_port.md for the planned algorithm.

  // Step B1: add killed cells to deadCellsPart. They remain in
  // activePart -- the existing LCM convention that keeps the
  // ACE_Bluff_Salinity field sizing intact.
  bulkData.modification_begin();
  for (stk::mesh::Entity cell : killed) {
    bulkData.change_entity_parts(
        cell,
        stk::mesh::PartVector{&deadCellsPart}, // add
        stk::mesh::PartVector{});              // remove nothing
  }
  bulkData.modification_end();

  // Step B2: let STK walk the active/dead interface and create the
  // newly-exposed face entities.
  bulkData.initialize_face_adjacent_element_graph();
  auto& elem_graph = bulkData.get_face_adjacent_element_graph();
  stk::mesh::impl::ParallelSelectedInfo remoteActiveSelector;
  stk::mesh::impl::populate_selected_value_for_remote_elements(
      bulkData, elem_graph,
      stk::mesh::Selector(activePart) & !stk::mesh::Selector(deadCellsPart),
      remoteActiveSelector);

  stk::mesh::process_killed_elements(
      bulkData, killed, activePart, remoteActiveSelector,
      sideSetParts,
      boundarySideSetParts.empty() ? nullptr : &boundarySideSetParts,
      stk::mesh::impl::MeshModification::modification_optimization::MOD_END_SORT);

  return true;
}

} // namespace Albany
