// @HEADER
// *****************************************************************************
//                           Albany Package
//
// Copyright 2016 NTESS and the Albany contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ALBANY_ELEMENTDEATH_HPP
#define ALBANY_ELEMENTDEATH_HPP

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Types.hpp>

namespace Albany {

/// LCM's element-death implementation.
///
/// Drives the active/dead-interface update for a set of newly-killed
/// elements using a clone-before-disconnect pattern (modeled on Adagio's
/// Apst_ElemDeath::disconnectElements). The pattern sidesteps an STK
/// bug in make_mesh_parallel_consistent_after_element_death that
/// corrupts {SHARES} buckets when 3+ ranks simultaneously kill cells
/// whose deaths cross shared face boundaries. See
/// doc/element-death.md (section "Implementation reference") for the
/// full algorithm and ~/LCM/stk_findings_draft.txt for the STK-bug
/// diagnosis.
///
/// Called only when the ALBANY_NEW_ELEMENT_DEATH env var is set;
/// otherwise Application::applyDeathToActivePart calls
/// stk::mesh::process_killed_elements directly.
///
/// Pre: bulkData is in synchronized state, killed has been deduplicated
/// against deadCellsPart, and all ranks have agreed (via
/// stk::is_true_on_any_proc) that there is work to do. The function is
/// collective: every rank must call it, even with an empty killed list.
///
/// Post: every cell in killed has been added to deadCellsPart, the
/// active/dead face interface has been updated, and bulkData is back
/// in synchronized state. activePart membership is preserved on killed
/// cells (the LCM convention for keeping field sizing intact).
///
/// Returns true if any cell was killed (matches the bool returned by
/// the surrounding Application::applyDeathToActivePart).
bool
applyElementDeath(
    stk::mesh::BulkData&            bulkData,
    stk::mesh::Part&                activePart,
    stk::mesh::Part&                deadCellsPart,
    const stk::mesh::EntityVector&  killed,
    const stk::mesh::PartVector&    sideSetParts,
    const stk::mesh::PartVector&    boundarySideSetParts);

} // namespace Albany

#endif // ALBANY_ELEMENTDEATH_HPP
