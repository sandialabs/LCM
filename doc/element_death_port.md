# Element death port: design and recon

## Why this document exists

The current `Albany::Application::applyDeathToActivePart` uses STK's
`stk::mesh::process_killed_elements` to walk the death boundary and
create/destroy face entities at the active/dead interface. Internally
that function calls
`BulkData::make_mesh_parallel_consistent_after_element_death`, which
fails when three or more MPI ranks simultaneously pass non-empty
`killed` lists whose deaths create or destroy faces at shared rank
boundaries: a stale `Entity` handle is left in a `{SHARES}` partition's
bucket and the next bucket sort trips
`Requirement(m_mesh.is_valid(curr_entity))` at `Partition.cpp:404`. Full
diagnosis: see `~/LCM/stk_findings_draft.txt`.

Sierra/SM's Adagio does not use `process_killed_elements`. Its
`Apst_ElemDeath::disconnectElements` in
`sierra/code/adagio/src/element_death/ElemDeath.C` (lines 1308-1613)
hand-rolls the algorithm using a clone-before-disconnect pattern that
avoids the STK harmonization path entirely. This document maps Adagio's
algorithm to a concrete LCM port plan: which STK primitives replace
which Fmwk calls, what we keep from the current LCM code, what we
discard, and where the design decisions are.

## Why Adagio's pattern works

The STK bug is triggered when STK's parallel-consistency code tries to
destroy or relocate a face entity that is still shared with a live cell
across a rank boundary. The cross-rank coordination corrupts the
`{SHARES}` bucket bookkeeping.

Adagio sidesteps the trigger by ensuring no shared face is ever
destroyed or modified while still referenced by a live cell on another
rank. For every dying cell:

1. The dying side's connection to a shared face is replaced by a new,
   locally-owned clone of that face.
2. The original face's relation to the dying cell is removed.
3. The original face is now exclusively on the live-cell side (one
   fewer back-reference) and is no longer shared from the dying side.
4. Only after global agreement do we destroy faces whose attached-cell
   count has reached zero, or unshared faces from the dying side.

All STK destructions therefore operate on entities that are local-only
at destruction time. The harmonization path is not exercised.

## Algorithm phases

Following Adagio (with the LCM-specific adaptations noted in
"Differences from Adagio" below):

### Phase 1 — Mark and synchronize entry

- Gather the local list of newly-killed cells (current
  `applyDeathToActivePart` logic, with the `deadCellsPart` dedup we
  just committed in `4da8e372b6`).
- Globally agree there is work to do
  (`stk::is_true_on_any_proc(comm, !killed.empty())`).
- Require the mesh be in `in_synchronized_state()` at entry.

### Phase 2 — Pre-modification parallel-assembled counts

Build two scratch fields, both `Real` valued, registered on all parts:

| Field name           | Rank | Purpose                                |
|----------------------|------|----------------------------------------|
| `nodeElemAttachCount`| NODE | Number of elements attached to a node  |
| `faceElemAttachCount`| FACE | Number of elements attached to a face  |

- Zero both fields.
- Loop over locally-owned elements; for each, increment
  `nodeElemAttachCount` for each connected node and
  `faceElemAttachCount` for each connected face.
- Parallel-assemble both fields so each rank sees globally-summed
  counts on shared nodes/faces.

STK primitive: **`stk::mesh::parallel_sum(bulkData, {field1, field2})`**.

### Phase 3 — Single modification block: clone-and-disconnect

Inside one `bulkData.modification_begin()` / `modification_end()` pair:

For each newly-killed cell `E`:

  - Determine which of its faces are shared (count > 1 from Phase 2).
    Unshared faces touch only `E` (or `E` plus other dying cells on the
    same rank) and need no cloning — they can be destroyed in Phase 5.
  - For each shared face `F` of `E`:
    1. Pre-allocate a new face entity `F'` (see "Global ID
       coordination" below).
    2. Connect `F'` to the same nodes as `F`, in the same order.
       (LCM's hex meshes use a single permutation; check whether
       Adagio's permutation handling at lines 1473-1488 is needed for
       our cases.)
    3. Add `F'` to the parts that `F` was in, minus
       `globally_shared_part`, plus `locally_owned_part`. Include the
       same side-set memberships, the `_created_during_death` part,
       and the `-erodible` side-sets per LCM's existing
       `side_parts`/`bc_mesh_parts` selection (lines 1270-1279 of
       `Albany_Application.cpp`).
    4. Copy all field data from `F` to `F'` (state by state).
    5. `bulkData.destroy_relation(E, F, side_ord)` — remove the
       relation from the dying cell to the old shared face.
    6. `bulkData.declare_relation(E, F', side_ord, perm)` — connect
       the dying cell to the new locally-owned clone.
    7. Increment `faceExposureCount` (a third scratch field, FACE-rank)
       on `F` to record that one back-reference has been removed.
  - For each unshared face `F` of `E`:
    1. Flag `F` for deletion via `faceDeleteFlag` (a fourth scratch
       field, FACE-rank).

  (No node cloning for the LCM port — see "Differences from Adagio".)

  - Move `E` into `deadCellsPart` (existing Step B1 logic, unchanged).

End the modification block.

### Phase 4 — Post-modification parallel-assembled exposure counts

Parallel-assemble `faceExposureCount` and `faceDeleteFlag` so that, for
shared faces touched on multiple ranks, every rank has the same
exposure-count and delete-flag values.

STK primitive: **`stk::mesh::parallel_sum(bulkData, {faceExposureCount,
faceDeleteFlag})`**.

### Phase 5 — Deletion pass

Inside a second `bulkData.modification_begin()` / `modification_end()`
pair:

- For each face `F`:
  - If `faceDeleteFlag[F] > 0.5` (unshared, was on dying cell): destroy
    `F`. (Locally-owned, no shared neighbors.)
  - Else if `faceExposureCount[F] >= 2` (shared and both sides have
    dropped their back-reference): destroy `F`. (No live cell
    references it on any rank.)
  - Else if `faceExposureCount[F] >= 1` and `F` is still attached to at
    least one live cell: leave `F` in the mesh, add it to the
    `exposedBoundaryPart` (the equivalent of Adagio's
    `exposed_boundary_part`; in LCM this is the `_created_during_death`
    part plus the `-erodible` side-sets, already in `side_parts`).

All these destructions are on faces that, by construction, no live cell
on any rank references at this point.

### Phase 6 — Post-conditions

- `bulkData.rebuild_face_adjacent_element_graph()` if cached.
- `stk_disc->rebuildWorksets()` (existing Step B3, unchanged).
- The dead cell is in `deadCellsPart`, its face relations are now to
  locally-owned faces, and STK's harmonization path was never invoked.

## Differences from Adagio

| Adagio behavior                              | LCM port behavior                | Reason                                                                                                                                                                       |
|----------------------------------------------|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Destroys the killed element                  | Keeps it (adds to deadCellsPart) | Existing LCM pattern; removing cells from `activePart` breaks `ACE_Bluff_Salinity` field-sizing. The dead cell stays computationally inert via the scatter-skip in J2Erosion.|
| Clones shared NODES                          | Skips node cloning               | We keep the dying element, so its node references stay valid. Shared nodes remain referenced by live cells too, so STK won't try to destroy them — no bug trigger.           |
| Clones the dying element                     | Doesn't clone                    | Same reason as above. Adagio clones because it deletes the original; we keep the original.                                                                                   |
| Uses Sierra Fmwk rosters and EntityDeletionOperationList for batched ops | Uses STK directly                | LCM doesn't link Fmwk. STK primitives suffice (`declare_element_side`, `destroy_entity`, etc.).                                                                              |
| Allocates new entity IDs via `commit_global_pending_create` | Allocates via `generate_new_entities` | STK's equivalent.                                                                                                                                                            |
| Uses `Fmwk::parallelAssemble` for field sum  | `stk::mesh::parallel_sum`        | STK's equivalent.                                                                                                                                                            |

## Fmwk-to-STK primitive map

| Adagio call                                                       | LCM/STK replacement                                                  |
|-------------------------------------------------------------------|----------------------------------------------------------------------|
| `Fmwk::parallelAssemble({field, ...})`                            | `stk::mesh::parallel_sum(bulkData, {field, ...})`                    |
| `fmwk_modification_begin(...)` / `fmwk_modification_end(...)`     | `bulkData.modification_begin()` / `bulkData.modification_end()`      |
| `fmwk_mesh.node_roster().create(id, NODE, parts, ...)`            | `bulkData.declare_node(generated_id, parts)`                         |
| `fmwk_mesh.face_roster().create(id, topo, parts, rels, ...)`      | `bulkData.declare_element_side(elem, side_ord, parts)` (preferred — STK assigns the right topology automatically) |
| `fmwk_mesh.elem_roster().create(...)`                             | Not needed in LCM port (we don't clone elements).                    |
| `fmwk_mesh.commit_global_pending_create(rank, true)`              | `bulkData.generate_new_entities(rank, count, new_entities)` for nodes; `declare_element_side` for faces (STK does the global ID work). |
| `Fmwk::EntityDeletionOperationList` + `execute_entity_deletion_operations` | Collect entities in a `vector<Entity>` and loop calling `bulkData.destroy_entity(e)`. |
| `Fmwk::PartChangeOperationList` + `execute_part_change_operations`| `bulkData.change_entity_parts(entity, add, remove)`.                 |
| `smtk::ParallelSum(local_int)`                                    | `stk::all_reduce_sum(bulkData.parallel(), &val, 1)`.                 |
| `Fmwk::FieldRef::state_new(field)`                                | LCM tends to use the field directly; for two-state fields, use `field.field_state(stk::mesh::StateNew)`. |
| `clone_shared_attribute_parts(entity, mesh, meta, parts)`         | Small helper: `parts = bulkData.bucket(entity).supersets()` filtered to keep only non-internal parts and the desired ownership/sharing flags. |
| `Fmwk::EntityHasPart(entity, part)`                               | `bulkData.bucket(entity).member(part)`.                              |
| `remove_relation(face, elem, ord, mesh)`                          | `bulkData.destroy_relation(elem, face, ord)`.                        |
| `meshtk::field_data<T>(field, entity)`                            | `stk::mesh::field_data(field, entity)`.                              |
| `sm::field_fill(value, field)`                                    | `stk::mesh::field_fill(value, field)`.                               |

## Open design decisions to resolve before coding

1. **`exposedBoundaryPart` naming.** Adagio has a dedicated
   `exposed_boundary_part` for marking newly-exposed surfaces. LCM
   currently paints exposed faces with `activePart`, `deadCellsPart`,
   and any `-erodible` side-sets (lines 1270-1279 of
   `Albany_Application.cpp`). The new code should preserve
   the LCM convention. Decide whether to:
   - Reuse `_created_during_death` part (created by STK in
     `ProcessKilledElements.cpp:236`) — requires us to also create
     the part.
   - Define our own equivalent and use it consistently.

2. **Face permutation handling.** Adagio's lines 1473-1488 track the
   permutation between an element and its face. STK's
   `declare_element_side` handles permutation internally for standard
   topologies, but the port's manual `declare_relation` calls
   need explicit permutation. Verify that
   `declare_element_side(elem, side_ord, parts)` gives the same
   result we'd want; if so, we can skip the manual permutation
   tracking entirely.

3. **Two-state scratch fields.** Adagio's fields are single-state
   per call. We can use single-state too; declare four scratch fields
   on `BulkData` startup (or lazy-register on first use):
   `nodeElemAttachCount`, `faceElemAttachCount`, `faceExposureCount`,
   `faceDeleteFlag`. Decide whether to register at mesh-creation
   time (per-app) or only when erosion is enabled.

4. **What to do with the dying cell's NEW (cloned) face references
   after the cell is in `deadCellsPart`.** Two options:
   - Keep the relations (dying cell connected to local clone faces).
     Computationally inert but tidy.
   - `destroy_relation` for all face relations of the dying cell.
     Faces lose their dying-cell back-reference; if no live cell
     references them either, they're destroyed in Phase 5.

   Option (b) is simpler — it makes Phase 5's "no remaining live
   references" check uniform across cloned and original faces.
   Recommend option (b).

5. **Orphan-node handling.** LCM's `fixOrphanNodesForElementDeath`
   currently regularizes zero-diagonal rows in the operator. With
   the new approach, nodes that lose all their live-cell
   references behave the same way (only dying cells reference them).
   The existing fix should still apply. No change needed.

6. **DBC propagation onto eroding surface.** Existing
   `computeNodeSets` clips `-erodible` node sets to the `-erodible`
   side-set's nodes. Hand-rolled clones get the same side-set
   memberships → existing clip logic should still work. Verify in
   the test.

## Sketch of LCM source structure

New file: `src/Albany_ElementDeath.{hpp,cpp}` (parallel to
`Albany_Application.cpp` where the entry point lives). Exports:

```cpp
namespace Albany {

// Top-level entry. Replaces the body of
// Application::applyDeathToActivePart from Step B1 onward (Steps B1-B3).
// Returns true if any cell was killed.
bool
applyElementDeath(
    stk::mesh::BulkData& bulkData,
    stk::mesh::Part& activePart,
    stk::mesh::Part& deadCellsPart,
    const stk::mesh::EntityVector& killed,
    const stk::mesh::PartVector& sideSetParts,        // -erodible side-sets
    const stk::mesh::PartVector& boundarySideSetParts);

// Internal helpers (also testable):
void countAttachedElements(
    const stk::mesh::BulkData&,
    const stk::mesh::Selector& live_elements,
    stk::mesh::Field<double>& nodeCounts,
    stk::mesh::Field<double>& faceCounts);

stk::mesh::Entity cloneFaceForDyingSide(
    stk::mesh::BulkData&,
    stk::mesh::Entity oldFace,
    stk::mesh::Entity dyingElement,
    unsigned faceOrd,
    const stk::mesh::PartVector& sideSetParts);

void deleteOrphanedAndExhaustedFaces(
    stk::mesh::BulkData&,
    stk::mesh::Field<double>& faceExposureCount,
    stk::mesh::Field<double>& faceDeleteFlag);

} // namespace Albany
```

`Albany_Application.cpp` change: replace the body of
`applyDeathToActivePart` from "Step B1" onward with a call to
`applyElementDeath(...)`. Preserve the prologue (gather
`killed`, dedup against `deadCellsPart`, the global early-out, the
`side_parts` and `bc_mesh_parts` construction). Drop the
`initialize_face_adjacent_element_graph` + `remoteActiveSelector`
machinery (we don't need it). Drop the
`stk::mesh::process_killed_elements` call.

`Albany_STKMeshStruct.{hpp,cpp}`: register the four scratch fields
when erosion is enabled. Suggested registration site: the existing
constructor path that sets up `activePart`/`deadCellsPart`.

## LOC estimate

Adagio's `disconnectElements` is ~300 LOC (lines 1308-1613) plus its
small helpers (`getPotentiallyKilledElements`,
`getNumberOfSharedNodesAndFaces`, `clone_shared_attribute_parts`,
`copy_all_variables`). Total in Adagio's element-death subsystem
relevant to this port: ~600 LOC.

LCM port, accounting for:

- Skipped node cloning and element cloning: ~120 LOC saved.
- Skipped Fmwk roster/deletion-list infrastructure: ~80 LOC saved.
- Added: scratch-field registration, STK-specific helpers: +60 LOC.
- Added: option-(b) post-clone relation cleanup: +30 LOC.

Realistic LCM port: **500-700 LOC of new code**, plus ~30 LOC of
deletions in `Albany_Application.cpp`. One new file pair, two
existing files touched (`Albany_Application.cpp`,
`Albany_STKMeshStruct.cpp`).

## Phased delivery plan

Three commits, each independently testable:

1. **Skeleton + single-rank parity.** Add the new file with the
   algorithm in place, wired behind an env flag
   (`ALBANY_NEW_ELEMENT_DEATH=1`). Leave
   `process_killed_elements` path as default. Verify single-rank
   parallel MiniErosion under the env flag matches serial.
2. **Multi-rank parity.** Iterate until 4-rank MiniErosion under the
   env flag matches serial bit-comparably (or to within an agreed
   tolerance). This is the actual STK-bug-avoidance proof.
3. **Cut over and retire.** Flip the default to the new path,
   remove the env gate, delete the `process_killed_elements` call,
   the `initialize_face_adjacent_element_graph` call, and the
   `populate_selected_value_for_remote_elements` machinery. Update
   `doc/element_death.md` to describe the new algorithm.

## What to read next (when starting the port)

- `~/sierra/code/adagio/src/element_death/ElemDeath.C:1308-1613` — the
  core algorithm.
- `~/sierra/code/adagio/src/element_death/ElemDeath.C` — the helper
  functions `getPotentiallyKilledElements`,
  `getNumberOfSharedNodesAndFaces`, `clone_shared_attribute_parts`,
  `remove_relation`.
- LCM's current `Application::applyDeathToActivePart` in
  `src/Albany_Application.cpp` (~line 1183 onward) — the code we are
  replacing.
- `doc/element_death.md` — the current algorithm description, to
  update at the end of Phase 3.
- `~/LCM/stk_findings_draft.txt` — full diagnosis of the bug we are
  routing around.
