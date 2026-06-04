# ACE Thermo-Mechanical Element Death

This document describes how the ACE sequential thermo-mechanical solver
decides that an element has "died" (failed and should no longer carry
load), how that decision propagates through assembly, and how the mesh
and boundary conditions follow the receding surface.

The first half is the user-facing description: the algorithm's flow, the
input parameters, the boundary-condition convention, and the output
fields. The second half (["Implementation
reference"](#implementation-reference)) is the developer-facing
description: data structures, code entry points, the detailed phases of
the structural update, the differences from the Sierra/SM Adagio
algorithm this code is modeled on, and the modification guide.

It describes the **current** algorithm. It is not a design rationale or
a history of past approaches.

---

## 1. Overview

Element death in the ACE solver is **non-destructive in the sense that
no element is ever deleted**. A dead element is moved between STK
*parts*; it remains in the mesh database and in the Exodus output, but
it stops participating in the physics. There are two stages, on two
different time scales:

- **Within a step** — a cell that fails is *skipped* during residual
  and Jacobian assembly, so it contributes nothing to the global
  system. The cell is still a member of the active part and still in
  the worksets.
- **Between steps** — once per accepted step, the structural update
  moves dead cells out of the active part into a dead-cells part, STK
  creates the newly-exposed boundary faces, and the worksets are
  rebuilt so dead cells are excluded permanently.

So the mesh *topology* is preserved (no entity is destroyed), but the
mesh *is* modified between steps: cells change part membership and new
face entities appear on the eroding surface. Death therefore lags by
one step — a cell flagged during step N is skipped in assembly that
same step and is structurally removed by the step-N observer.

The death decision is made by the `J2Erosion` constitutive model, which
runs per integration point. The ACE coupling solver
(`LCM::ACEThermoMechanical`) reads the result at the start of each
mechanical step and passes it to the assembly evaluators.

Two ideas are kept deliberately separate:

- **Bookkeeping** — *how* each integration point failed (which of the
  five criteria tripped). Recorded for diagnostics and never discarded.
- **The death predicate** — *whether* the element is dead. This is a
  single rule: **an element dies when every one of its integration
  points has failed in at least one way.** It does not matter which
  criteria tripped or how many.

---

## 2. The five failure criteria

Each integration point (IP) is checked against up to five independent
failure criteria every time `J2Erosion` is evaluated. Each criterion is
controlled by a material parameter in the mechanics material YAML;
setting the parameter to `0.0` disables that criterion.

| Bit  | Mode          | Tripped when                                              | YAML parameter                     |
|------|---------------|-----------------------------------------------------------|------------------------------------|
| 0x01 | tension       | max principal Cauchy stress >= tensile strength           | `ACE Tensile Strength`             |
| 0x02 | strain        | deviatoric stretch norm >= strain limit                   | `ACE Strain Limit`                 |
| 0x04 | yield         | the J2 return map yielded this point                      | (always active; the J2 model)      |
| 0x08 | angle         | abs polar-rotation tilt angle >= critical angle           | `ACE Critical Angle`               |
| 0x10 | displacement  | displacement norm > maximum displacement                  | `ACE Maximum Displacement`         |

`ACE Strain Limit` and `ACE Maximum Displacement` are mandatory
parameters — the model aborts at construction if they are absent — so
disable them explicitly with `0.0` rather than by omission.

Independently of failure, each criterion also writes a continuous
**indicator** field — the ratio of the actual quantity to its threshold
(`tensile_indicator`, `strain_indicator`, `angle_indicator`,
`displacement_indicator`, `yield_indicator`). Indicators are *always*
computed, even when erosion is disabled, so they are useful for
"how close to failure" visualization.

---

## 3. Within-step: dead cells are skipped in assembly

The ACE solver, at the start of each mechanical step, reads the current
per-cell death flag from the mesh state and builds a per-workset
device-friendly `death_status_vec` (one entry per cell, `> 0` means
dead). This is attached to every workset for the duration of the step.

The scatter evaluator begins each kernel with a single check:

```cpp
if (this->have_death_status_ && this->death_status_(cell) > 0.0) return;
```

A dead cell contributes nothing to the global residual (and, in the
Jacobian/Tangent specializations, nothing to the global matrix). Its
stiffness and internal force never reach the global system.

During the fill, if `J2Erosion` flags a *new* cell as dead at the last
integration point of that cell, it updates `death_status_vec` in place
so the scatter evaluator and the orphan-node fix observe the new death
within the same fill. This prevents Newton from pushing a just-failed
cell deeper into a nonphysical regime while the scatter continues to
assemble its stress contributions.

---

## 4. Between-step: clone-before-disconnect structural update

After each accepted step, the observer runs the structural update. It
turns the *flags* set during the step into a *structural* change of the
mesh: dead cells are moved out of the active part, the worksets are
rebuilt, and STK creates the new faces that have just been exposed at
the death boundary.

### 4.1 Why clone-before-disconnect

The naïve approach would be to ask STK to walk the death boundary and
fix up the active/dead interface directly
(`stk::mesh::process_killed_elements`). That code path fails on three
or more MPI ranks when deaths create or destroy faces at shared rank
boundaries: STK's cross-rank harmonization leaves a stale `Entity`
handle in a `{SHARES}` partition's bucket and the next bucket sort
aborts. Sierra/SM's Adagio solves the same problem by never letting
STK harmonize a shared face that a live cell still references on
another rank. We use the same pattern.

The idea: for every dying cell whose face is shared with a live cell on
another rank, *clone* the face into a new locally-owned face, attach
the dying cell to the clone, and drop the dying cell's relation to the
original. After this, the original face is on the live side only (with
one fewer back-reference), and the clone is on the dying side only.
Both ranks now agree on what's where. The only faces eventually
destroyed are ones with zero remaining live references, on any rank —
operations STK's serial code path handles cleanly.

### 4.2 The five phases

The structural update runs in five phases:

1. **Pre-count.** Each rank counts how many elements are attached to
   every face over its locally-owned live cells, then a parallel sum
   aggregates the counts so every rank sees the global total on shared
   faces. A face with global count > 1 is shared with a live cell on
   another rank.
2. **Clone-and-disconnect (one modification block).** For every face of
   every newly-killed cell:
    - If the face is *shared*: detach the killed cell from the face,
      declare a new locally-owned face on the killed cell's side
      (placed in the same side-set parts the original belongs to), and
      mark the original face as having one fewer back-reference.
    - If the face is *unshared*: leave it attached. LCM keeps killed
      cells in the mesh, so their exclusive boundary faces stay too.

    Then move every killed cell into the dead-cells part.
3. **Post-count.** A parallel sum of the exposure and boundary markers
   so every rank sees the global totals on shared faces.
4. **Part-membership update and deletion (second modification block).**
   Each rank, for its owned faces: faces newly exposed by death are
   added to the `-erodible` side sets (STK propagates the part change
   to the ghost copy on the other rank, so both ranks see the same
   `-erodible` membership). Faces whose live neighbors are also gone
   (both sides have dropped their back-reference) are destroyed.
5. **Workset rebuild.** Worksets are recomputed against the active
   part, so subsequent fills never include the dead cells.

No element entity is ever destroyed; killed cells remain in the
dead-cells part. New faces created on the killed-cell side stay; only
shared faces whose live neighbors are also gone are destroyed.

---

## 5. Boundary conditions on the eroding surface: the "-erodible" convention

As the surface erodes it recedes, so a boundary condition on that
surface must move with it. This is handled by a naming convention
together with a specific interaction between a side set and one or more
node sets. Any side set or node set whose name **contains the
substring `erodible`** participates; the code matches the substring, so
the `<name>-erodible` form is just a convention.

### 5.1 The two roles — a side set and the node sets

The eroding-surface boundary condition is built from two kinds of mesh
object that play different roles. Both must be defined.

- **An erodible side set — the surface.** It represents the exposed
  eroding surface itself. It must be defined as the *actual exposed
  surface at the start of the run* — for MiniErosion, the faces on the
  x+ bluff. It does **not** need to be defined for any later state: as
  cells die the solver extends it automatically (§5.2), so at every
  step it is exactly the current, receding surface.

- **Erodible node sets — the bands and the values.** Each carries the
  actual Dirichlet data. They are defined as full-depth slabs: each
  spans its band across the whole block — bluff, interior, and back
  face. This is a convenience, so the definition does not have to
  anticipate where the receding surface will be at any future step.

The Dirichlet BC is assigned (in the problem YAML) to the node sets.
The side set is what restricts that BC to the current surface.

### 5.2 How the side set tracks the receding surface

In the between-step structural update (§4), every `-erodible` side set
is passed to the clone-and-disconnect logic. When a killed cell's
shared face is cloned, the new face is declared into the same
`-erodible` side set the original belongs to (so the clone lands
directly in the side set). The original face on the live-side ghost is
added to the side set in the post-modification update so the live
rank's clipping sees it in `-erodible` too. An `-erodible` side set
therefore *grows*, one layer at a time, and at every step it is exactly
the set of currently exposed erodible faces — on both sides of every
rank boundary.

### 5.3 How the node sets are clipped to the side set

On every workset rebuild, the discretization forms the union of all
`-erodible` side sets and intersects every `-erodible` node set with
it:

    effective node set = defined slab  AND  nodes on the erodible side set

STK induces face -> node membership, so "nodes on the erodible side
set" is well defined: a node belongs to it exactly when it is a node of
one of the side set's faces. Because this intersection is recomputed on
every rebuild and the side set has grown (§5.2), the effective node set
follows the surface inward as it erodes.

### 5.4 The resulting effect

A Dirichlet BC assigned to an `-erodible` node set is applied only
where that node set's band overlaps the current eroding surface. It is
applied on the exposed surface within the band, follows it inward as
the surface recedes, and never reaches interior or back-face nodes —
those never become part of the erodible side set, so the intersection
removes them, from the first step onward. (A Neumann BC declared
directly on an `-erodible` side set likewise covers only currently
exposed faces, since the side set is exactly those faces.)

### 5.5 Definition requirement

The intersection in §5.3 restricts the BC only if the erodible side set
is defined as the *real* surface. If it is defined too broadly — for
example with a selection test that ends up matching every face of the
block — then every node counts as being "on the erodible surface", the
intersection removes nothing, and the BC spreads across the whole
domain. The erodible side set must therefore be exactly the actual
initial exposed surface, and nothing more. The node sets, by contrast,
are meant to be broad: the intersection is what restricts them.

### 5.6 MiniErosion example

`tests/LCM/ACE/MiniErosion/grid/cuboid_denudation.jou` defines:

- Side set `bluff_face-erodible` — the faces on the x+ bluff, i.e. the
  actual exposed surface at t = 0.
- Node sets `interval1-erodible` ... `interval4-erodible` — four z
  bands, each a full-depth slab, each carrying one temperature value of
  the thermal Dirichlet ramp.

The temperature for band k is applied on the portion of the bluff
inside band k and follows the bluff inward as it denudes; the back
(x-) face never joins `bluff_face-erodible`, so it stays at the initial
condition.

### 5.7 Cell-level `is_erodible` for material models

Two ACE material models change their constitutive behavior on cells
that sit on the eroding surface:

- `LCM::J2Erosion` (`models/J2Erosion_Def.hpp`) — when the cell is
  erodible AND the integration point is below sea level, the elastic
  modulus, yield strength, and strain limit are each divided by their
  respective "weakening factors" to soften the bluff face.
- `LCM::ACEThermalParameters` (`evaluators/ACEThermalParameters_Def.hpp`)
  — under the same condition, `bluff_salinity` is overridden with the
  ocean salinity, and `thermal_conductivity` / `heat_capacity` are scaled
  by a user-supplied `thermal_factor` (Jenn's sub-grid niche-formation
  model).

Both rely on a per-cell predicate that reuses the same `-erodible`
convention as §5. At workset build time `STKDiscretization::
computeWorksetInfoErodibleCells` constructs the union of all side-set
parts whose name contains `erodible`, walks each owned cell's face
entities, and flags the cell when any face is in that selector. The
result is exposed as `Workset::cell_is_erodible` (a per-cell
`std::uint8_t`) and the evaluators dereference it directly. Because the
predicate is rebuilt on every `rebuildWorksets()` call, newly exposed
cells become erodible automatically when the bluff recedes — the same
"side set tracks the receding surface" mechanism §5.2 already
described.

This is the third independent consumer of the `-erodible` substring
convention, alongside the death side-part propagation (§5.2) and the
node-set BC clipping (§5.3). One mesh tag, three behaviors.

---

## 6. Orphan-node fix

Skipping dead elements can leave **orphan nodes** — nodes connected
only to dead elements. They receive no stiffness, so their Jacobian
rows would be singular.

After assembly, the solver scans the assembled Jacobian for rows whose
diagonal is exactly zero — those are the orphan DOFs — and sets the
diagonal entry to a representative magnitude (the average of the
non-zero diagonal entries). Off-diagonals are already zero. The result
is a well-posed global system in which orphan DOFs simply hold their
value, with a diagonal scaled like the rest of the matrix so the
preconditioner is not disturbed.

This is purely operator-based — the fix consults the assembled
operator, not the death flags or the mesh.

---

## 7. Two discretizations share one mesh

The ACE solver runs two `Albany::Application` subdomains — thermal and
mechanical — over a single shared STK `BulkData`. Each owns its own
discretization, and each caches raw pointers into STK field storage.

Erosion happens during the mechanical phase and reallocates that
storage (the two modification cycles of §4.2). **Every** discretization
viewing the shared mesh must be re-synced afterward, not only the
eroding application's own — a stale discretization would dereference
dangling pointers. The ACE solver coordinates this: the mechanical
discretization is rebuilt by the between-step update, and the thermal
discretization is rebuilt before the next thermal solve.

---

## 8. `Disable Erosion`

Setting `Disable Erosion: true` in the mechanics material YAML
disables the element-death algorithm entirely for that material:

- No failure bit is ever set and no cell is ever marked dead.
- The all-points-failed predicate can never be satisfied, so no cell
  is skipped in assembly and the structural update finds nothing to
  kill.
- Failure **indicators** are still computed, so "proximity to
  failure" visualization still works with erosion disabled.

In other words, `Disable Erosion: true` gives you the full
constitutive response and all the diagnostic fields, with the death
pathway disabled.

---

## 9. Output fields and ParaView visualization

Enable the per-cell fields in the mechanics material YAML:

```yaml
      Output Failure State: true   # decimal-encoded per-mode histogram
      Output Cell Death: true      # binary 0.0 / 1.0 death flag
      Output Failure Modes: true   # per-IP bitmask (optional, verbose)
```

`failure_state` is a decimal-encoded histogram per cell: each set bit
at each point contributes a decimal magnitude — `1` (tension), `10`
(strain), `100` (yield), `1000` (angle), `10000` (displacement) —
summed over the integration points of the cell. A value of `30201`
decodes as 3 displacement trips, 0 angle, 2 yield, 0 strain, 1
tension. It is **purely diagnostic** — it does not determine death;
that is what `cell_death` is for.

To show **only alive cells** in ParaView:

1. Load the mechanical Exodus output.
2. Apply a `Threshold` filter.
3. Scalars = `cell_death`, range `-0.5 .. 0.5` (keep cells equal to
   0).
4. Apply. Invert the range to `0.5 .. 1.5` to show only dead cells.

Threshold on `cell_death`, **not** on `failure_state`:
`failure_state` becomes non-zero as soon as a single integration point
fails, so it would hide cells that are still alive under the
all-points-failed predicate.

`cell_death` and `failure_state` are monotone non-decreasing over the
run — an accurate per-cell history — because the underlying bitmask is
an STK-backed state that survives the workset rebuilds of §4.

The continuous indicator fields (`*_indicator`) can be enabled with
their own `Output * Indicator` flags for "how close to failure"
colormaps.

> **Limitation — elements with 10 or more integration points.**
> Each decimal "digit" of `failure_state` is really a column sum over
> all the integration points of the cell. The decode above is only
> valid while each per-mode count stays <= 9. Standard 8-point hexes
> are safe; higher-order elements with >=10 quadrature points are not.
> This affects **only** the `failure_state` diagnostic — the death
> predicate reads the bitmask directly, so it is unaffected.

---

# Implementation reference

The rest of this document is developer-facing: data structures, code
entry points, the detailed phases of the structural update with the
STK primitives used, the differences from Adagio that motivated the
LCM-specific choices, and the modification guide.

---

## A. Data structures

Five objects carry failure/death information.

### A.1 `failure_modes` — the per-(cell, point) bitmask

- An Albany integration-point state variable (`dl->qp_scalar`),
  registered **with an old-state companion**. One scalar per
  integration point, used as a `uint8_t` bitmask: bits 0-4 are the
  five modes of §2.
- `failure_modes_old` is the bitmask converged at the **previous**
  step; `failure_modes` (the new value) is what the current fill
  writes back.
- **OR-accumulation**: each fill computes
  `failure_modes = failure_modes_old | (bits tripped this fill)`. A
  bit, once set, stays set for the rest of the run. A criterion that
  keeps tripping at the same point contributes exactly once.
- **STK-backed.** This is essential: `STKDiscretization::
  computeWorksetInfo` re-maps element states to their cells whenever
  the worksets are rebuilt (which erosion does constantly, §4). A
  bitmask held in an Application-side, workset-position-indexed cache
  would silently mis-map on a rebuild and corrupt the recorded death
  history.
- **This is the authoritative record for the death predicate.**

### A.2 `failure_state` — the per-cell diagnostic encoding

- An Albany cell-scalar state variable (`dl->cell_scalar2`).
- At the start of every `J2Erosion` evaluation it is recomputed from
  `failure_modes_old` (§9), then the current fill's new trips are
  added on top.
- Purely diagnostic; does not determine death.

### A.3 `cell_death` — the per-cell binary death flag

- An Albany cell-scalar state variable: `1.0` if the element is dead,
  `0.0` if alive.
- Set by `J2Erosion`. `cell_death` is the **only** field that encodes
  the death predicate, and it is what the ACE solver reads.

### A.4 `death_status_vec` — the per-workset assembly-time status

- Type: `std::vector<double>`, indexed by cell within a workset; `> 0`
  means dead.
- Owned by `Albany::Application` (`death_status_vecs_`, one per
  workset), attached to the workset in `loadWorksetBucketInfo`.
- Populated by the ACE solver at the start of each mechanical step
  from the current `cell_death` state.
- Read by the scatter evaluator during assembly (§3) and by
  `applyDeathToActivePart` between steps (§4). Can also be updated
  *mid-fill* by `J2Erosion` (see [B.3](#b3-same-fill-death-propagation)).

### A.5 `activePart` and `deadCellsPart` — the STK parts

- `activePart` — elements that participate in assembly.
  `computeWorksetInfo` builds the worksets from `& activePart`, so
  dead cells are excluded from the worksets once they leave this
  part.
- `deadCellsPart` — elements that have been structurally killed;
  kept so visualization and BC queries can find them.
- Both are obtained from the STK mesh struct (`getActivePart()`,
  `getDeadCellsPart()`). A cell keeps its element-block membership
  when it dies — death only changes active/dead membership.

---

## B. `J2Erosion` — failure marking and death

All of this happens in `J2ErosionKernel`, in
`src/LCM/models/J2Erosion_Def.hpp`.

### B.1 Seeding at the start of a fill (`init`)

Before the per-point kernel runs, `init()` iterates over every cell
and, from `failure_modes_old`:

- recomputes `failure_state(cell, 0)` (the decimal encoding of §A.2),
  and
- sets `cell_death(cell, 0) = 1.0` iff **every** point of that cell has
  a non-zero bitmask entry, else `0.0`.

This makes both per-cell fields consistent with the bitmask converged
at the previous step; `operator()` then adds the current fill's trips.

### B.2 Tripping criteria (`operator()`, per point)

After the stress update, each of the five criteria is checked. The
kernel keeps a local `mask`, seeded from `failure_modes_old(cell, pt)`,
and a local lambda `trip(fired, bit, magnitude)`:

- if the criterion did not fire — return;
- if `disable_erosion_` is true — return (see §8);
- if the bit is already set in `mask` — return;
- otherwise set the bit in `mask` and add `magnitude` to the running
  `failed` accumulator (`failure_state`) for this cell.

After all five checks, the updated `mask` is written back to
`failure_modes(cell, pt)`. A dead cell (skipped early) carries its old
bitmask forward unchanged so the state is still written every fill.

The exact tests are the `trip(...)` calls following the stress update.

### B.3 Same-fill death propagation (`operator()`, last point of a cell)

When the kernel processes the last integration point of a cell
(`pt == num_pts_ - 1`) and the workset carries a `death_status_vec`,
it checks `failure_modes` for every point of the cell. If every point
now has at least one bit set and the cell is not already marked dead,
it sets *both* `death_status_vec[cell] = 1.0` and
`cell_death(cell, 0) = 1.0` **in the same fill**.

This is what lets the scatter evaluator and the orphan-node fix
observe the new death within the same fill (§3).

---

## C. ACE solver: seeding `death_status_vec` each step

`LCM::ACEThermoMechanical::ThermoMechanicalLoopDynamics`
(`src/LCM/solvers/ACE_ThermoMechanical.cpp`), at the start of each
mechanical step:

1. Reads the mechanical application's current `cell_death`
   element-state array. The states persist in the shared STK mesh
   between steps, so no snapshot save/restore is involved.
2. Builds `Application::death_status_vecs_[ws]` so that
   `death_status_vec[c] = cell_death(c)` for every cell in every
   workset.

`cell_death` is a cell scalar — there is no integration-point
dimension to index past. From this point on, every residual/Jacobian
fill of that step uses the set of dead cells as of the end of the
previous converged step, plus any cells that die during the fill
itself (§B.3).

---

## D. Structural update: `applyDeathToActivePart` and `applyElementDeath`

`Application::applyDeathToActivePart`
(`src/Albany_Application.cpp`) is called once per accepted step from
the observer. It scans `death_status_vecs_` for cells flagged dead
that still belong to `activePart` (deduplicating against
`deadCellsPart`), builds the side-set and boundary-side-set
`PartVector`s required by §5, then delegates to
`Albany::applyElementDeath` (`src/Albany_ElementDeath.cpp`).

`applyElementDeath` is the clone-before-disconnect implementation
described in §4.2. The five phases use these STK primitives:

| Phase | What | STK primitive |
|-------|------|---------------|
| 1. Pre-count | sum face attachments across ranks | `stk::mesh::parallel_sum(bulkData, {deathFaceElemAttachCount})` |
| 2. Clone & disconnect | declare new face on dying side, drop old relation | `declare_element_side(elem, side_ord, sideSetParts)`, `destroy_relation(elem, face, side_ord)` |
| 3. Post-count | sum exposure + boundary markers across ranks | `stk::mesh::parallel_sum(bulkData, {faceExposureCount, deathBoundaryFaceMarker})` |
| 4. Part update + delete | add new-boundary faces to `-erodible`, destroy unreferenced | `change_entity_parts(face, add)`, `destroy_entity(face)` |
| 5. Workset rebuild | refresh worksets from `& activePart` | `STKDiscretization::rebuildWorksets()` |

Both modification cycles are bracketed by
`bulkData.modification_begin()` / `modification_end()`. The four
scratch fields (`deathFaceElemAttachCount`, `faceExposureCount`,
`deathBoundaryFaceMarker`, and the optional unshared-face delete flag)
are registered on `BulkData` when erosion is enabled — see
`Albany_GenericSTKMeshStruct.cpp`.

The dying cell's new (cloned) face relations are kept after the cell
is moved to `deadCellsPart`; only shared faces with no live
back-references on any rank are destroyed in Phase 4.

---

## E. Differences from Sierra/SM Adagio

The algorithm is modeled on Adagio's `Apst_ElemDeath::disconnectElements`
(`sierra/code/adagio/src/element_death/ElemDeath.C:1308-1613`), with
these LCM-specific adaptations:

| Adagio behavior                                                          | LCM port behavior                | Reason |
|--------------------------------------------------------------------------|----------------------------------|--------|
| Destroys the killed element                                              | Keeps it (adds to `deadCellsPart`) | Existing LCM pattern; removing cells from `activePart` breaks `ACE_Bluff_Salinity` field-sizing. The dead cell stays computationally inert via the scatter-skip in `J2Erosion`. |
| Clones shared nodes                                                      | Skips node cloning               | We keep the dying element, so its node references stay valid. Shared nodes remain referenced by live cells too, so STK won't try to destroy them — no bug trigger. |
| Clones the dying element                                                 | Doesn't clone                    | Same reason as above. Adagio clones because it deletes the original; we keep the original. |
| Uses Sierra Fmwk rosters and `EntityDeletionOperationList` for batched ops | Uses STK directly              | LCM doesn't link Fmwk. STK primitives suffice (`declare_element_side`, `destroy_entity`, etc.). |
| Allocates new entity IDs via `commit_global_pending_create`              | STK assigns IDs through `declare_element_side` | STK's equivalent for face entities. |
| Uses `Fmwk::parallelAssemble` for field sum                              | `stk::mesh::parallel_sum`        | STK's equivalent. |

A more granular Fmwk-to-STK call map is preserved in the git history
(commit message of the original port doc); it's not reproduced here
because the LCM code only uses the STK side and the Fmwk references
date code that LCM never linked against.

---

## F. The STK bug this code routes around

Full diagnosis: `~/LCM/stk_findings_draft.txt`.

Short version. The naïve approach is
`stk::mesh::process_killed_elements`, which internally calls
`BulkData::make_mesh_parallel_consistent_after_element_death`. That
function fails when three or more MPI ranks simultaneously pass
non-empty `killed` lists whose deaths create or destroy faces at
shared rank boundaries: a stale `Entity` handle is left in a
`{SHARES}` partition's bucket and the next bucket sort trips
`Requirement(m_mesh.is_valid(curr_entity))` at `Partition.cpp:404`.

Adagio's clone-before-disconnect pattern sidesteps the trigger by
ensuring no shared face is ever destroyed or modified while still
referenced by a live cell on another rank. All STK destructions
operate on entities that are local-only at destruction time, so the
broken harmonization path is never exercised.

---

## G. Source layout

- `src/LCM/models/J2Erosion.hpp` / `J2Erosion_Def.hpp` — the failure
  criteria, the bitmask update, the cell-death predicate, the
  same-fill propagation.
- `src/LCM/solvers/ACE_ThermoMechanical.cpp` — the per-step seeding
  of `death_status_vec` from the mechanical mesh state.
- `src/evaluators/scatter/PHAL_ScatterResidual_Def.hpp` — the
  assembly-time skip of dead cells (`death_status_(cell) > 0.0`).
- `src/Albany_Application.cpp` — `applyDeathToActivePart`: the
  observer entry point, dedup against `deadCellsPart`, `side_parts`
  and `bc_mesh_parts` construction, the orphan-node fix.
- `src/Albany_ElementDeath.hpp` / `Albany_ElementDeath.cpp` — the
  clone-before-disconnect implementation (`applyElementDeath` and
  helpers).
- `src/disc/stk/Albany_GenericSTKMeshStruct.cpp` — registration of
  the four scratch fields when erosion is enabled, plus
  `activePart` / `deadCellsPart` setup.

---

## H. Modification guide

Common modifications and where to make them:

- **Change a criterion's threshold** — material YAML parameter (§2).
  No code change.
- **Add a sixth failure mode** — add a bit (e.g. `0x20`) and a
  `trip(...)` call in `J2ErosionKernel::operator()`; pick a decimal
  magnitude (`100000`) and add it to the seeding loop in `init()`.
  The death predicate needs no change — it only asks whether each
  point's mask is non-zero.
- **Change the death predicate** — implemented in exactly two
  places, and both must remain consistent:
  - `J2ErosionKernel::init()` — the per-cell seeding of `cell_death`.
  - `J2ErosionKernel::operator()` — the same-fill last-point
    propagation.

  For example, to require a *fraction* of points to fail rather than
  all of them, change the "every point has a non-zero mask" test in
  both places.
- **Change what counts as dead at assembly time** — the scatter skip
  test in `PHAL_ScatterResidual_Def.hpp`
  (`death_status_(cell) > 0.0`).
- **Change how `death_status_vec` is seeded** — the loop in
  `ACEThermoMechanical::ThermoMechanicalLoopDynamics` that reads
  `cell_death` from the current element state (§C).
- **Change the between-step structural update** — the phases in
  `Albany_ElementDeath.cpp` (§D).
- **Change BC propagation onto the eroding surface** — the
  `-erodible` side-set handling in `applyDeathToActivePart` /
  `applyElementDeath` and the node-set clip in
  `STKDiscretization::computeNodeSets` (§5).

### H.1 Invariants to preserve

- `failure_modes` is OR-accumulating and persistent — never clear a
  bit once set (except via the `disable_erosion_` short-circuit,
  which prevents bits from being set in the first place).
- Any per-cell or per-(cell, point) erosion state that must survive
  erosion must be an **STK-backed element state**, so
  `computeWorksetInfo` re-maps it to its cell across workset rebuilds.
  An Application-owned, workset-position-indexed cache silently
  mis-maps when the worksets repartition.
- `failure_state`, `cell_death`, and `failure_modes` must stay
  mutually consistent. `init()` rebuilds the two cell-scalar fields
  from the bitmask every fill, so the bitmask is authoritative.
- Death is monotonic: a cell that is dead stays dead. Nothing in the
  current algorithm resurrects a cell.
