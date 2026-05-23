# ACE Thermo-Mechanical Element Death

This document describes how the ACE sequential thermo-mechanical solver
decides that an element has "died" (failed and should no longer carry
load), how that decision propagates through assembly, and how the mesh and
boundary conditions follow the receding surface. It is written for
developers who want to understand or modify the algorithm.

It describes the **current** algorithm. It is not a design rationale or a
history of past approaches.

---

## 1. Overview

Element death in the ACE solver is **non-destructive in the sense that no
element is ever deleted**. A dead element is moved between STK *parts*; it
remains in the mesh database and in the Exodus output, but it stops
participating in the physics. There are two stages, on two different time
scales:

- **Within a step** — a cell that fails is *skipped* during residual and
  Jacobian assembly, so it contributes nothing to the global system. The
  cell is still a member of `activePart` and still in the worksets.
- **Between steps** — once per accepted step, the structural update runs
  (`Application::applyDeathToActivePart`, §7): the dead cells are moved out
  of `activePart` into `deadCellsPart`, STK creates the newly-exposed
  boundary faces, and the worksets are rebuilt so dead cells are excluded
  permanently.

So the mesh *topology* is preserved (no entity is destroyed), but the mesh
*is* modified between steps: cells change part membership and new face
entities appear on the eroding surface. Death therefore lags by one step —
a cell flagged during step N is skipped in assembly that same step and is
structurally removed by the step-N observer.

The death decision is made by the `J2Erosion` constitutive model, which
runs per integration point. The ACE coupling solver
(`LCM::ACEThermoMechanical`) reads the result at the start of each
mechanical step and passes it to the assembly evaluators.

Two ideas are kept deliberately separate:

- **Bookkeeping** — *how* each integration point failed (which of the five
  criteria tripped). Recorded for diagnostics and never discarded.
- **The death predicate** — *whether* the element is dead. This is a single
  rule: **an element dies when every one of its integration points has
  failed in at least one way.** It does not matter which criteria tripped
  or how many.

---

## 2. The five failure criteria

Each integration point (IP) is checked against up to five independent
failure criteria every time `J2Erosion` is evaluated. Each criterion is
controlled by a material parameter in the mechanics material YAML; setting
the parameter to `0.0` disables that criterion.

| Bit  | Mode          | Tripped when                                              | YAML parameter                     |
|------|---------------|-----------------------------------------------------------|------------------------------------|
| 0x01 | tension       | max principal Cauchy stress >= tensile strength           | `ACE Tensile Strength`             |
| 0x02 | strain        | deviatoric stretch norm >= strain limit                   | `ACE Strain Limit`                 |
| 0x04 | yield         | the J2 return map yielded this point                      | (always active; the J2 model)      |
| 0x08 | angle         | abs polar-rotation tilt angle >= critical angle           | `ACE Critical Angle`               |
| 0x10 | displacement  | displacement norm > maximum displacement                  | `ACE Maximum Displacement`         |

The exact tests are in `J2ErosionKernel::operator()` in
`src/LCM/models/J2Erosion_Def.hpp` (the `trip(...)` calls following the
stress update). `ACE Strain Limit` and `ACE Maximum Displacement` are
mandatory parameters — the model aborts at construction if they are
absent — so disable them explicitly with `0.0` rather than by omission.

Independently of failure, each criterion also writes a continuous
**indicator** field — the ratio of the actual quantity to its threshold
(`tensile_indicator`, `strain_indicator`, `angle_indicator`,
`displacement_indicator`, `yield_indicator`). Indicators are *always*
computed, even when erosion is disabled, so they are useful for "how close
to failure" visualization.

---

## 3. Data structures

Five objects carry failure/death information. Knowing what each one is and
who owns it is the key to modifying the algorithm.

### 3.1 `failure_modes` — the per-(cell, point) bitmask

- An Albany integration-point state variable (`dl->qp_scalar`), registered
  **with an old-state companion**. One scalar per integration point, used
  as a `uint8_t` bitmask: bits 0-4 are the five modes of §2.
- `failure_modes_old` is the bitmask converged at the **previous** step;
  `failure_modes` (the new value) is what the current fill writes back.
- **OR-accumulation**: each fill computes `failure_modes = failure_modes_old
  | (bits tripped this fill)`. A bit, once set, stays set for the rest of
  the run. A criterion that keeps tripping at the same point contributes
  exactly once.
- **STK-backed.** This is essential: `STKDiscretization::computeWorksetInfo`
  re-maps element states to their cells whenever the worksets are rebuilt
  (which erosion does constantly, §7). A bitmask held in an
  Application-side, workset-position-indexed cache would silently mis-map
  on a rebuild and corrupt the recorded death history.
- **This is the authoritative record for the death predicate.**
- Output to Exodus when the material YAML sets `Output Failure Modes: true`.

### 3.2 `failure_state` — the per-cell diagnostic encoding

- An Albany cell-scalar state variable (`dl->cell_scalar2`), one value per
  element.
- At the start of every `J2Erosion` evaluation it is recomputed from
  `failure_modes_old`: each set bit at each point of the cell adds a
  decimal magnitude — `1` (tension), `10` (strain), `100` (yield),
  `1000` (angle), `10000` (displacement) — and the current fill's new
  trips are then added.
- The result is a decimal-encoded histogram. Example: a value of `30201`
  decodes as 3 displacement trips, 0 angle, 2 yield, 0 strain, 1 tension,
  summed over all the points of that cell.
- **Purely diagnostic.** It does not determine death. It is what you
  inspect to see *how* a region is failing.
- Output to Exodus when the material YAML sets `Output Failure State: true`.

> **Limitation — elements with 10 or more integration points.**
> Each decimal "digit" of `failure_state` is really a column sum over all
> the integration points of the cell. The decode above is only valid while
> each per-mode count stays <= 9. If a cell has 10 or more integration
> points **and** 10 or more fail in the same mode, that column carries into
> the next mode's digit and the decode is silently corrupted. Standard
> 8-point hexes are safe; higher-order elements with >=10 quadrature points
> are not. This affects **only** the `failure_state` diagnostic — the death
> predicate reads the bitmask directly, so it is unaffected.

### 3.3 `cell_death` — the per-cell binary death flag

- An Albany cell-scalar state variable, one value per element: `1.0` if
  the element is dead, `0.0` if alive.
- Set by `J2Erosion` (see §4). `cell_death` is the **only** field that
  encodes the death predicate, and it is what the ACE solver reads.
- Output to Exodus when the material YAML sets `Output Cell Death: true`.

### 3.4 `death_status_vec` — the per-workset assembly-time death status

- Type: `std::vector<double>`, indexed by cell within a workset; `> 0`
  means dead.
- Owned by `Albany::Application` (`death_status_vecs_`, one per workset),
  attached to the workset in `loadWorksetBucketInfo`.
- Populated by the ACE solver at the **start of each mechanical step** from
  the current `cell_death` state (see §5).
- Read by the scatter evaluator during assembly (§6) and by
  `applyDeathToActivePart` between steps (§7). It can also be updated
  *mid-fill* by `J2Erosion` (see §4.3).

### 3.5 `activePart` and `deadCellsPart` — the STK parts

- `activePart` — elements that participate in assembly.
  `computeWorksetInfo` builds the worksets from `& activePart`, so dead
  cells are excluded from the worksets once they leave this part.
- `deadCellsPart` — elements that have been structurally killed; kept so
  visualization and BC queries can find them.
- Both are obtained from the STK mesh struct (`getActivePart()`,
  `getDeadCellsPart()`). A cell keeps its element-block membership when it
  dies — death only changes active/dead membership.

---

## 4. How `J2Erosion` marks failure and death

All of this happens in `J2ErosionKernel`, in `src/LCM/models/J2Erosion_Def.hpp`.

### 4.1 Seeding at the start of a fill (`init`)

Before the per-point kernel runs, `init()` iterates over every cell and,
from `failure_modes_old`:

- recomputes `failure_state(cell, 0)` (the decimal encoding of §3.2), and
- sets `cell_death(cell, 0) = 1.0` iff **every** point of that cell has a
  non-zero bitmask entry, else `0.0`.

This makes both per-cell fields consistent with the bitmask converged at
the previous step; `operator()` then adds the current fill's trips.

### 4.2 Tripping criteria (`operator()`, per point)

After the stress update, each of the five criteria is checked. The kernel
keeps a local `mask`, seeded from `failure_modes_old(cell, pt)`, and a
local lambda `trip(fired, bit, magnitude)`:

- if the criterion did not fire — return;
- if `disable_erosion_` is true — return (see §10);
- if the bit is already set in `mask` — return;
- otherwise set the bit in `mask` and add `magnitude` to the running
  `failed` accumulator (`failure_state`) for this cell.

After all five checks, the updated `mask` is written back to
`failure_modes(cell, pt)`. A dead cell (skipped early, §4.3 / §6) carries
its old bitmask forward unchanged so the state is still written every fill.

### 4.3 Same-fill death propagation (`operator()`, last point of a cell)

When the kernel processes the last integration point of a cell
(`pt == num_pts_ - 1`) and the workset carries a `death_status_vec`, it
checks `failure_modes` for every point of the cell. If every point now has
at least one bit set and the cell is not already marked dead, it sets
*both* `death_status_vec[cell] = 1.0` and `cell_death(cell, 0) = 1.0`
**in the same fill**.

This matters for nonlinear convergence: without it, Newton could push a
cell that just crossed the all-points-failed threshold further into a
nonphysical regime while the scatter evaluator — which has not run yet this
fill — keeps assembling its stress contributions. Updating
`death_status_vec` during the fill lets the scatter evaluator and the
orphan-node fix observe the new death within the same fill.

---

## 5. How the ACE solver seeds `death_status_vec` each step

`LCM::ACEThermoMechanical::ThermoMechanicalLoopDynamics`
(`src/LCM/solvers/ACE_ThermoMechanical.cpp`), at the start of each
mechanical step:

1. Reads the mechanical application's current `cell_death` element-state
   array. The states persist in the shared STK mesh between steps, so no
   snapshot save/restore is involved.
2. Builds `Application::death_status_vecs_[ws]` so that
   `death_status_vec[c] = cell_death(c)` for every cell in every workset.

`cell_death` is a cell scalar — there is no integration-point dimension to
index past. From this point on, every residual/Jacobian fill of that step
uses the set of dead cells as of the end of the previous converged step,
plus any cells that die during the fill itself (§4.3).

---

## 6. How dead cells are skipped in assembly

In `ScatterResidual` (`src/evaluators/scatter/PHAL_ScatterResidual_Def.hpp`):

- `evaluateFields` copies `workset.death_status_vec` into a device view
  `death_status_` and sets `have_death_status_` if any cell is dead.
- Each scatter kernel begins with:

  ```cpp
  if (this->have_death_status_ && this->death_status_(cell) > 0.0) return;
  ```

  A dead cell contributes nothing to the global residual (and, in the
  Jacobian/Tangent specializations, nothing to the global matrix).

A dead element is therefore "removed" from the physics, *within the step*,
purely by being skipped here — its stiffness and internal force never reach
the global system. The structural removal from the worksets happens
between steps (§7).

---

## 7. Between-step structural update (`applyDeathToActivePart`)

`Application::applyDeathToActivePart` (`src/Albany_Application.cpp`) is
called once per accepted step from the observer. It turns the *flags* set
during the step into a *structural* change of the mesh.

It scans `death_status_vecs_` for cells that are flagged dead but still
members of `activePart` and not yet in `deadCellsPart`, builds the
`side_parts` and `bc_mesh_parts` part vectors (§8), and then delegates
to `Albany::applyElementDeath` (`src/Albany_ElementDeath.cpp`), which
runs a clone-before-disconnect algorithm modeled on Sierra/SM's Adagio
(see `doc/element_death_port.md` and `~/LCM/stk_findings_draft.txt`
for the STK bug this routes around). The phases:

- **Pre-modification count.** Each rank counts elements attached to
  every face over its locally-owned live cells (writes to a
  side-rank scratch field `deathFaceElemAttachCount`), then
  `stk::mesh::parallel_sum` aggregates the counts so shared faces
  see the global total.
- **Clone-and-disconnect (single modification block).** For every
  face of every killed cell:
  - If the face is *shared* (count > 1): destroy the killed cell's
    relation to it, declare a new locally-owned face on the killed
    cell's side via `declare_element_side(elem, side_ord, sideSetParts)`,
    bump the original face's exposure marker, and set its boundary-face
    marker.
  - If the face is *unshared*: leave it attached. LCM keeps killed
    cells in the mesh (see below), so their exclusive boundary faces
    stay too.
  Then move every killed cell into `deadCellsPart`.
- **Post-modification parallel sum.** `stk::mesh::parallel_sum` of
  the exposure and boundary markers, so every rank sees the global
  totals on shared faces.
- **Part-membership update and deletion (second modification block).**
  Each rank, for its owned faces: faces whose boundary marker is set
  but whose exposure is not 2 are added to `sideSetParts` (STK then
  propagates the part change to the ghost copy on the other rank, so
  the `-erodible` node-set comes out the same on both sides — §8.3);
  faces whose exposure hit 2 have had both sides drop their back-
  reference and are destroyed (zero remaining live relations on any
  rank).
- **`rebuildWorksets()`.** The worksets are recomputed. Because
  `computeWorksetInfo` selects `& activePart`, the dead cells are
  excluded from every workset, and subsequent fills never include
  them.

No element entity is ever destroyed; the cells remain in
`deadCellsPart`. Faces created on the killed-cell side via
`declare_element_side` are kept; only shared faces whose live
neighbors are also gone are destroyed.

---

## 8. Boundary conditions on the eroding surface: the "-erodible" convention

As the surface erodes it recedes, so a boundary condition on that surface
must move with it. This is handled by a naming convention together with a
specific interaction between a side set and one or more node sets. Any
side set or node set whose name **contains the substring `erodible`**
participates; the code matches the substring, so the `<name>-erodible`
form is just a convention.

### 8.1 The two roles — a side set and the node sets

The eroding-surface boundary condition is built from two kinds of mesh
object that play different roles. Both must be defined.

- **An erodible side set — the surface.** It represents the exposed
  eroding surface itself. It must be defined as the *actual exposed
  surface at the start of the run* — for MiniErosion, the faces on the
  x+ bluff. It does **not** need to be defined for any later state: as
  cells die the solver extends it automatically (§8.2), so at every step
  it is exactly the current, receding surface.

- **Erodible node sets — the bands and the values.** Each carries the
  actual Dirichlet data. They are defined as full-depth slabs: each
  spans its band across the whole block — bluff, interior, and back
  face. This is a convenience, so the definition does not have to
  anticipate where the receding surface will be at any future step.

The Dirichlet BC is assigned (in the problem YAML) to the node sets. The
side set is what restricts that BC to the current surface.

### 8.2 How the side set tracks the receding surface

In the between-step structural update (§7), `applyDeathToActivePart`
passes every `-erodible` side set in `sideSetParts` to
`applyElementDeath`. When a killed cell's shared face is cloned, the
new face is declared into `sideSetParts` (so the clone lands directly
in the `-erodible` side set). The original face on the live-side
ghost is added to `sideSetParts` in the post-modification update so
the live rank's clipping sees it in `-erodible` too. An `-erodible`
side set therefore *grows*, one layer at a time, and at every step
it is exactly the set of currently exposed erodible faces — on both
sides of every rank boundary.

### 8.3 How the node sets are clipped to the side set

`STKDiscretization::computeNodeSets`, which runs on every
`rebuildWorksets()`, forms the union of all `-erodible` side sets and
intersects every `-erodible` node set with it:

    effective node set = defined slab  AND  nodes on the erodible side set

STK induces face -> node membership, so "nodes on the erodible side set"
is well defined: a node belongs to it exactly when it is a node of one of
the side set's faces. Because this intersection is recomputed on every
rebuild and the side set has grown (§8.2), the effective node set follows
the surface inward as it erodes.

### 8.4 The resulting effect

A Dirichlet BC assigned to an `-erodible` node set is applied only where
that node set's band overlaps the current eroding surface. It is applied
on the exposed surface within the band, follows it inward as the surface
recedes, and never reaches interior or back-face nodes — those never
become part of the erodible side set, so the intersection removes them,
from the first step onward. (A Neumann BC declared directly on an
`-erodible` side set likewise covers only currently exposed faces, since
the side set is exactly those faces.)

### 8.5 Definition requirement

The intersection in §8.3 restricts the BC only if the erodible side set
is defined as the *real* surface. If it is defined too broadly — for
example with a selection test that ends up matching every face of the
block — then every node counts as being "on the erodible surface", the
intersection removes nothing, and the BC spreads across the whole domain.
The erodible side set must therefore be exactly the actual initial
exposed surface, and nothing more. The node sets, by contrast, are meant
to be broad: the intersection is what restricts them.

### 8.6 MiniErosion example

`tests/LCM/ACE/MiniErosion/grid/cuboid_denudation.jou` defines:

- Side set `bluff_face-erodible` — the faces on the x+ bluff, i.e. the
  actual exposed surface at t = 0.
- Node sets `interval1-erodible` ... `interval4-erodible` — four z
  bands, each a full-depth slab, each carrying one temperature value of
  the thermal Dirichlet ramp.

The temperature for band k is applied on the portion of the bluff inside
band k and follows the bluff inward as it denudes; the back (x-) face
never joins `bluff_face-erodible`, so it stays at the initial condition.

---

## 9. Orphan-node fix

Skipping dead elements can leave **orphan nodes** — nodes connected only to
dead elements. They receive no stiffness, so their Jacobian rows would be
singular.

`Application::fixOrphanNodesForElementDeath`
(`src/Albany_Application.cpp`), called on the assembled Jacobian, is purely
operator-based — it does not consult `death_status_vec` or the mesh:

1. Reads the operator's diagonal (it arrives fill-complete) and computes a
   representative magnitude — the average of the non-zero diagonal entries.
2. Collects every row whose diagonal is exactly zero — those are the
   orphan DOFs — together with their column layout.
3. Re-opens the operator and, for each orphan row, sets the diagonal entry
   to the representative magnitude (off-diagonals already zero).

The result is a well-posed global system in which orphan DOFs simply hold
their value, with a diagonal scaled like the rest of the matrix so the
preconditioner is not disturbed.

---

## 10. `Disable Erosion`

Setting `Disable Erosion: true` in the mechanics material YAML disables
the element-death algorithm entirely for that material:

- `trip(...)` returns immediately, so **no bit is ever set, the `failed`
  accumulator never moves, and the bitmask stays empty.**
- Because the bitmask stays empty, the all-points-failed predicate can
  never be satisfied, so `cell_death` stays `0.0`, no cell is skipped in
  assembly, and `applyDeathToActivePart` finds nothing to kill.
- Failure **indicators** are still computed, so "proximity to failure"
  visualization still works with erosion disabled.

In other words, `Disable Erosion: true` gives you the full constitutive
response and all the diagnostic fields, with the death pathway disabled.

---

## 11. Two discretizations share one mesh

The ACE solver runs two `Albany::Application` subdomains — thermal and
mechanical — over a single shared STK `BulkData`. Each owns its own
`STKDiscretization`, and each caches raw pointers into STK field storage.

Erosion happens during the mechanical phase and reallocates that storage
(`applyElementDeath`'s two modification cycles). **Every**
discretization viewing the shared mesh must be re-synced with
`rebuildWorksets()` afterward, not only the eroding application's own — a
stale discretization would dereference dangling pointers. The ACE solver
coordinates this: `applyDeathToActivePart` rebuilds the mechanical
discretization, and `AdvanceThermalDynamics` rebuilds the thermal
discretization before the next thermal solve.

---

## 12. Output fields and ParaView visualization

Enable the per-cell fields in the mechanics material YAML:

```yaml
      Output Failure State: true   # decimal-encoded per-mode histogram
      Output Cell Death: true      # binary 0.0 / 1.0 death flag
      Output Failure Modes: true   # per-IP bitmask (optional, verbose)
```

To show **only alive cells** in ParaView:

1. Load the mechanical Exodus output.
2. Apply a `Threshold` filter.
3. Scalars = `cell_death`, range `-0.5 .. 0.5` (keep cells equal to 0).
4. Apply. Invert the range to `0.5 .. 1.5` to show only dead cells.

Threshold on `cell_death`, **not** on `failure_state`: `failure_state`
becomes non-zero as soon as a single integration point fails, so it would
hide cells that are still alive under the all-points-failed predicate.

`cell_death` and `failure_state` are monotone non-decreasing over the run —
an accurate per-cell history — because the bitmask they are derived from is
an STK-backed state that survives the workset rebuilds of §7.

The continuous indicator fields (`*_indicator`) can be enabled with their
own `Output * Indicator` flags for "how close to failure" colormaps.

---

## 13. Modification guide

Common modifications and where to make them:

- **Change a criterion's threshold** — material YAML parameter (§2). No
  code change.
- **Add a sixth failure mode** — add a bit (e.g. `0x20`) and a `trip(...)`
  call in `J2ErosionKernel::operator()`; pick a decimal magnitude
  (`100000`) and add it to the seeding loop in `init()`. The death
  predicate needs no change — it only asks whether each point's mask is
  non-zero.
- **Change the death predicate** — it is implemented in exactly two places,
  and both must remain consistent:
  - `J2ErosionKernel::init()` — the per-cell seeding of `cell_death`.
  - `J2ErosionKernel::operator()` — the same-fill last-point propagation.
  For example, to require a *fraction* of points to fail rather than all
  of them, change the "every point has a non-zero mask" test in both
  places.
- **Change what counts as dead at assembly time** — the scatter skip test
  in `PHAL_ScatterResidual_Def.hpp` (`death_status_(cell) > 0.0`).
- **Change how `death_status_vec` is seeded** — the loop in
  `ACEThermoMechanical::ThermoMechanicalLoopDynamics` that reads
  `cell_death` from the current element state (§5).
- **Change the between-step structural update** — `applyDeathToActivePart`
  in `Albany_Application.cpp` (§7).
- **Change BC propagation onto the eroding surface** — the `-erodible`
  side-set handling in `applyDeathToActivePart` and the node-set clip in
  `STKDiscretization::computeNodeSets` (§8).

### Invariants to preserve

- `failure_modes` is OR-accumulating and persistent — never clear a bit
  once set (except via the `disable_erosion_` short-circuit, which prevents
  bits from being set in the first place).
- Any per-cell or per-(cell, point) erosion state that must survive erosion
  must be an **STK-backed element state**, so `computeWorksetInfo` re-maps
  it to its cell across workset rebuilds. An Application-owned,
  workset-position-indexed cache silently mis-maps when the worksets
  repartition.
- `failure_state`, `cell_death`, and `failure_modes` must stay mutually
  consistent. `init()` rebuilds the two cell-scalar fields from the bitmask
  every fill, so the bitmask is authoritative.
- Death is monotonic: a cell that is dead stays dead. Nothing in the
  current algorithm resurrects a cell.
