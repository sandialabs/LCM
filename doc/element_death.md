# ACE Thermo-Mechanical Element Death

This document describes how the ACE sequential thermo-mechanical solver
decides that an element has "died" (failed and should no longer carry
load), and how that decision propagates through assembly. It is written
for power users who want to understand or modify the algorithm.

It describes the **current** algorithm. It is not a design rationale or a
history of past approaches.

---

## 1. Mental model

Element death in the ACE solver is **non-destructive**. The mesh is never
modified — no elements are deleted, no topology is rebuilt. Instead, a
dead element is simply *skipped* during residual and Jacobian assembly, so
it contributes nothing to the global system. The element still exists in
the mesh and in the Exodus output; it is just inert.

The death decision is made by the `J2Erosion` constitutive model, which
runs per integration point. The ACE coupling solver
(`LCM::ACEThermoMechanical`) reads the result at the start of each time
step and broadcasts it to the assembly evaluators.

Two ideas are kept deliberately separate:

- **Bookkeeping** — *how* each integration point failed (which of the five
  criteria tripped). This is recorded for diagnostics and never thrown
  away.
- **The death predicate** — *whether* the element is dead. This is a
  single rule: **an element dies when every one of its integration points
  has failed in at least one way.** It does not matter which criteria
  tripped or how many.

---

## 2. The five failure criteria

Each integration point (IP) is checked against up to five independent
failure criteria every time `J2Erosion` is evaluated. Each criterion is
controlled by a material parameter in the mechanics material YAML; setting
the parameter to `0.0` disables that criterion.

| Bit  | Mode          | Tripped when                                              | YAML parameter                     |
|------|---------------|-----------------------------------------------------------|------------------------------------|
| 0x01 | tension       | max principal Cauchy stress ≥ tensile strength            | `ACE Tensile Strength`             |
| 0x02 | strain        | deviatoric stretch norm ≥ strain limit                    | `ACE Strain Limit`                 |
| 0x04 | yield         | the J2 return map yielded this point                      | (always active; the J2 model)      |
| 0x08 | angle         | abs polar-rotation tilt angle ≥ critical angle            | `ACE Critical Angle`               |
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

Four objects carry failure/death information. Knowing what each one is and
who owns it is the key to modifying the algorithm.

### 3.1 `failure_mode_vec_` — the per-(cell, point) bitmask

- Type: `std::vector<std::vector<uint8_t>>`, indexed `[cell][pt]`.
- One `uint8_t` per integration point. Bits 0–4 are the five modes above.
- **OR-accumulation**: once a bit is set at a `(cell, pt)`, it stays set
  for the rest of the run. A criterion that keeps tripping at the same
  point therefore contributes exactly once.
- Owned by `Albany::Application` (`failure_mode_vecs_`, one per workset),
  attached to the workset in `loadWorksetBucketInfo`
  (`src/Albany_Application.hpp`). It persists across Phalanx fills and
  across time steps. The `[cell][pt]` grid is sized lazily by `J2Erosion`
  on first contact, because only the kernel knows `num_pts` for its model.
- **This is the source of truth for the death predicate.**

### 3.2 `failure_state` — the per-cell diagnostic encoding

- An Albany cell-scalar state variable (`dl->cell_scalar2`), one value per
  element.
- At the top of every `J2Erosion` evaluation it is recomputed from
  `failure_mode_vec_`: each set bit at each point of the cell adds a
  decimal magnitude — `1` (tension), `10` (strain), `100` (yield),
  `1000` (angle), `10000` (displacement).
- The result is a decimal-encoded histogram. Example: a value of `30201`
  decodes as 3 displacement trips, 0 angle, 2 yield, 0 strain, 1 tension,
  summed over all the points of that cell.
- **Purely diagnostic.** It does not gate death. It is what you inspect to
  see *how* a region is failing.
- Output to Exodus when the material YAML sets `Output Failure State: true`.

### 3.3 `cell_death` — the per-cell binary death flag

- An Albany cell-scalar state variable, one value per element: `1.0` if
  the element is dead, `0.0` if alive.
- Set by `J2Erosion` (see §4). `cell_death` is the **only** field that
  encodes the death predicate, and it is what the ACE solver reads.
- Output to Exodus when the material YAML sets `Output Cell Death: true`.

### 3.4 `death_status_vec` — the per-workset assembly-time death view

- Type: `std::vector<double>`, indexed by cell within a workset; `> 0`
  means dead.
- Owned by `Albany::Application` (`death_status_vecs_`, one per workset),
  attached to the workset in `loadWorksetBucketInfo`.
- Populated by the ACE solver at the **start of each step** from the
  previous step's converged `cell_death` (see §5).
- Read by the scatter evaluator and the orphan-node fix during assembly
  (see §6, §7). It can also be updated *mid-fill* by `J2Erosion` (see §4.3).

---

## 4. How `J2Erosion` marks failure and death

All of this happens in `J2ErosionKernel`, in `src/LCM/models/J2Erosion_Def.hpp`.

### 4.1 Seeding at the top of a fill (`init`)

Before the per-point kernel runs, `init()` walks every cell:

- recomputes `failure_state(cell, 0)` from the persistent bitmask
  (the decimal encoding of §3.2), and
- sets `cell_death(cell, 0) = 1.0` iff **every** point of that cell has a
  non-zero bitmask entry, else `0.0`.

This makes both per-cell fields consistent with the persistent bitmask at
the start of every evaluation.

### 4.2 Tripping criteria (`operator()`, per point)

After the stress update, each of the five criteria is checked. A local
lambda, `trip(fired, bit, magnitude)`, does the accumulation:

- if the criterion did not fire — return;
- if `disable_erosion_` is true — return (see §8);
- if the bit is already set in `failure_mode_vec_[cell][pt]` — return;
- otherwise set the bit and add `magnitude` to the running `failed`
  accumulator for this cell.

So a fresh trip flips one bit and bumps the diagnostic encoding; a repeat
trip of the same mode at the same point is a no-op.

### 4.3 Live death propagation (`operator()`, last point of a cell)

When the kernel processes the last integration point of a cell
(`pt == num_pts_ - 1`) **and** the workset carries a `death_status_vec`,
it scans the cell's full bitmask row. If every point has at least one bit
set and the cell is not already marked dead, it sets *both*
`death_status_vec[cell] = 1.0` and `cell_death(cell, 0) = 1.0`
**in the same fill**.

This matters for nonlinear convergence: without it, Newton could push a
cell that just crossed the all-points-failed threshold further into a
nonphysical regime while the scatter evaluator — which has not run yet
this fill — keeps assembling its stress contributions. Updating
`death_status_vec` live lets scatter and the orphan-node fix see the new
death within the same fill.

---

## 5. How the ACE solver seeds `death_status_vec` each step

`LCM::ACEThermoMechanical::ThermoMechanicalLoopDynamics`
(`src/LCM/solvers/ACE_ThermoMechanical.cpp`), once per mechanical step:

1. Restores the subdomain's saved internal state.
2. Reads the `cell_death` element state array from that saved state.
3. Builds `Application::death_status_vecs_[ws]` so that
   `death_status_vec[c] = cell_death(c)` for every cell in every workset.

`cell_death` is a cell scalar — there is no integration-point dimension to
index past. From this point on, every residual/Jacobian fill of that step
sees the dead set as it stood at the end of the previous converged step,
plus any cells that die *live* during the fill (§4.3).

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

A dead element is therefore "removed" from the physics purely by being
skipped here — its stiffness and internal force never reach the global
system.

---

## 7. Orphan-node fix

Skipping dead elements can leave **orphan nodes** — nodes connected only to
dead elements. They receive no stiffness, so their Jacobian rows would be
singular.

`Application::fixOrphanNodesForElementDeath`
(`src/Albany_Application.cpp`), called after global assembly:

1. Scans `death_status_vecs_`; if nothing is dead, returns immediately.
2. Marks every node touched by at least one alive element as "has alive".
3. Any node not so marked is an orphan.
4. For each orphan DOF: zeros the Jacobian row and sets the diagonal to
   the average alive-DOF diagonal magnitude (so the preconditioner sees
   uniform scaling), and zeros the residual entry.

The result is a well-posed global system in which orphan DOFs simply hold
their value.

---

## 8. `Disable Erosion`

Setting `Disable Erosion: true` in the mechanics material YAML makes the
element-death machinery completely inert for that material:

- `trip(...)` returns immediately, so **no bit is ever set, the `failed`
  accumulator never moves, and the bitmask stays empty.**
- Because the bitmask stays empty, the all-points-failed predicate can
  never be satisfied, so `cell_death` stays `0.0` and no cell is ever
  skipped.
- Failure **indicators** are still computed, so "proximity to failure"
  visualization still works with erosion disabled.

In other words, `Disable Erosion: true` gives you the full constitutive
response and all the diagnostic fields, with the death pathway switched
off.

---

## 9. Output fields and ParaView visualization

Enable the per-cell fields in the mechanics material YAML:

```yaml
      Output Failure State: true   # decimal-encoded per-mode histogram
      Output Cell Death: true      # binary 0.0 / 1.0 death flag
```

To show **only alive cells** in ParaView:

1. Load the mechanical Exodus output.
2. Apply a `Threshold` filter.
3. Scalars = `cell_death`, range `-0.5 .. 0.5` (keep cells equal to 0).
4. Apply. Invert the range to `0.5 .. 1.5` to show only dead cells.

Threshold on `cell_death`, **not** on `failure_state`: `failure_state`
becomes non-zero as soon as a single integration point fails, so it would
hide cells that are still alive under the all-points-failed predicate.

The continuous indicator fields (`*_indicator`) can be enabled with their
own `Output * Indicator` flags for "how close to failure" colormaps.

---

## 10. Tinkering guide

Common modifications and where to make them:

- **Change a criterion's threshold** — material YAML parameter (§2). No
  code change.
- **Add a sixth failure mode** — add a bit (e.g. `0x20`) and a `trip(...)`
  call in `J2ErosionKernel::operator()`; pick a decimal magnitude
  (`100000`) and add it to the seeding loop in `init()`. The death
  predicate needs no change — it only asks whether each point's mask is
  non-zero.
- **Change the death predicate** — it lives in exactly two places, and
  both must stay in sync:
  - `J2ErosionKernel::init()` — the per-cell seeding of `cell_death`.
  - `J2ErosionKernel::operator()` — the live last-point propagation.
  For example, to require a *fraction* of points to fail rather than all
  of them, change the "every point has a non-zero mask" test in both
  spots to "at least N points have a non-zero mask."
- **Change what counts as dead at assembly time** — the scatter skip test
  in `PHAL_ScatterResidual_Def.hpp` (`death_status_(cell) > 0.0`) and the
  orphan-node test in `Application::fixOrphanNodesForElementDeath`
  (`ds[cell] > 0.0`). These consume `death_status_vec`, which is `0.0` or
  `1.0` today.
- **Change how `death_status_vec` is seeded** — the loop in
  `ACEThermoMechanical::ThermoMechanicalLoopDynamics` that reads
  `cell_death` from the saved internal state.

### Invariants to preserve

- `failure_mode_vec_` is OR-accumulating and persistent — never clear a
  bit once set (except via the `disable_erosion_` short-circuit, which
  prevents bits from being set in the first place).
- `failure_state`, `cell_death`, and `failure_mode_vec_` must stay
  mutually consistent. `init()` rebuilds the two cell-scalar fields from
  the bitmask every fill, so the bitmask is authoritative.
- Death is monotonic: a cell that is dead stays dead. Nothing in the
  current algorithm resurrects a cell.
