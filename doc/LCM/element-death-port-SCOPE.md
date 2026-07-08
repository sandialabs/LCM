# Scope: Port the reference implementation's element-death algorithm to LCM

Original scope note (the plan below has since been implemented).

## Why

Element death in LCM has been fixed in partial increments for weeks (erodible-BC
propagation, map rebuild for serial==parallel, dead-node Dirichlet, gradual
death in Permafrost, the stop-7139 wall). Each fix has converged on the same
place: **the reference implementation's proven element-death algorithm.** The GH #114
investigation made this explicit — the two live defects (mid-Newton death
"schism" between the live scatter-skip set and the frozen pin set; and a
self-certifying kill-all where a garbage Newton step annihilates the residual)
are exactly the failure modes the reference implementation's *between-solve death + reset-on-cutback*
are built to prevent. So: stop patching, port the algorithm.

Good news from the study: **LCM already owns the hard pieces.** The remaining
work is architectural refactoring, not new numerics.

---

## The target: the reference implementation's algorithm (condensed)

the reference implementation's implicit the outer death-control loop mode is a **coupled equilibrium + death
fixed-point loop** sitting *above* the Newton solver:

```
per load step:
  loadStepInit()                      # advance decay one step; commit finished decays
  loop (death control iteration):
    solve equilibrium to convergence  # FROZEN active set — Newton never sees a set change
    evaluate death on the CONVERGED state
    if new deaths:
      remove elements between solves  # STK part change + create exposed faces
      re-solve equilibrium
    else:
      break                           # converged: a full solve produced ZERO new marks
    if iters > limit or solve failed:
      reset tentative deaths; restore state; cut dt   # reset-on-cutback
```

Seven invariants to replicate (each verified in the reference source):

1. **Active set is constant within a Newton solve** — death only writes status
   fields; the mesh part-change happens strictly between solves.
2. **Death decided only on converged data** — never mid-iteration.
3. **Coupled problem is a fixed-point loop** — converged only when a full
   equilibrium solve yields zero new death marks.
4. **One (softening/material) element dies per outer iteration, globally**
   (rank-reduced single-winner) — the key to Newton robustness for softening
   materials.
5. **Stiffness stays well-conditioned** — dying elements *fade* over
   `Num Death Steps` (a decay scale multiplies stiffness); an element is only
   removed when the scale hits ~1e-6; fully-dead DOFs are zeroed and orphan
   nodes/faces deactivated so no rigid-body null space enters the tangent.
6. **Reset-on-cutback is total and reversible** — tentative marks and the
   solution are rolled back (`state_new <- state_old`); committed prior deaths
   are preserved.
7. **Newly-exposed surface inherits BCs/contact.**

The reference algorithm also has a simpler an in-solve death mode mode (no outer loop; death
evaluated once at step start + once at converged) — a useful fallback for the
first landing.

---

## LCM today: KEEP / REPLACE / ADD

**KEEP — LCM already has these, and several are already reference-modeled:**

| Piece | Where |
|---|---|
| Clone-before-disconnect surgery (== the reference clone-before-disconnect surgery) | `Albany_ElementDeath.cpp`, `Application::applyDeathToActivePart` (Application.cpp ~1837) |
| `-erodible` side/node-set tracking + calving reachability | `Application.cpp` ~1949; `STKDiscretization::computeNodeSets`, `findDetachedCells` ~2911 |
| SPD hold-in-place pin + connectivity-dead query | `fixOrphanNodesForElementDeath`/`zeroResidualAtDeadNodes` (Application.cpp ~1738/1814); `getDeadNodeDOFGids` (STKDiscretization ~3061) |
| Post-convergence state pass (natural "evaluate death at converged" hook) | `ObserverImpl::observeSolution` (ObserverImpl.cpp ~74) — used by ACE, Piro, Schwarz |
| State snapshot/restore | `snapshot/restoreSharedMeshStates` (ACE_ThermoMechanical.cpp ~1088/1113) |
| dt cutback | ACE_ThermoMechanical.cpp ~873 |
| Map/graph rebuild + solution migration after death | ACE_ThermoMechanical.cpp ~931; `rebuildAfterTopologyChange` (STKDiscretization ~2876) |
| **Gradual-decay prototype** (closest thing to the reference implementation's stiffness ramp) | Permafrost `death_steps_`/`death_decay_` (Permafrost_Def.hpp ~895-924) |

**REPLACE:**

| Piece | Problem | Target |
|---|---|---|
| Death DECISION in the Phalanx fill (last-IP, live) | J2Erosion_Def.hpp ~768-782, Permafrost_Def.hpp ~883-933 — flips `cell_death` mid-assembly (the #114 schism) | Models emit per-IP `failure_modes` only; the manager decides death on the converged state |
| Per-model duplicated death logic | J2Erosion (instant) vs Permafrost (instant+decay) each hand-roll the predicate + dead-cell short-circuit | One generic failure->death evaluator |
| Instant removal via scatter-skip | PHAL_ScatterResidual_Def.hpp | Gradual decay; scatter-skip becomes the decay->0 terminal case |
| Driver-specific death wiring | ACE-only (ACE_ThermoMechanical.cpp ~827-849, ~931-997); no-op on the Piro path | Shared, driver-agnostic manager on `Application` |

**ADD (genuinely new):**

- Outer death-iteration loop with a **death-fraction / iteration limit** (none today).
- **One-death-per-iteration** global single-winner throttle for softening materials (reference invariant #4).
- Generic, model-agnostic **gradual stiffness decay** (promote Permafrost's).
- **Piro/Tempus step-rejection death hook** so standard (non-ACE) problems get death.
- **Quasistatic ACE death** — `ThermoMechanicalLoopQuasistatics` is an empty stub.
- **Reset-of-tentative-deaths** on cutback (undo marks made this step).

---

## Target LCM architecture

A new `Albany::ElementDeathManager`, owned by `Albany::Application` (absorbing
today's `death_status_vecs_`, `frozen_dead_dof_gids_`, `topology_changed_`, and
the ACE-inlined rebuild sequence). It exposes a small driver-agnostic API:

- `beginDeathStep()` — snapshot state; freeze the active set for the coming solve(s).
- `evaluateDeathAtConverged() -> DeathStats{numMarked, numDecayed}` — run the
  criteria on the converged solution (in the SFM pass that already runs), apply
  one decay step, select at most one softening element (throttle), mark.
- `commitDeathAndRebuild()` — clone-death surgery + create exposed faces +
  rebuild disc/maps/DBC/solver + migrate solution vectors.
- `resetTentativeDeaths()` — undo this step's marks/decay for cutback.

Drivers use it three ways:
- **ACE Sequential** (has a hand-rolled outer loop): calls the API explicitly
  around its mechanical block. Mechanical.
- **Standard Piro (NOX/Tempus)**: a Tempus step-rejection observer that rejects
  a step when `numMarked>0`, forcing Tempus's own cutback + re-solve. New infra.
- **Schwarz**: inherits via its observer, same as today.

Constitutive models (J2Erosion, Permafrost) change from *deciding* death to
*reporting* per-IP failure + honoring a decay scale supplied by the manager.

---

## Phased plan

Sequenced so **#114 is fixed early** (Phases 0-1) and the full generalization
follows. Effort sizes are rough (S ≈ days, M ≈ 1-2 wk, L ≈ 3+ wk) for one
developer.

**Phase 0 — Lift the death decision out of the fill. [M] (fixes #114 core)**
- Add `workset.allow_death_propagation`, true only in `evaluateStateFieldManager`.
- Gate the live-death blocks in J2Erosion/Permafrost on it, so during a Newton
  solve the death set is frozen; death is decided only in the post-convergence
  SFM pass.
- This alone closes the mid-Newton schism *and* the self-certifying kill-all
  (Fable's #114 "Option B" gating). Highest value per unit effort.

**Phase 1 — Outer death iteration in the ACE driver. [M]**
- Wrap the mechanical block in a ControlFailure-style loop: solve -> evaluate
  death at converged -> if new deaths, `commitDeathAndRebuild` + re-solve; stop
  when a solve yields zero new marks. Reuse the existing snapshot/restore +
  cutback + rebuild.
- **Conservative first landing:** handle in-place deaths inline; route
  clone-deaths (mid-step topology change) to the existing dt-cutback path. This
  de-risks Phase 3 while delivering a correct, non-degenerate #114.
- Add the death-fraction/iteration limit.

**Phase 2 — Generic manager + gradual decay + throttle. [L]**
- Extract `ElementDeathManager`; move `death_status_vecs_`/`frozen_dead_dof_gids_`/
  rebuild into it; make it driver-agnostic.
- Promote Permafrost's decay to a model-agnostic decay scale (the reference element-death algorithm
  `HOURGLASS_DECAY`) applied at assembly; `Num Death Steps` knob.
- Add the one-softening-element-per-iteration global single-winner selection.
- Models reduced to emitting `failure_modes` + honoring the decay scale.

**Phase 3 — Safe within-load-step topology rebuild. [L, highest risk]**
- Make `commitDeathAndRebuild` safe *inside* a load step (per death iteration),
  not just between steps: rebuild maps/graph, re-eliminate DBCs, migrate
  `x/xdot/xdotdot`, and refresh the model evaluator's `x_space` mid-`evalModel`.
  Today this is deliberately deferred to a between-step point precisely because
  a mid-`evalModel` rebuild invalidates `x_space`.
- This removes the Phase-1 conservative fallback (clone-deaths no longer need a
  cutback).

**Phase 4 — Driver-agnostic Piro/Tempus death hook. [M]**
- Tempus step-rejection observer that rejects on `numMarked>0` -> Tempus cutback
  + re-solve, so standard problems get reference element death without a hand-rolled loop.

**Phase 5 — Quasistatic ACE death + parity. [M]**
- Implement `ThermoMechanicalLoopQuasistatics` death (currently a stub); ensure
  Schwarz parity.

**Phase 6 — Recalibration, golds, serial/parallel. [M, ongoing]**
- Rebaseline the ACE erosion golds (death timing moves deliberately).
- First-cut recalibration of the thermal-thaw chain (the separate #114
  calibration issue: soil-yield floor, near-step ice(T) curve smoothing,
  rate-limited death) now that gradual decay exists.
- Re-verify serial==parallel (with the direct-solver diagnostic and MueLu test
  from the #114 solver question); confirm GH #113 restart bit-identity still
  holds after the death-timing change.

Incremental value: **Phases 0-1 subsume the #114 point fix and can land first.**
Phases 2-5 are the true "clone the reference algorithm" generalization. Phase 6 is continuous.

---

## Risk register

1. **Mid-load-step topology rebuild (Phase 3)** — the single biggest engineering
   item; mitigated by the Phase-1 conservative fallback (defer clone-deaths to
   cutback) so it isn't on the critical path for fixing #114.
2. **Removing mid-fill death regresses cascade convergence** — the live writes
   exist to stop Newton assembling nonphysical stress through a cascade; they
   must be removed *together with* the outer re-solve loop (Phases 0+1 as a
   unit), not independently.
3. **Gold rebaselines across the ACE erosion suite** — death timing changes; a
   deliberate, reviewed rebaseline (list in the PR). Fold into #114 recalibration.
4. **Parallel reproducibility** — the manager must re-snapshot the pin set and
   scatter set *together* each death iteration and keep the `MPI_Allgatherv`
   lockstep; any drift reintroduces the free-fall/stall modes. (Note: the
   remaining ILUT serial/parallel divergence from #114 is a *separate* solver
   issue, not fixed by this port.)
5. **Two divergent model death paths** — a generic manager must subsume both
   J2Erosion (instant) and Permafrost (decay) semantics without changing their
   calibrated failure behavior.
6. **Observer firing on failed solves** — guard death evaluation on a
   solve-succeeded flag so a garbage state can't trip irreversible deaths.

---

## Recommendation

Land **Phase 0 + Phase 1 (with the conservative topology fallback)** first as
the element-death v1 — it fixes #114's degenerate solve and
whole-mesh-death artifact using the between-solve/reset-on-cutback core, with
minimal blast radius. Then generalize into the shared manager, gradual decay,
throttle, driver-agnostic hook, and quasistatic path (Phases 2-5), rebaselining
and recalibrating as we go (Phase 6). This converts weeks of point fixes into a
single principled algorithm that all drivers share.
