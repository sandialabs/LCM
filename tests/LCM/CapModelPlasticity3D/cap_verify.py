#!/usr/bin/env python3
"""Verification harness: compare the LCM CapModel/Permafrost response
(single-element Albany run, homogeneous strain) against the independent
reference implementation in cap_reference.py, plus the analytic
crush-curve identity on the hydrostat and the per-mode failure/death
bookkeeping.

Usage: cap_verify.py <exo-file> <path>
  path in {hydrostatic, confined, triaxial, confined_fd, triaxial_fd,
           permafrost_thaw, permafrost_f05, death_tension,
           death_backstress, death_crush, death_eqps, death_displacement}
Exit code 0 on PASS.
"""

import sys
import numpy as np
import netCDF4

import cap_reference as ref


def exo_open(fname):
    ds = netCDF4.Dataset(fname)
    names = [''.join(c.decode() for c in row if c != b'')
             for row in ds.variables['name_elem_var'][:]]
    idx = {n: i for i, n in enumerate(names)}

    def var(name):
        # vals_elem_var<j>eb1, j is 1-based variable index; second array
        # index is the element (single-element meshes here). Cell-level
        # states may or may not carry an _1 suffix depending on layout.
        if name not in idx and name + '_1' in idx:
            name = name + '_1'
        return np.array(ds.variables[f'vals_elem_var{idx[name]+1}eb1'][:, 0])

    t = np.array(ds.variables['time_whole'][:])
    return t, var


def exo_series(fname):
    t, var = exo_open(fname)
    # First index = tensor component (row-major 1..9), second = int. point.
    sxx = var('Cauchy_Stress_1_1')
    syy = var('Cauchy_Stress_5_1')
    szz = var('Cauchy_Stress_9_1')
    sxy = var('Cauchy_Stress_2_1')
    kappa = var('Cap_Parameter_1')
    evp = var('volPlastic_Strain_1')
    return t, sxx, syy, szz, sxy, kappa, evp


# Salem-limestone sediment-skeleton (shared) parameters, matching the
# materials_permafrost_death_*.yaml decks (both end members = Salem,
# constant ice saturation 1.0: the trajectories are the already-verified
# CapModel paths, so the death tests isolate the failure bookkeeping).
SALEM_SHARED = dict(D=3.94e-4, theta=0.0, L=3.94e-4, phi=0.0,
                    R=28.0, Q=28.0, psi=1.0, D2=0.0)

# Per-mode death tests: load path, indicator field, deck threshold, and
# the failure_modes bit / failure_state decimal magnitude. The thresholds
# here must match the materials decks.
DEATH_MODES = {
    'death_tension':    dict(eps=lambda t: np.diag([0.0015 * t, 0.0, 0.0]),
                             key='tension', ind='Tension_Indicator',
                             threshold=0.9, bit=1, mag=1.0),
    'death_backstress': dict(eps=lambda t: np.diag([-0.04 * t, 0.0, 0.0]),
                             key='backstress', ind='Backstress_Indicator',
                             threshold=0.95, bit=2, mag=10.0),
    'death_crush':      dict(eps=lambda t: -0.02 * t * ref.I3,
                             key='crush', ind='Crush_Indicator',
                             threshold=0.25, bit=4, mag=100.0),
    'death_eqps':       dict(eps=lambda t: np.diag([-0.04 * t, 0.0, 0.0]),
                             key='eqps', ind='Eqps_Indicator',
                             threshold=1.0, bit=32, mag=100000.0,
                             max_eqps=0.01),
    'death_displacement': dict(eps=lambda t: np.diag([-0.04 * t, 0.0, 0.0]),
                               key=None, ind='Displacement_Indicator',
                               bit=16, mag=10000.0, max_disp=0.005),
}

NUM_PTS = 8


def death_main(exo, path):
    cfg = DEATH_MODES[path]
    t, var = exo_open(exo)
    nsteps = len(t) - 1

    hist = ref.drive_permafrost(cfg['eps'], lambda t: 1.0, nsteps,
                                ref.SALEM_END, ref.SALEM_END, SALEM_SHARED)
    P = ref.permafrost_map(1.0, ref.SALEM_END, ref.SALEM_END, SALEM_SHARED)

    # LOCA's observer writes a stale duplicate of the initial state into
    # the first post-initial output slot (index 1); mask it out.
    mask = np.ones(len(t), dtype=bool)
    if len(t) > 1:
        mask[1] = False

    ok = True

    def check(name, a, b, scale, tol):
        nonlocal ok
        err = np.abs(np.asarray(a)[mask] - np.asarray(b)[mask]).max() / scale
        status = 'ok' if err < tol else 'FAIL'
        print(f'  {name:24s} max rel diff = {err:.3e}  [{status}]')
        ok &= err < tol

    print(f'== {path}: Albany ({exo}) vs cap_reference, {nsteps} steps ==')
    TOL = 1.0e-4

    # Trajectory integrity (same comparison as the base verification).
    o_sxx = np.array([h[1][0, 0] for h in hist])
    o_szz = np.array([h[1][2, 2] for h in hist])
    o_kap = np.array([h[2] for h in hist])
    scale_s = max(np.abs(o_sxx).max(), 1.0)
    check('sigma_xx', var('Cauchy_Stress_1_1'), o_sxx, scale_s, TOL)
    check('sigma_zz', var('Cauchy_Stress_9_1'), o_szz, scale_s, TOL)
    check('kappa', var('Cap_Parameter_1'), o_kap,
          max(np.abs(o_kap).max(), 1.0), TOL)

    # All four rational indicators, every mode (they are smooth functions
    # of the verified state, so they must match wherever the trajectory
    # does).
    max_eqps = cfg.get('max_eqps', 0.0)
    o_inds = {k: np.array([ref.indicators(h[1], h[4], h[2], h[5], P,
                                          maximum_eqps=max_eqps)[k]
                           for h in hist])
              for k in ('tension', 'backstress', 'crush', 'eqps')}
    for k, exo_name in (('tension', 'Tension_Indicator_1'),
                        ('backstress', 'Backstress_Indicator_1'),
                        ('crush', 'Crush_Indicator_1'),
                        ('eqps', 'Eqps_Indicator_1')):
        check(f'{k} indicator', var(exo_name), o_inds[k],
              max(np.abs(o_inds[k]).max(), 1.0), TOL)

    n_idx = np.arange(len(t))

    if path == 'death_displacement':
        # Confined compression of the unit cube: u_x(x, t) = -0.04 t x,
        # so the displacement norm at a Gauss point is 0.04 t x_qp with
        # x_qp = (1 -+ 1/sqrt(3))/2. Four points sit at each abscissa;
        # compare the sorted per-point indicator multiset, which is
        # independent of the integration-point numbering.
        md = cfg['max_disp']
        x_near = 0.5 - 0.5 / np.sqrt(3.0)
        x_far = 0.5 + 0.5 / np.sqrt(3.0)
        o_sorted = np.sort(np.array(
            [[0.04 * ti * x / md for x in (x_near,) * 4 + (x_far,) * 4]
             for ti in t]), axis=1)
        a_sorted = np.sort(np.array(
            [var(f'Displacement_Indicator_{p}') for p in range(1, NUM_PTS + 1)]).T,
            axis=1)
        check('disp indicator (sorted)', a_sorted, o_sorted, 1.0, 1.0e-12)

        # Per-group trip steps (strict > as in the kernel), then the
        # decimal failure_state and the cell-death predicate (all 8 points
        # must fail).
        trip_far = n_idx[np.array([0.04 * ti * x_far > md for ti in t])].min()
        trip_near = n_idx[np.array([0.04 * ti * x_near > md for ti in t])].min()
        o_state = 4.0 * cfg['mag'] * (n_idx >= trip_far) + 4.0 * cfg['mag'] * (n_idx >= trip_near)
        # cell_death lags the bitmask by one step here: without the ACE
        # solver there is no death-status vector, so the live-death write
        # is disabled and cell_death carries init()'s seed from the
        # previous step's converged mask.
        o_dead = (n_idx >= trip_near + 1).astype(float)
        a_modes_sum = sum(var(f'failure_modes_{p}') for p in range(1, NUM_PTS + 1))
        o_modes_sum = float(cfg['bit']) * (4.0 * (n_idx >= trip_far) + 4.0 * (n_idx >= trip_near))
        print(f'  trip steps: far group {trip_far}, near group {trip_near}'
              f' of {nsteps}')
        check('failure_modes (sum)', a_modes_sum, o_modes_sum, 1.0, 1.0e-12)
    else:
        # Homogeneous deformation: every integration point trips at the
        # same step, the first one whose converged indicator reaches the
        # deck threshold.
        ind = o_inds[cfg['key']]
        trips = n_idx[ind >= cfg['threshold']]
        assert trips.size > 0, 'oracle predicts no trip: bad test setup'
        trip = trips.min()
        assert 2 < trip < nsteps, f'trip step {trip} too close to the ends'
        print(f'  trip step: {trip} of {nsteps}')
        o_state = NUM_PTS * cfg['mag'] * (n_idx >= trip)
        # One-step lag: see the displacement branch above.
        o_dead = (n_idx >= trip + 1).astype(float)
        check('failure_modes', var('failure_modes_1'),
              float(cfg['bit']) * (n_idx >= trip), 1.0, 1.0e-12)

    check('failure_state', var('failure_state'), o_state,
          max(o_state.max(), 1.0), 1.0e-12)
    check('cell_death', var('cell_death'), o_dead, 1.0, 1.0e-12)

    print('VERIFICATION', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


def main():
    exo, path = sys.argv[1], sys.argv[2]
    if path in DEATH_MODES:
        return death_main(exo, path)
    p = ref.CapParams()

    if path == 'hydrostatic':
        eps = lambda t: -0.02 * t * ref.I3
    elif path == 'confined':
        eps = lambda t: np.diag([-0.04 * t, 0.0, 0.0])
    elif path == 'triaxial':
        eps = lambda t: np.diag([-0.025 * t, -0.008 * t, -0.012 * t])
        p = ref.CapParams(psi=0.8, L=3.5e-4, phi=0.085, theta=0.1, Q=20.0)
    elif path == 'confined_fd':
        # u = h(t) x  =>  F = I + diag(h)
        F_of_t = lambda t: np.diag([1.0 - 0.04 * t, 1.0, 1.0])
    elif path == 'triaxial_fd':
        F_of_t = lambda t: np.diag([1.0 - 0.025 * t, 1.0 - 0.008 * t, 1.0 - 0.012 * t])
        p = ref.CapParams(psi=0.8, L=3.5e-4, phi=0.085, theta=0.1, Q=20.0)
    elif path == 'permafrost_thaw':
        eps = lambda t: np.diag([-0.04 * t, 0.0, 0.0])
        # Two-point table [0,1] -> [1.0, 0.3]: same end-clamped linear
        # interpolation as the kernel's interpolate_table.
        f_of_t = lambda t: 1.0 + (0.3 - 1.0) * (t - 0.0) / (1.0 - 0.0)
    elif path == 'permafrost_f05':
        eps = lambda t: np.diag([-0.04 * t, 0.0, 0.0])
        f_of_t = lambda t: 0.5
    else:
        sys.exit(f'unknown path {path}')

    t, sxx, syy, szz, sxy, kappa, evp = exo_series(exo)
    nsteps = len(t) - 1
    if path.startswith('permafrost_'):
        hist = ref.drive_permafrost(eps, f_of_t, nsteps, ref.SALEM_END,
                                    ref.THAWED_TEST_END, ref.THAWED_TEST_SHARED)
    elif path.endswith('_fd'):
        hist = ref.drive_fd(F_of_t, nsteps, p)
    else:
        hist = ref.drive(eps, nsteps, p)

    # LOCA's observer writes a stale duplicate of the initial state into
    # the first post-initial output slot (index 1); mask it out.
    mask = np.ones(len(t), dtype=bool)
    if len(t) > 1:
        mask[1] = False

    o_sxx = np.array([h[1][0, 0] for h in hist])
    o_syy = np.array([h[1][1, 1] for h in hist])
    o_szz = np.array([h[1][2, 2] for h in hist])
    o_kap = np.array([h[2] for h in hist])
    o_evp = np.array([h[3] for h in hist])

    scale_s = max(np.abs(o_sxx).max(), 1.0)
    scale_k = max(np.abs(o_kap).max(), 1.0)
    scale_e = max(np.abs(o_evp).max(), 1.0e-12)

    ok = True

    def check(name, a, b, scale, tol):
        nonlocal ok
        err = np.abs(a[mask] - b[mask]).max() / scale
        status = 'ok' if err < tol else 'FAIL'
        print(f'  {name:24s} max rel diff = {err:.3e}  [{status}]')
        ok &= err < tol

    print(f'== {path}: Albany ({exo}) vs cap_reference, {nsteps} steps ==')
    # The oracle uses finite-difference gradients and an independent
    # transcription; agreement at 1e-3 of scale over the full history is a
    # strong match for a path-dependent explicit integration.
    TOL = 1.0e-4
    check('sigma_xx', sxx, o_sxx, scale_s, TOL)
    check('sigma_yy', syy, o_syy, scale_s, TOL)
    check('sigma_zz', szz, o_szz, scale_s, TOL)
    check('kappa', kappa, o_kap, scale_k, TOL)
    check('vol plastic strain', evp, o_evp, scale_e, TOL)

    if path == 'hydrostatic':
        # Analytic identities at the final state: I1 = X(kappa) on the
        # hydrostat, and evp on the crush curve.
        I1 = sxx[-1] + syy[-1] + szz[-1]
        X = ref.X_of_kappa(kappa[-1], p)
        evc = ref.evp_of_kappa(kappa[-1], p)
        e1 = abs(I1 - X) / abs(X)
        e2 = abs(evp[-1] - evc) / abs(evc)
        print(f'  I1 vs X(kappa)           rel diff = {e1:.3e}  '
              f'[{"ok" if e1 < 1e-2 else "FAIL"}]')
        print(f'  evp vs crush curve       rel diff = {e2:.3e}  '
              f'[{"ok" if e2 < 1e-2 else "FAIL"}]')
        ok &= e1 < 1e-2 and e2 < 1e-2

    print('VERIFICATION', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
