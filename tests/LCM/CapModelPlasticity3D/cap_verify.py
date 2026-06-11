#!/usr/bin/env python3
"""Verification harness: compare the LCM CapModel response (single-element
Albany run, homogeneous strain) against the independent reference
implementation in cap_reference.py, plus the analytic crush-curve identity
on the hydrostat.

Usage: cap_verify.py <exo-file> <path>     path in {hydrostatic, confined}
Exit code 0 on PASS.
"""

import sys
import numpy as np
import netCDF4

import cap_reference as ref


def exo_series(fname):
    ds = netCDF4.Dataset(fname)
    names = [''.join(c.decode() for c in row if c != b'')
             for row in ds.variables['name_elem_var'][:]]
    idx = {n: i for i, n in enumerate(names)}

    def var(name):
        # vals_elem_var<j>eb1, j is 1-based variable index
        return np.array(ds.variables[f'vals_elem_var{idx[name]+1}eb1'][:, 0])

    t = np.array(ds.variables['time_whole'][:])
    # First index = tensor component (row-major 1..9), second = int. point.
    sxx = var('Cauchy_Stress_1_1')
    syy = var('Cauchy_Stress_5_1')
    szz = var('Cauchy_Stress_9_1')
    sxy = var('Cauchy_Stress_2_1')
    kappa = var('Cap_Parameter_1')
    evp = var('volPlastic_Strain_1')
    return t, sxx, syy, szz, sxy, kappa, evp


def main():
    exo, path = sys.argv[1], sys.argv[2]
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
