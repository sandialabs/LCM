#!/usr/bin/env python3
"""Independent reference implementation of the three-invariant cap
plasticity model, transcribed directly from the publications:

  [3] Sun, Chen & Ostien, Acta Geotechnica 9 (2014) 903-934, Section 3
      (yield/potential functions, hardening laws, explicit Algorithms 1-2),
  [2] Regueiro & Foster, IJNAMG 35 (2011) 201-225, eq. (14)
      (X^g built from the yield failure function F_f),

with the two as-printed errata in [3] corrected:
  - Algorithm 1, Step 2 omits C^e in the stress update (the Delta-gamma
    derivation and Algorithm 2 both include it),
  - Algorithm 2's normal-correction fallback mixes C^e into the update
    while omitting it from chi-tilde; the consistent Sloan et al. (2001)
    form (pure geometric projection, no C^e) is used,
and two conventions arbitrated by the LAME GeoModel reference
implementation (iso_geomodel_model.F):
  - the crush curve is evaluated at the YIELD cap position
    X(kappa) = kappa - R*F_f(kappa),
  - dkappa <= 0 (no cap contraction).

This file is the verification oracle for the LCM CapModel kernel: it is
a deliberately separate transcription, NOT a port of the C++.

Usage:
  cap_reference.py selfcheck          run internal FD checks + crush curve
  cap_reference.py hydrostatic        print t, sigma, kappa, evp history
  cap_reference.py confined           idem for 1D confined compression
"""

import sys
import numpy as np

I3 = np.eye(3)


class CapParams:
    """Salem limestone, associative set: Table 1 of [3]. Units: MPa."""

    def __init__(self, **kw):
        self.E      = 22547.0
        self.nu     = 0.2524
        self.A      = 689.2
        self.C      = 675.2
        self.D      = 3.94e-4
        self.theta  = 0.0
        self.L      = 3.94e-4
        self.phi    = 0.0
        self.R      = 28.0
        self.Q      = 28.0
        self.kappa0 = -8.05
        self.W      = 0.08
        self.D1     = 1.47e-3
        self.D2     = 0.0
        self.calpha = 1.0e5
        self.psi    = 1.0
        self.N      = 6.0
        for k, v in kw.items():
            assert hasattr(self, k), k
            setattr(self, k, v)

    @property
    def lame(self):
        return self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)

    @property
    def mu(self):
        return self.E / 2 / (1 + self.nu)

    def Ce(self, eps):
        """Isotropic elastic tangent applied to a strain tensor."""
        return self.lame * np.trace(eps) * I3 + 2 * self.mu * eps


def invariants(sigma, alpha):
    xi = sigma - alpha
    I1 = np.trace(xi)
    s = xi - I1 / 3 * I3
    J2 = 0.5 * np.tensordot(s, s)
    J3 = np.linalg.det(s)
    return I1, J2, J3, s


def _re(x):
    return x.real if np.iscomplexobj(x) else x


def Gamma(J2, J3, psi):
    """Lode-angle function, eq. after the beta definition in [3]:
    Gamma = (1/2)[1 - q + (1+q)/psi], q = 3 sqrt(3) J3 / (2 J2^{3/2})."""
    if psi == 0 or _re(J2) == 0:
        return 1.0
    q = 3 * np.sqrt(3.0) * J3 / (2 * J2 ** 1.5)
    return 0.5 * (1 - q + (1 + q) / psi)


def Ff(I1, p):
    return p.A - p.C * np.exp(p.D * I1) - p.theta * I1


def Ffg(I1, p):
    return p.A - p.C * np.exp(p.L * I1) - p.phi * I1


def X_of_kappa(kappa, p):
    return kappa - p.R * Ff(kappa, p)


def Xg_of_kappa(kappa, p):
    # [2] eq. (14): X^g = kappa - Q * F_f(kappa) with the YIELD F_f.
    return kappa - p.Q * Ff(kappa, p)


def Fc(I1, kappa, X):
    if _re(I1) >= _re(kappa) or X == kappa:
        return 1.0
    return 1.0 - ((I1 - kappa) / (X - kappa)) ** 2


def yield_f(sigma, alpha, kappa, p):
    I1, J2, J3, _ = invariants(sigma, alpha)
    G = Gamma(J2, J3, p.psi)
    return G * G * J2 - Fc(I1, kappa, X_of_kappa(kappa, p)) * (Ff(I1, p) - p.N) ** 2


def potential_g(sigma, alpha, kappa, p):
    I1, J2, J3, _ = invariants(sigma, alpha)
    G = Gamma(J2, J3, p.psi)
    return G * G * J2 - Fc(I1, kappa, Xg_of_kappa(kappa, p)) * (Ffg(I1, p) - p.N) ** 2


def num_grad_sigma(fun, sigma, alpha, kappa, p, h=1.0e-20):
    """Complex-step gradient w.r.t. sigma: machine-exact for the analytic
    branches (the Heaviside switch is evaluated on the real part)."""
    g = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            dp = np.zeros((3, 3), dtype=complex); dp[i, j] = 1j * h
            g[i, j] = fun(sigma.astype(complex) + dp, alpha, kappa, p).imag / h
    return g


def num_dfdkappa(sigma, alpha, kappa, p, h=1.0e-20):
    return yield_f(sigma, alpha, complex(kappa, h), p).imag / h


def evp_of_kappa(kappa, p):
    """Crush curve at the YIELD cap position (LAME reference convention)."""
    dX = X_of_kappa(kappa, p) - X_of_kappa(p.kappa0, p)
    return p.W * (np.exp(p.D1 * dX - p.D2 * dX * dX) - 1.0)


def dedkappa(kappa, p, h=1.0e-20):
    """d(eps_v^p)/d(kappa) by complex step (no shared closed form with C++)."""
    return evp_of_kappa(complex(kappa, h), p).imag / h


def halpha(dgds, alpha, p):
    J2a = 0.5 * np.tensordot(alpha, alpha)
    # Clamped at zero per the LAME GeoModel reference (GFUN floored at 0):
    # kinematic hardening stops at the limit surface, never reverses.
    Galpha = max(0.0, 1.0 - np.sqrt(J2a) / p.N) if p.N != 0 else 0.0
    dev = dgds - np.trace(dgds) / 3 * I3
    return p.calpha * Galpha * dev


def hkappa(dgds, kappa, p):
    ded = dedkappa(kappa, p)
    if ded == 0.0:
        return 0.0
    h = np.trace(dgds) / ded
    # Cap-lock saturation (LAME reference): crush curve exhausted.
    if abs(evp_of_kappa(kappa, p)) >= abs(p.W):
        K = p.lame + 2 * p.mu / 3
        h = -0.01 * K * K
    return h


def plastic_rates(sigma, alpha, kappa, deps_sub, p):
    """Forward-Euler increments at the given state for substrain deps_sub.
    dgamma clamped at zero (elastic unloading within a substep); dkappa
    clamped from above (no cap contraction)."""
    dfds = num_grad_sigma(yield_f, sigma, alpha, kappa, p)
    dgds = num_grad_sigma(potential_g, sigma, alpha, kappa, p)
    dfda = -dfds
    ha = halpha(dgds, alpha, p)
    hk = hkappa(dgds, kappa, p)
    dfdk = num_dfdkappa(sigma, alpha, kappa, p)
    Cedg = p.Ce(dgds)
    chi = np.tensordot(dfds, Cedg) - np.tensordot(dfda, ha) - dfdk * hk
    dgam = np.tensordot(p.Ce(dfds), deps_sub) / chi if chi != 0 else 0.0
    dgam = max(dgam, 0.0)
    dsig = p.Ce(deps_sub) - dgam * Cedg
    dalp = dgam * ha
    dkap = min(dgam * hk, 0.0)
    return dsig, dalp, dkap


def drift_correct(sigma, alpha, kappa, p, tol, max_iter=20):
    """Algorithm 2 of [3] with the Sloan normal-correction fallback."""
    for _ in range(max_iter + 1):
        f = yield_f(sigma, alpha, kappa, p)
        if abs(f) < tol:
            break
        dfds = num_grad_sigma(yield_f, sigma, alpha, kappa, p)
        dgds = num_grad_sigma(potential_g, sigma, alpha, kappa, p)
        dfda = -dfds
        ha = halpha(dgds, alpha, p)
        hk = hkappa(dgds, kappa, p)
        dfdk = num_dfdkappa(sigma, alpha, kappa, p)
        Cedg = p.Ce(dgds)
        chi = np.tensordot(dfds, Cedg) - np.tensordot(dfda, ha) - dfdk * hk
        dg = f / chi if chi != 0 else 0.0
        dk = min(dg * hk, 0.0)
        sigma_k = sigma - dg * Cedg
        alpha_k = alpha + dg * ha
        kappa_k = kappa + dk
        if abs(yield_f(sigma_k, alpha_k, kappa_k, p)) > abs(f):
            denom = np.tensordot(dfds, dfds)
            dg = f / denom if denom != 0 else 0.0
            sigma_k = sigma - dg * dfds
            alpha_k = alpha
            kappa_k = kappa
        sigma, alpha, kappa = sigma_k, alpha_k, kappa_k
    return sigma, alpha, kappa


def integrate_step(sigma_n, alpha_n, kappa_n, deps, p,
                   tol_scale=None, stol=1.0e-4, max_substeps=200):
    """Sloan-style adaptive substepping: modified-Euler (RK1/RK2) pairs
    with relative stress-error control; each accepted substep is followed
    by the drift correction. Mirrors the C++ kernel exactly."""
    sigma_tr = sigma_n + p.Ce(deps)
    if yield_f(sigma_tr, alpha_n, kappa_n, p) <= 0.0:
        return sigma_tr, alpha_n.copy(), kappa_n

    tol = 1.0e-12 * p.E * p.E if tol_scale is None else tol_scale
    dT_min = 1.0 / max_substeps

    sigma, alpha, kappa = sigma_n.copy(), alpha_n.copy(), kappa_n
    T, dT = 0.0, 1.0
    nsub, nsub_max = 0, 4 * max_substeps
    while T < 1.0 and nsub < nsub_max:
        deps_sub = dT * deps
        dsig1, dalp1, dkap1 = plastic_rates(sigma, alpha, kappa, deps_sub, p)
        dsig2, dalp2, dkap2 = plastic_rates(sigma + dsig1, alpha + dalp1,
                                            kappa + dkap1, deps_sub, p)
        sig_new = sigma + 0.5 * (dsig1 + dsig2)
        alp_new = alpha + 0.5 * (dalp1 + dalp2)
        kap_new = kappa + 0.5 * (dkap1 + dkap2)

        err = np.linalg.norm(dsig2 - dsig1)
        scale = max(np.linalg.norm(sig_new), 1.0e-12)
        R = 0.5 * err / scale

        if R > stol and dT > dT_min:
            q = max(0.9 * np.sqrt(stol / R), 0.1)
            dT = max(q * dT, dT_min)
            nsub += 1
            continue

        sigma, alpha, kappa = drift_correct(sig_new, alp_new, kap_new, p, tol)
        T += dT
        q = min(0.9 * np.sqrt(stol / max(R, 1.0e-30)), 1.1)
        dT = min(q * dT, 1.0 - T)
        if dT < dT_min:
            dT = min(dT_min, 1.0 - T)
        nsub += 1
    return sigma, alpha, kappa


def drive(eps_of_t, nsteps, p):
    """Integrate a strain history; returns list of (t, sigma, kappa, evp)."""
    sigma = np.zeros((3, 3))
    alpha = np.zeros((3, 3))
    kappa = p.kappa0
    eps_prev = eps_of_t(0.0)
    out = [(0.0, sigma.copy(), kappa, 0.0)]
    evp = 0.0
    Cinv_vol = 1.0 / (3 * p.lame + 2 * p.mu)   # for volumetric plastic strain
    for n in range(1, nsteps + 1):
        t = n / nsteps
        eps = eps_of_t(t)
        deps = eps - eps_prev
        sigma_tr = sigma + p.Ce(deps)
        sigma_new, alpha, kappa = integrate_step(sigma, alpha, kappa, deps, p)
        # plastic strain increment = C^-1 : (sigma_tr - sigma_new); only the
        # volumetric part is tracked here.
        dsig = sigma_tr - sigma_new
        devolps = Cinv_vol * np.trace(dsig)
        evp += devolps
        sigma = sigma_new
        eps_prev = eps
        out.append((t, sigma.copy(), kappa, evp))
    return out


def _logm_sym(A):
    w, V = np.linalg.eigh(A)
    return (V * np.log(w)) @ V.T


def _expm_sym(A):
    w, V = np.linalg.eigh(A)
    return (V * np.exp(w)) @ V.T


def _sqrtm_sym(A):
    w, V = np.linalg.eigh(A)
    return (V * np.sqrt(w)) @ V.T


def compliance(tau, p):
    """C^-1 : tau for the isotropic elastic tangent."""
    lam, mu = p.lame, p.mu
    return tau / (2 * mu) - lam / (2 * mu * (3 * lam + 2 * mu)) * np.trace(tau) * I3


def drive_fd(F_of_t, nsteps, p):
    """Finite-deformation driver mirroring the kernel's exp/log-map
    kinematics: the small-strain integrator runs unchanged in logarithmic
    elastic strain / Kirchhoff stress space. Returns (t, cauchy, kappa,
    evp) tuples."""
    alpha = np.zeros((3, 3))
    kappa = p.kappa0
    Fp = np.eye(3)
    F_prev = F_of_t(0.0)
    out = [(0.0, np.zeros((3, 3)), kappa, 0.0)]
    evp = 0.0
    Cinv_vol = 1.0 / (3 * p.lame + 2 * p.mu)
    for n in range(1, nsteps + 1):
        t = n / nsteps
        F = F_of_t(t)
        Fpinv = np.linalg.inv(Fp)
        Cpinv = Fpinv @ Fpinv.T
        eps_tr = 0.5 * _logm_sym(F @ Cpinv @ F.T)
        eps_e_n = 0.5 * _logm_sym(F_prev @ Cpinv @ F_prev.T)
        deps = eps_tr - eps_e_n
        tau_n = p.Ce(eps_e_n)
        tau_tr = tau_n + p.Ce(deps)
        tau, alpha, kappa = integrate_step(tau_n, alpha, kappa, deps, p)
        evp += Cinv_vol * np.trace(tau_tr - tau)
        # plastic update: elastic log strain consistent with tau
        be_new = _expm_sym(2.0 * compliance(tau, p))
        Finv = np.linalg.inv(F)
        Cpinv_new = Finv @ be_new @ Finv.T
        Fp = _sqrtm_sym(np.linalg.inv(Cpinv_new))
        out.append((t, tau / np.linalg.det(F), kappa, evp))
        F_prev = F
    return out


def selfcheck():
    p = CapParams()
    ok = True

    # 1. FD-vs-FD consistency is trivially true here; instead check the
    #    closed-form invariant identities at a general (non-principal) state.
    rng = np.random.default_rng(42)
    M = rng.normal(size=(3, 3))
    sigma = -50.0 * I3 + 5.0 * (M + M.T)   # compressive + random shear
    alpha = np.zeros((3, 3))
    a = rng.normal(size=(3, 3)); a = a + a.T
    alpha = 0.5 * (a - np.trace(a) / 3 * I3)   # deviatoric backstress
    kappa = -20.0

    # dJ3/dsigma identity: numerical gradient of J3 must equal
    # s.s - (2/3) J2 I (the matrix-product form -- the bug we fixed).
    def J3fun(s_, a_, k_, p_):
        return invariants(s_, a_)[2]
    nJ3 = num_grad_sigma(J3fun, sigma, alpha, kappa, p)
    I1, J2, J3, s = invariants(sigma, alpha)
    closed = s @ s - 2.0 / 3.0 * J2 * I3
    elementwise = s * s - 2.0 / 3.0 * J2 * I3
    err_closed = np.abs(nJ3 - closed).max()
    err_elem = np.abs(nJ3 - elementwise).max()
    print(f"dJ3/dsigma: |FD - matrix-product| = {err_closed:.3e}  "
          f"(elementwise-square error would be {err_elem:.3e})")
    ok &= err_closed < 1.0e-5 and err_elem > 1.0e-1

    # 2. Hydrostatic crush-curve identity: on the hydrostat the converged
    #    stress sits at I1 = X(kappa) and evp must equal the crush curve.
    p1 = CapParams()
    hist = drive(lambda t: -0.02 * t * I3, 200, p1)
    t, sig, kap, evp = hist[-1]
    I1f = np.trace(sig)
    Xf = X_of_kappa(kap, p1)
    evp_curve = evp_of_kappa(kap, p1)
    print(f"hydrostatic end: I1 = {I1f:.4f}  X(kappa) = {Xf:.4f}  "
          f"rel diff = {abs(I1f-Xf)/abs(Xf):.3e}")
    print(f"hydrostatic end: evp = {evp:.6e}  crush curve = {evp_curve:.6e}  "
          f"rel diff = {abs(evp-evp_curve)/abs(evp_curve):.3e}")
    ok &= abs(I1f - Xf) / abs(Xf) < 1.0e-2
    ok &= abs(evp - evp_curve) / abs(evp_curve) < 1.0e-2

    print("SELFCHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def print_history(hist):
    print("# t  sxx syy szz sxy sxz syz  kappa  evp")
    for t, sig, kap, evp in hist:
        print(f"{t:.6f} {sig[0,0]:.8e} {sig[1,1]:.8e} {sig[2,2]:.8e} "
              f"{sig[0,1]:.8e} {sig[0,2]:.8e} {sig[1,2]:.8e} "
              f"{kap:.8e} {evp:.8e}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "selfcheck"
    if mode == "selfcheck":
        sys.exit(selfcheck())
    elif mode == "hydrostatic":
        print_history(drive(lambda t: -0.02 * t * I3, 200, CapParams()))
    elif mode == "confined":
        eps = lambda t: np.diag([-0.04 * t, 0.0, 0.0])
        print_history(drive(eps, 200, CapParams()))
    else:
        sys.exit(f"unknown mode {mode}")
