#!/usr/bin/env python3
"""Calibrate Permafrost (cap plasticity) end members from the
experimental fits behind J2Erosion.

The 2022 experimental program behind J2Erosion is encoded in its fit
polynomials (J2Erosion_Def.hpp, E_fit_max / Y_fit_max, 2025-03-05
revision: fits to purely experimental values), which give the stiffness
and strength multipliers as functions of ice saturation x and porosity
y. This script evaluates those fits at the end members (x = 1 frozen,
x = 0 thawed) for a given porosity and deck scales, maps them to cap
parameters, and emits the Frozen/Thawed Parameters YAML sublists.

Mapping rules (one per mechanism; see cap-plasticity.tex, "Planned
Extension: Permafrost and Erosion"):
  elasticity   (K, G) from the fitted E at the deck Poisson ratio.
               The thawed fit is negative (the experiments do not
               constrain the thawed branch), so the thawed member uses
               the deck residual floors -- the drained-skeleton choice.
               The undrained alternative (K floored at the Reuss/Wood
               saturated-mixture bound, ~3.5 GPa at porosity 0.6) is
               deliberately NOT used until pore pressure is resolved
               (u-p coupling, Phase 5): without it, an undrained K
               makes thawed mush bulk-stiffer than frozen soil while
               the nu cap erases the bound anyway when G collapses.
  cohesion     zero-pressure shear strength A - C = Y/sqrt(3) (the von
               Mises image of the fitted yield stress), N at 20% of it
               (kinematic-hardening room), A = 1.1c / C = 0.1c so the
               envelope is friction-dominated rather than
               exponential-dominated.
  friction     theta = 0.2309: Drucker-Prager triaxial-compression
               match to a 30-degree silty-sand friction angle;
               D = 4e-10 1/Pa (negligible curvature in the bluff
               stress range). Associative: L = D, phi = theta, Q = R.
               f-independent (sediment skeleton).
  crush        W from porosity (the kernel overrides W per point via
               ACE Bulk Porosity / ACE Porosity File; the deck value
               here is the same number). kappa0 COMMON to both end
               members (see the calibration note in the doc: a frozen
               state read under a thawed kappa0 is reinterpreted as
               precompacted), placed outside the gravity stress range;
               D1 sized so the crush curve develops over ~1 MPa.
  hardening    calpha sized so the backstress saturates over ~1% eqps.

Failure criteria (parity with the J2Erosion deck):
  Maximum Displacement and Critical Angle carry over unchanged.
  The distortion strain limit maps to an eqps limit: for simple shear,
  |C_dev|/sqrt(3) = limit corresponds to gamma = sqrt(3(limit - 1)/2)
  and eqps ~ gamma/sqrt(3).
  Backstress saturation (0.95) replaces yield-onset; tension (0.90)
  replaces the principal-stress check.

NOTE: Y_fit_max as deployed multiplies the leading constant by y
(-0.0419*y), while its comment says -0.0419. This script uses the
code's form for parity. The difference is under 1% of the frozen
strength; flagged for review.

Usage: calibrate_permafrost.py [porosity] [E0_Pa] [Y0_Pa] [Er_Pa] [Yr_Pa]
Defaults are the MiniErosion deck values: 0.60 1.0e9 3.0e6 1.0e4 5.0e3
"""

import sys

import math


def E_fit(x, y):
    """Stiffness multiplier (J2Erosion E_fit_max, 2025-03-05)."""
    return (-24.6901 - 167.6662 * x - 25.9496 * y + 819.0673 * x * y) / 600.7614


def Y_fit(x, y):
    """Strength multiplier (J2Erosion Y_fit_max, as coded: the leading
    constant is multiplied by y -- see module docstring)."""
    return (-0.0419 * y - 0.2972 * x - 0.0418 * y + 4.7013 * x * y) / 4.3204


def end_member(E, Y, nu, porosity, kappa0, D1):
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    c = Y / math.sqrt(3.0)          # zero-pressure shear strength A - C
    return dict(K=K, G=G, A=1.1 * c, C=0.1 * c, N=0.2 * c,
                kappa0=kappa0, W=porosity, D1=D1, calpha=0.2 * c / 0.01)


def calibrate(porosity=0.60, E0=1.0e9, Y0=3.0e6, Er=1.0e4, Yr=5.0e3,
              nu=0.2, strain_limit=1.05):
    kappa0 = -5.0e5     # common to both end members; |kappa0| >> rho g h
    D1 = 1.0e-6         # crush curve develops over ~1 MPa

    E_f = max(E_fit(1.0, porosity) * E0, Er)
    Y_f = max(Y_fit(1.0, porosity) * Y0, Yr)
    E_t = max(E_fit(0.0, porosity) * E0, Er)   # fit negative -> floor
    Y_t = max(Y_fit(0.0, porosity) * Y0, Yr)   # fit negative -> floor

    frozen = end_member(E_f, Y_f, nu, porosity, kappa0, D1)
    thawed = end_member(E_t, Y_t, nu, porosity, kappa0, D1)
    shared = dict(D=4.0e-10, theta=0.2309, L=4.0e-10, phi=0.2309,
                  R=5.0, Q=5.0, psi=1.0, D2=0.0)

    gamma = math.sqrt(3.0 * (strain_limit - 1.0) / 2.0)
    max_eqps = gamma / math.sqrt(3.0)

    return frozen, thawed, shared, max_eqps, (E_f, Y_f, E_t, Y_t)


def emit(frozen, thawed, shared, max_eqps, raw):
    E_f, Y_f, E_t, Y_t = raw
    print(f"# Calibrated from the J2Erosion experimental fits:")
    print(f"#   frozen E = {E_f:.6e} Pa, Y = {Y_f:.6e} Pa")
    print(f"#   thawed E = {E_t:.6e} Pa, Y = {Y_t:.6e} Pa (residual floors)")
    print("      Frozen Parameters:")
    for k in ("K", "G", "A", "C", "N", "kappa0", "W", "D1", "calpha"):
        print(f"        {k}: {frozen[k]:.16e}")
    print("      Thawed Parameters:")
    for k in ("K", "G", "A", "C", "N", "kappa0", "W", "D1", "calpha"):
        print(f"        {k}: {thawed[k]:.16e}")
    for k in ("D", "theta", "L", "phi", "R", "Q", "psi", "D2"):
        print(f"        {k}: {shared[k]}")
    print(f"# Maximum Equivalent Plastic Strain: {max_eqps:.4f}"
          f"  (image of the distortion strain limit)")


if __name__ == "__main__":
    args = [float(a) for a in sys.argv[1:]]
    emit(*calibrate(*args))
