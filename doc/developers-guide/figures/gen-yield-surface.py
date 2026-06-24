#!/usr/bin/env python3
"""Generate the Permafrost cap-plasticity yield-surface illustration.

Renders the yield surface f = Gamma^2 J2 - Fc(I1,kappa) (Ff(I1) - N)^2 = 0
in principal-stress space, for the frozen (f=1) and thawed (f=0) end-member
parameter sets used in the ACE permafrost bluff simulation. Sign convention:
tension positive, compression negative (I1 < 0 under confinement), so the cap
sits on the negative-I1 side and the cone opens toward the tensile apex.

Re-run to regenerate doc/developers-guide/figures/yield-surface-permafrost.pdf
if the calibrated parameters change.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- Calibrated end-member parameters (ACE permafrost bluff deck) ----------
# Friction/shape parameters are f-independent (sediment skeleton, thawed set).
D, THETA, R, PSI = 4.0e-10, 0.2309, 5.0, 1.0
KAPPA0 = -5.0e5

FROZEN = dict(A=1.5707945680e6, C=1.4279950618e5, N=2.8559901236e5)
THAWED = dict(A=3.1754264805e4, C=2.8867513459e3, N=5.7735026919e3)


def Ff(I1, p):
    return p["A"] - p["C"] * np.exp(D * I1) - THETA * I1


def X_of_kappa(kappa, p):
    return kappa - R * Ff(kappa, p)


def apex_I1(p):
    # tensile apex: Ff(I1) = N  (bisection on the shear branch, I1 > 0)
    lo, hi = 0.0, 1.0e9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if Ff(mid, p) > p["N"] else (lo, mid)
    return 0.5 * (lo + hi)


def surface(p, kappa=KAPPA0, n_i1=160, n_theta=120):
    """Return principal-stress meshes (Pa) on the yield surface."""
    I1 = np.linspace(X_of_kappa(kappa, p), apex_I1(p), n_i1)
    th = np.linspace(0.0, 2.0 * np.pi, n_theta)
    I1g, thg = np.meshgrid(I1, th)

    ff = Ff(I1g, p)
    X = X_of_kappa(kappa, p)
    fc = np.where(I1g < kappa, 1.0 - (I1g - kappa) ** 2 / (X - kappa) ** 2, 1.0)
    fc = np.clip(fc, 0.0, 1.0)
    # PSI = 1  ->  Gamma = 1  ->  circular deviatoric cross-section
    sqrtJ2 = np.sqrt(fc) * np.clip(ff - p["N"], 0.0, None)

    c = 2.0 / np.sqrt(3.0) * sqrtJ2
    s1 = c * np.cos(thg)
    s2 = c * np.cos(thg - 2.0 * np.pi / 3.0)
    s3 = c * np.cos(thg + 2.0 * np.pi / 3.0)
    return I1g / 3.0 + s1, I1g / 3.0 + s2, I1g / 3.0 + s3


def meridian(p, kappa=KAPPA0, n=400):
    """Mean stress p=I1/3 and von Mises q=sqrt(3 J2) along the meridian (MPa)."""
    I1 = np.linspace(X_of_kappa(kappa, p), apex_I1(p), n)
    ff = Ff(I1, p)
    X = X_of_kappa(kappa, p)
    fc = np.where(I1 < kappa, 1.0 - (I1 - kappa) ** 2 / (X - kappa) ** 2, 1.0)
    fc = np.clip(fc, 0.0, 1.0)
    sqrtJ2 = np.sqrt(fc) * np.clip(ff - p["N"], 0.0, None)
    return I1 / 3.0 / 1e6, np.sqrt(3.0) * sqrtJ2 / 1e6


MPA = 1e6
fig = plt.figure(figsize=(11.0, 4.6))

# --- (a) 3D frozen yield surface ------------------------------------------
ax = fig.add_subplot(1, 2, 1, projection="3d")
X1, X2, X3 = (a / MPA for a in surface(FROZEN))
ax.plot_surface(
    X1, X2, X3, rstride=2, cstride=2, color="#4c72b0", alpha=0.55,
    linewidth=0.0, antialiased=True, shade=True,
)
# hydrostatic axis for reference
lim = np.array([X1.min(), X1.max()])
ax.plot(lim, lim, lim, color="0.35", lw=1.0, ls="--")
ax.set_xlabel(r"$\sigma_1$ [MPa]", labelpad=2)
ax.set_ylabel(r"$\sigma_2$ [MPa]", labelpad=2)
ax.set_zlabel(r"$\sigma_3$ [MPa]", labelpad=2)
ax.set_title("(a) frozen end member ($f=1$)", fontsize=10)
ax.view_init(elev=22, azim=-58)
ax.tick_params(labelsize=7)

# --- (b) meridian: frozen vs thawed, same axes ----------------------------
ax2 = fig.add_subplot(1, 2, 2)
for p, col, lab in ((FROZEN, "#4c72b0", "frozen ($f=1$)"),
                    (THAWED, "#c44e52", "thawed ($f=0$)")):
    pm, qm = meridian(p)
    ax2.plot(pm, qm, color=col, lw=1.8, label=lab)
    ax2.plot(pm, -qm, color=col, lw=1.8)
ax2.axhline(0, color="0.6", lw=0.6)
ax2.axvline(0, color="0.6", lw=0.6)
ax2.set_xlabel(r"mean stress $p = I_1/3$ [MPa]")
ax2.set_ylabel(r"von Mises stress $q = \sqrt{3 J_2}$ [MPa]")
ax2.set_title("(b) meridian profile (true relative scale)", fontsize=10)
ax2.legend(fontsize=8, loc="upper left")
ax2.annotate("tensile\napex", xy=(1.65, 0), xytext=(0.2, 2.3),
             fontsize=7, ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.7, color="0.4"))
ax2.annotate("cap\nclosure", xy=(-2.74, 0), xytext=(-2.4, 2.3),
             fontsize=7, ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.7, color="0.4"))

fig.tight_layout()
out = __file__.rsplit("/", 1)[0] + "/yield-surface-permafrost.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

# Echo key geometry for the caption / sanity check.
for name, p in (("frozen", FROZEN), ("thawed", THAWED)):
    print(f"{name}: A-C-N={p['A']-p['C']-p['N']:.4g} Pa  "
          f"apex I1={apex_I1(p):.4g} Pa  cap X={X_of_kappa(KAPPA0, p):.4g} Pa")
