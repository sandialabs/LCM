"""Build a MatCal ``UserExecutableModel`` that runs the LCM cap-plasticity
single-element (material-point) simulation for calibration.

The model, for a given load path:

  * renders the jinja-templated materials file (cap parameters) with the study
    parameters + the ``SALEM_LIMESTONE`` default constants in each MatCal
    evaluation working directory,
  * runs ``Albany <deck>`` there (an internally generated single-element STK
    mesh -- no mesh file needed; the serial Albany binary resolves its
    Trilinos libraries via a baked-in RUNPATH, so no module load is required
    on the supported workstations),
  * reads the resulting Exodus stress/strain series back through
    ``read_lcm_cap_exodus``.

The Albany path comes from :mod:`site_matcal.platforms` (rigel, sirius, cee).
Calibrate a subset of parameters by defining ``matcal.Parameter`` objects whose
names match the jinja placeholders; study parameters override the defaults
(MatCal precedence: study params > model constants > state vars).

Everything here is in **base SI**: stress in Pa, no magnitude prefixes,
magnitudes written in scientific notation. That covers the defaults below, the
bounds and initial values passed as ``matcal.Parameter``, the experimental
curves, and the stresses the Exodus reader returns. See ``SALEM_LIMESTONE``.

KINEMATICS. ``finite_deformation`` renders the materials file's ``Finite
Deformation`` flag and defaults to True. The finite-deformation kernel wraps
exponential/logarithmic-map kinematics around the same verified integrator,
working in logarithmic elastic strain and Kirchhoff stress and converting to
Cauchy on output; it consumes F, J and Fp and writes no Strain field at all,
which is why the harness reconstructs strain from nodal displacement (see
``site_matcal.exodus_reader``). The Salem limestone constants below were
identified under a small-strain formulation, so they remain a sound starting
point under finite deformation but are not the finite-deformation answer to
the same data.
"""

import os

from matcal.core.models import UserExecutableModel

from site_matcal.exodus_reader import (read_lcm_cap_exodus,
                                       PreloadTrimmedReader)
from site_matcal.platforms import get_albany, get_platform
from site_matcal.load_paths import get_load_path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.environ.get(
    "LCM_MATCAL_TEMPLATES",
    os.path.join(os.path.dirname(_THIS_DIR), "templates"),
)

# Salem limestone, associative parameter set: Table 1 of Sun, Chen & Ostien,
# Acta Geotechnica 9 (2014) 903-934, converted to base SI. Keys are the jinja
# placeholder names used in the templated materials files.
#
# UNITS. The cap model is unit-agnostic: it only requires one consistent unit
# system. This harness uses base SI throughout -- stress in Pa, no magnitude
# prefixes -- which is what the ACE/permafrost production decks use, so a set
# calibrated here can be pasted into one without rescaling. Magnitudes are
# written in scientific notation rather than as prefixed units (2.2547e10 Pa,
# not 22547 MPa). Sun, Chen & Ostien tabulate in MPa; each stress-like value
# below is therefore their value times 1e6, each 1/stress value theirs times
# 1e-6, and D2 (1/stress^2) theirs times 1e-12.
#
# Dimensions (sigma = Pa):
#   sigma        A, C, N, kappa0, calpha, elastic_modulus
#   1/sigma      D, L, D1
#   1/sigma^2    D2
#   dimensionless  poissons_ratio, theta, R, W, psi, phi, Q
#
# Experimental data fed to the harness must be in Pa as well: the objective
# compares raw stress values, so a curve in MPa would be fit by parameters
# 1e6 too small.
SALEM_LIMESTONE = dict(
    elastic_modulus=2.2547e10,   # Pa      (22547 MPa)
    poissons_ratio=0.2524,       # -
    A=6.892e8,                   # Pa      (689.2 MPa)
    D=3.94e-10,                  # 1/Pa    (3.94e-4 1/MPa)
    C=6.752e8,                   # Pa      (675.2 MPa)
    theta=0.0,                   # -
    R=28.0,                      # -
    kappa0=-8.05e6,              # Pa      (-8.05 MPa)
    W=0.08,                      # -
    D1=1.47e-9,                  # 1/Pa    (1.47e-3 1/MPa)
    D2=0.0,                      # 1/Pa^2
    calpha=1.0e11,               # Pa      (1.0e5 MPa)
    psi=1.0,                     # -
    N=6.0e6,                     # Pa      (6.0 MPa)
    L=3.94e-10,                  # 1/Pa    (3.94e-4 1/MPa)
    phi=0.0,                     # -
    Q=28.0,                      # -
    # Cohesion softening (CapSoftening.hpp), inert unless softening=True:
    # residual coherence, damage strain at half loss, failure speed.
    coherence_residual=1.0,      # -
    failure_strain=1.0,          # -
    failure_speed=1.0,           # -
)


# Frozen permafrost end member, the same parameter set as the Frozen
# Parameters block of tests/LCM/ACE/MiniErosionPermafrost/
# materials_mechanical_permafrost.yaml, with (K, G) = (5.5556e8, 4.1667e8)
# written back as (E, nu) = (1.0e9, 0.2) and the shape parameters D, theta, L,
# phi, R, Q, psi, D2 taken from the same file. Base SI, same keys as
# SALEM_LIMESTONE.
#
# This is the set to start a frozen-soil calibration from, and the one the txc
# path needs: SALEM_LIMESTONE is a rock whose cap sits at kappa0 = -8.05e6 Pa
# while its unconfined deviatoric strength is 3.5e7 Pa, so every triaxial test
# of it at a confinement inside its own cap is cap-dominated rather than a test
# of the shear envelope, and A and C (6.892e8 and 6.752e8, whose difference is
# the 1.4e7 that sets the strength) are not separately identifiable from one.
PERMAFROST_FROZEN = dict(
    elastic_modulus=1.0e9,               # Pa   (K = 5.5556e8, G = 4.1667e8)
    poissons_ratio=0.2,                  # -
    A=2.0e6,                             # Pa
    D=4.0e-10,                           # 1/Pa
    C=2.6794919243112305e5,              # Pa
    theta=0.10,                          # -
    R=5.0,                               # -
    kappa0=-1.0e7,                       # Pa
    W=0.60,                              # -    (the pore space; ACE porosity)
    D1=1.0e-8,                           # 1/Pa
    D2=0.0,                              # 1/Pa^2
    calpha=1.0e9,                        # Pa
    psi=1.0,                             # -
    N=3.4641016151377546e5,              # Pa
    L=4.0e-10,                           # 1/Pa
    phi=0.08,                            # -
    Q=5.0,                               # -
    coherence_residual=1.0,              # -    (softening inert by default)
    failure_strain=1.0,                  # -
    failure_speed=1.0,                   # -
)

# Named starting parameter sets, selected with --defaults. SALEM_LIMESTONE is
# the default because it is the set the three verification decks were built
# around; PERMAFROST_FROZEN is the one the txc path and any frozen-soil
# calibration wants.
DEFAULT_SETS = {
    "salem": SALEM_LIMESTONE,
    "permafrost": PERMAFROST_FROZEN,
}
DEFAULT_SET = "salem"

assert set(PERMAFROST_FROZEN) == set(SALEM_LIMESTONE), (
    "every named default set must cover exactly the calibratable placeholders")


# Kinematics used unless a caller says otherwise. This is the single place to
# change it: both the library default below and the --finite-deformation /
# --small-strain flag in harness/calibrate.py read it.
DEFAULT_FINITE_DEFORMATION = True


def make_lcm_cap_model(load_path="confined", albany=None, defaults=None,
                       platform=None, name=None,
                       finite_deformation=DEFAULT_FINITE_DEFORMATION,
                       softening=False):
    """Return a configured ``UserExecutableModel`` for an LCM cap load path.

    Parameters
    ----------
    load_path : str
        One of :data:`site_matcal.load_paths.LOAD_PATHS`
        (``hydrostatic``/``confined``/``triaxial``/``txc``).
    albany : str, optional
        Albany executable path. Default: the selected platform's Albany
        (``$LCM_ALBANY`` overrides).
    defaults : dict, optional
        Constant overrides merged onto ``SALEM_LIMESTONE`` and onto the load
        path's own deck constants, for placeholders not being calibrated. This
        is how a ``txc`` run selects its confining pressure.
    platform : str, optional
        Platform name for Albany resolution / environment (default: auto).
    name : str, optional
        Model name and per-evaluation working-directory name
        (default ``lcm_cap_<load_path>``).
    finite_deformation : bool, optional
        Kinematics for the ``Finite Deformation`` flag in the materials file.
        Default :data:`DEFAULT_FINITE_DEFORMATION`. See the KINEMATICS note in
        the module docstring.
    softening : bool, optional
        Enable cohesion softening (the ``Softening`` flag). Off by default;
        the three softening placeholders are inert until it is on.
    """
    lp = get_load_path(load_path)
    plat = get_platform(platform)
    plat.apply_env()                       # no-op where the platform needs no env
    albany = albany or get_albany(platform)
    # The flag is a jinja substitution like any other, but it is a YAML
    # boolean rather than a number, so it is kept out of SALEM_LIMESTONE
    # (which doubles as the whitelist of calibratable placeholders) and
    # rendered as a lowercase literal Albany's YAML parser accepts.
    # Deck constants (the confining pressure and the like) sit between the
    # material defaults and the caller's overrides: they are defaults too, but
    # they belong to the load path rather than to the material.
    constants = {**SALEM_LIMESTONE, **lp.constants, **(defaults or {}),
                 "finite_deformation": "true" if finite_deformation else "false",
                 "softening": "true" if softening else "false"}
    name = name or f"lcm_cap_{lp.name}"

    deck_path = os.path.join(TEMPLATES_DIR, lp.deck)
    materials_path = os.path.join(TEMPLATES_DIR, lp.materials)
    for p in (deck_path, materials_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"template not found: {p}")

    model = UserExecutableModel(albany, lp.deck, results_filename=lp.exodus)
    # Both files are copied into every evaluation working directory and, being
    # text, are jinja-rendered there. The materials file gets the parameter
    # substitutions; the three verification decks have no placeholders and are
    # rendered unchanged, while input_txc.yaml uses the deck constants and
    # derives its boundary conditions from the elastic constants, so it tracks
    # a calibrated elastic_modulus or poissons_ratio instead of going stale.
    model.add_necessary_files(deck_path, materials_path)
    model.add_constants(**constants)
    # A path that consolidates before it loads reports its curve from the end
    # of the consolidation; every other path reads the whole run.
    preload_time = lp.preload_time(constants)
    model._set_results_reader_object(
        PreloadTrimmedReader(preload_time) if preload_time > 0.0
        else read_lcm_cap_exodus)
    model.set_name(name)
    return model
