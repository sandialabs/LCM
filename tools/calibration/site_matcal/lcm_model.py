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

from site_matcal.exodus_reader import read_lcm_cap_exodus
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
)


def make_lcm_cap_model(load_path="confined", albany=None, defaults=None,
                       platform=None, name=None, finite_deformation=True):
    """Return a configured ``UserExecutableModel`` for an LCM cap load path.

    Parameters
    ----------
    load_path : str
        One of :data:`site_matcal.load_paths.LOAD_PATHS`
        (``hydrostatic``/``confined``/``triaxial``).
    albany : str, optional
        Albany executable path. Default: the selected platform's Albany
        (``$LCM_ALBANY`` overrides).
    defaults : dict, optional
        Constant overrides merged onto ``SALEM_LIMESTONE`` for placeholders not
        being calibrated.
    platform : str, optional
        Platform name for Albany resolution / environment (default: auto).
    name : str, optional
        Model name and per-evaluation working-directory name
        (default ``lcm_cap_<load_path>``).
    finite_deformation : bool, optional
        Kinematics for the ``Finite Deformation`` flag in the materials file.
        Default True. See the KINEMATICS note in the module docstring.
    """
    lp = get_load_path(load_path)
    plat = get_platform(platform)
    plat.apply_env()                       # no-op where the platform needs no env
    albany = albany or get_albany(platform)
    # The flag is a jinja substitution like any other, but it is a YAML
    # boolean rather than a number, so it is kept out of SALEM_LIMESTONE
    # (which doubles as the whitelist of calibratable placeholders) and
    # rendered as a lowercase literal Albany's YAML parser accepts.
    constants = {**SALEM_LIMESTONE, **(defaults or {}),
                 "finite_deformation": "true" if finite_deformation else "false"}
    name = name or f"lcm_cap_{lp.name}"

    deck_path = os.path.join(TEMPLATES_DIR, lp.deck)
    materials_path = os.path.join(TEMPLATES_DIR, lp.materials)
    for p in (deck_path, materials_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"template not found: {p}")

    model = UserExecutableModel(albany, lp.deck, results_filename=lp.exodus)
    # Both files are copied into every evaluation working directory and, being
    # text, are jinja-rendered there. The deck has no placeholders (rendered
    # unchanged); the materials file gets the parameter substitutions.
    model.add_necessary_files(deck_path, materials_path)
    model.add_constants(**constants)
    model._set_results_reader_object(read_lcm_cap_exodus)
    model.set_name(name)
    return model
