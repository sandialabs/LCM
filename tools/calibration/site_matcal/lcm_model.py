"""Build a MatCal ``UserExecutableModel`` that runs the LCM cap-plasticity
single-element (material-point) simulation for calibration.

The model, for a given load path:

  * renders the jinja-templated materials file (cap parameters) with the study
    parameters + the ``SALEM_LIMESTONE`` default constants in each MatCal
    evaluation working directory,
  * runs ``Albany <deck>`` there (an internally generated single-element STK
    mesh -- no mesh file needed; the serial Albany binary resolves its
    Trilinos libraries via a baked-in RUNPATH, so no module load is required
    on rigel),
  * reads the resulting Exodus stress/strain series back through
    ``read_lcm_cap_exodus``.

The Albany path comes from :mod:`site_matcal.platforms` (rigel now, cee next).
Calibrate a subset of parameters by defining ``matcal.Parameter`` objects whose
names match the jinja placeholders; study parameters override the defaults
(MatCal precedence: study params > model constants > state vars).
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
# Acta Geotechnica 9 (2014) 903-934. Units: MPa. Keys are the jinja
# placeholder names used in the templated materials files.
SALEM_LIMESTONE = dict(
    elastic_modulus=22547.0,
    poissons_ratio=0.2524,
    A=689.2,
    D=0.000394,
    C=675.2,
    theta=0.0,
    R=28.0,
    kappa0=-8.05,
    W=0.08,
    D1=0.00147,
    D2=0.0,
    calpha=100000.0,
    psi=1.0,
    N=6.0,
    L=0.000394,
    phi=0.0,
    Q=28.0,
)


def make_lcm_cap_model(load_path="confined", albany=None, defaults=None,
                       platform=None, name=None):
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
    """
    lp = get_load_path(load_path)
    plat = get_platform(platform)
    plat.apply_env()                       # no-op on rigel; CEE env when added
    albany = albany or get_albany(platform)
    constants = {**SALEM_LIMESTONE, **(defaults or {})}
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
