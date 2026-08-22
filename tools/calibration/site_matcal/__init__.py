"""site_matcal: LCM/Albany integration for MatCal cap-plasticity calibration.

MatCal imports this package as ``from site_matcal import *`` at the end of its
own package initialization, so names listed in ``__all__`` here also become
available as ``matcal.<name>``. Importing this package runs the factory
registration in ``register_factories`` (clean shell environment + jinja
templating of input decks).

Make it importable by putting the ``tools/calibration`` directory on
PYTHONPATH (the ``matcal`` conda env's activate hook does this).
"""

__all__ = []

# Side-effect import: registers site factories on import.
from . import register_factories  # noqa: F401

from .exodus_reader import read_lcm_cap_exodus
from .lcm_model import make_lcm_cap_model, SALEM_LIMESTONE, TEMPLATES_DIR
from .platforms import get_platform, get_albany, Platform
from .load_paths import LOAD_PATHS, get_load_path, LoadPath

__all__ += [
    "read_lcm_cap_exodus",
    "make_lcm_cap_model",
    "SALEM_LIMESTONE",
    "TEMPLATES_DIR",
    "get_platform",
    "get_albany",
    "Platform",
    "LOAD_PATHS",
    "get_load_path",
    "LoadPath",
]
