"""Platform abstraction for running LCM/Albany under MatCal.

Different machines locate the Albany executable differently and may need
different environment setup (module loads, extra library paths, job dispatch).
This module keeps those differences in one place so the model/harness code is
platform-agnostic.

Two platforms are anticipated:

  * ``rigel`` -- the local workstation (current). The serial Albany binary
    resolves its Trilinos libraries via a baked-in RUNPATH, so no environment
    setup is required.
  * ``cee``   -- the SRN CEE LAN (next). STUBBED: fill in the Albany path and
    any module/environment setup when we move there.

Selection order:
  1. ``$LCM_MATCAL_PLATFORM`` (explicit: "rigel" or "cee"), else
  2. hostname match, else
  3. the local default (rigel).

Override just the executable with ``$LCM_ALBANY`` on any platform.
"""

import os
import socket


class Platform:
    """A calibration target platform.

    Parameters
    ----------
    name : str
        Platform identifier.
    albany : str
        Path to (or PATH name of) the Albany executable.
    env : dict, optional
        Extra environment variables to apply to the Albany subprocess
        (e.g. library paths). Applied by ``apply_env`` before a run.
    hostnames : tuple of str, optional
        Substrings matched against ``socket.gethostname()`` for auto-detection.
    """

    def __init__(self, name, albany, env=None, hostnames=()):
        self.name = name
        self.albany = albany
        self.env = dict(env or {})
        self.hostnames = tuple(hostnames)

    def resolve_albany(self):
        """The Albany path to use, honoring the $LCM_ALBANY override."""
        return os.environ.get("LCM_ALBANY", os.path.expanduser(self.albany))

    def apply_env(self):
        """Apply this platform's environment variables to the current process
        (inherited by the Albany subprocess MatCal launches). No-op on rigel."""
        for key, value in self.env.items():
            os.environ[key] = value

    def __repr__(self):
        return f"Platform(name={self.name!r}, albany={self.resolve_albany()!r})"


# --- rigel (local workstation) --------------------------------------------
RIGEL = Platform(
    name="rigel",
    albany="~/LCM/lcm-build-serial-gcc-release/src/Albany",
    env={},                    # serial build; Trilinos libs via RUNPATH
    hostnames=("rigel",),
)

# --- cee (SRN CEE LAN workstations, e.g. hpws*) -----------------------------
# The LCM Albany serial build lives at the same ~/LCM/... path as on rigel.
# Dakota is provided by the on-disk CEE install /projects/dakota/install/rhel8/
# 6.24.0 (cpython-312 bindings) and is wired via the conda env's activate hook
# (DAKOTA_ROOT), not here. env is empty because the serial Albany build
# resolves its Trilinos libraries via RUNPATH; if a freshly built CEE Albany
# turns out to need runtime module libraries, add them to `env` (or use a
# module-load wrapper script as the executable) -- see README
# "Adding a platform (CEE)". matcal itself is installed via conda+pip (the CEE
# matcal module is pinned to the older rhel8 analyst stack); see docs/SETUP.md.
CEE = Platform(
    name="cee",
    albany="~/LCM/lcm-build-serial-gcc-release/src/Albany",
    env={},                    # serial build; Trilinos libs via RUNPATH
    hostnames=("hpws", "hpv", "cee", "skybridge", "ghost", "eclipse", "attaway", "solo"),
)

_PLATFORMS = {p.name: p for p in (RIGEL, CEE)}


def get_platform(name=None):
    """Return the selected :class:`Platform`.

    ``name`` (or ``$LCM_MATCAL_PLATFORM``) forces a choice; otherwise the
    hostname is matched; otherwise rigel is used as the local default.
    """
    name = name or os.environ.get("LCM_MATCAL_PLATFORM")
    if name:
        try:
            return _PLATFORMS[name.lower()]
        except KeyError:
            raise KeyError(f"unknown platform {name!r}; known: {sorted(_PLATFORMS)}")
    host = socket.gethostname().lower()
    for platform in _PLATFORMS.values():
        if any(h in host for h in platform.hostnames):
            return platform
    return RIGEL


def get_albany(name=None):
    """Convenience: the Albany executable path for the selected platform."""
    return get_platform(name).resolve_albany()
