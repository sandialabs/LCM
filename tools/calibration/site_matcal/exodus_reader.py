"""Read a single-element LCM cap-model Exodus output into a MatCal ``Data``.

The cap verification decks run a single element with all DOFs prescribed, so
the FE problem *is* the material point: every element variable is a scalar
time series. This reader extracts those series (Cauchy stress components, the
cap hardening parameter kappa, and the volumetric plastic strain) and returns
them as a MatCal ``Data`` object keyed by field name, with ``time`` as the
independent variable.

Units are whatever the deck was run in, which for this harness is base SI:
stresses come back in Pa (see ``site_matcal.lcm_model.SALEM_LIMESTONE``).

MatCal calls a results-reader object as ``reader(file_path, file_type=None)``
(see ``matcal.core.models._ResultsInformation.read``), which is exactly the
signature of ``read_lcm_cap_exodus`` below. Register it on a model with
``model._set_results_reader_object(read_lcm_cap_exodus)`` -- the
``make_lcm_cap_model`` helper does this for you.
"""

import numpy as np

from matcal import convert_dictionary_to_data


def _open_exo(fname):
    """Open an Exodus (classic netCDF) file with whichever backend is present:
    netCDF4 if installed, else scipy.io.netcdf_file. Both expose
    ``ds.variables[name][:]`` as ndarrays."""
    try:
        import netCDF4
        return netCDF4.Dataset(fname)
    except ImportError:
        from scipy.io import netcdf_file
        # mmap=False so arrays stay valid after the file object is closed.
        return netcdf_file(fname, "r", mmap=False)


def _exo_accessor(ds):
    """Return ``(time, var)`` where ``var(name)`` fetches an element variable's
    scalar time series. Mirrors the extraction in the LCM cap_verify.py harness:
    element-variable names live in ``name_elem_var``; values live in
    ``vals_elem_var<j>eb1`` with ``j`` the 1-based variable index."""
    names = [b"".join(row).decode("ascii", "ignore").strip().strip("\x00")
             for row in ds.variables["name_elem_var"][:]]
    idx = {n: i for i, n in enumerate(names)}

    def var(name):
        # Cell-level states may or may not carry an _1 (integration-point)
        # suffix depending on layout; fall back to the suffixed name.
        if name not in idx and name + "_1" in idx:
            name = name + "_1"
        return np.array(ds.variables[f"vals_elem_var{idx[name] + 1}eb1"][:, 0])

    t = np.array(ds.variables["time_whole"][:])
    return t, var, idx


# Map MatCal field name -> Exodus element-variable name. Any entry whose
# Exodus variable is absent from a given run is silently skipped, so the same
# reader works across load paths / model variants.
_FIELD_MAP = {
    "stress_xx": "Cauchy_Stress_1_1",
    "stress_yy": "Cauchy_Stress_5_1",
    "stress_zz": "Cauchy_Stress_9_1",
    "stress_xy": "Cauchy_Stress_2_1",
    "kappa":     "Cap_Parameter_1",
    "evp":       "volPlastic_Strain_1",
}


def read_lcm_cap_exodus(file_path, file_type=None):
    """Read ``file_path`` (an LCM cap-model Exodus output) and return a MatCal
    ``Data`` with fields ``time`` plus whichever of ``stress_xx/yy/zz/xy``,
    ``kappa``, ``evp`` are present. ``file_type`` is accepted for MatCal's
    reader protocol and ignored (the format is always Exodus)."""
    ds = _open_exo(file_path)
    try:
        t, var, idx = _exo_accessor(ds)
        data = {"time": t}
        for field, exo_name in _FIELD_MAP.items():
            base = exo_name[:-2] if exo_name.endswith("_1") else exo_name
            if exo_name in idx or base in idx:
                data[field] = var(exo_name)
    finally:
        close = getattr(ds, "close", None)
        if callable(close):
            close()
    return convert_dictionary_to_data(data)
