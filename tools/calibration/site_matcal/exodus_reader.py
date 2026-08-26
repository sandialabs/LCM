"""Read a single-element LCM cap-model Exodus output into a MatCal ``Data``.

The cap verification decks run a single element with all DOFs prescribed, so
the FE problem *is* the material point: every element variable is a scalar
time series. This reader extracts those series and returns them as a MatCal
``Data`` object keyed by field name.

Three families of field come back, which is what lets the harness calibrate
against whichever curve the experiment actually produced:

  * ``stress_xx/yy/zz/xy`` and ``strain_xx/yy/zz/xy`` -- the material-point
    Cauchy stress and strain, for a **stress-strain** curve;
  * ``displacement_x/y/z`` and ``force_x/y/z`` -- the displacement of, and the
    reaction force on, the loaded face, for a **load-displacement** curve;
  * ``time``, ``kappa`` and ``evp`` -- the LOCA continuation parameter, the cap
    hardening parameter, and the volumetric plastic strain.

Displacement and force are read from the nodal fields ``solution_<axis>`` and
``residual_<axis>`` on the face at the maximum coordinate along that axis:
the displacement is the mean over the face nodes and the force is their sum.
Because every DOF is prescribed, that sum is the reaction force, and for the
uniform single element it equals (Cauchy stress) x (face area) to roundoff.
The unit-cube mesh these decks generate has unit face area and unit edge
length, so the load-displacement curve is numerically equal to the
stress-strain curve; that is a property of this particular mesh, not an
identity, and the two are computed independently here.

Signs follow the simulation: compression is negative in both stress and
strain, and force and displacement share the sign of the stress and strain
they come from. Experimental data must use the same convention.

Units are whatever the deck was run in, which for this harness is base SI:
stresses in Pa, forces in N, displacements in m, strains dimensionless (see
``site_matcal.lcm_model.SALEM_LIMESTONE``).

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


def _names(ds, key):
    """Decode an Exodus name table (``name_elem_var``/``name_nod_var``) into a
    ``{name: index}`` dict. Returns an empty dict if the table is absent."""
    if key not in ds.variables:
        return {}
    names = [b"".join(row).decode("ascii", "ignore").strip().strip("\x00")
             for row in ds.variables[key][:]]
    return {n: i for i, n in enumerate(names)}


def _exo_accessor(ds):
    """Return ``(time, var, idx)`` where ``var(name)`` fetches an element
    variable's scalar time series. Mirrors the extraction in the LCM
    cap_verify.py harness: element-variable names live in ``name_elem_var``;
    values live in ``vals_elem_var<j>eb1`` with ``j`` the 1-based index."""
    idx = _names(ds, "name_elem_var")

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
#
# Tensor components are stored flattened row-major over a 3x3, so component
# (i, j) is index 3*(i-1) + j: xx -> 1, xy -> 2, yy -> 5, zz -> 9. The trailing
# _1 is the integration point.
_FIELD_MAP = {
    "stress_xx": "Cauchy_Stress_1_1",
    "stress_xy": "Cauchy_Stress_2_1",
    "stress_yy": "Cauchy_Stress_5_1",
    "stress_zz": "Cauchy_Stress_9_1",
    "strain_xx": "Strain_1_1",
    "strain_xy": "Strain_2_1",
    "strain_yy": "Strain_5_1",
    "strain_zz": "Strain_9_1",
    "kappa":     "Cap_Parameter_1",
    "evp":       "volPlastic_Strain_1",
}

_AXES = ("x", "y", "z")


def _load_displacement(ds):
    """Return ``{displacement_<axis>: ..., force_<axis>: ...}`` for each axis
    whose nodal fields and coordinates are present.

    The loaded face is the one at the maximum coordinate along the axis. Its
    displacement is the mean of ``solution_<axis>`` over the face nodes (they
    are equal for these homogeneous decks; the mean is simply robust) and its
    force is the sum of ``residual_<axis>``, which is the reaction because
    every DOF is prescribed. Returns ``{}`` when the deck writes no nodal
    output.
    """
    idx = _names(ds, "name_nod_var")
    if not idx:
        return {}

    def nodal(name):
        return np.array(ds.variables[f"vals_nod_var{idx[name] + 1}"][:])

    out = {}
    for axis in _AXES:
        coord_key = f"coord{axis}"
        disp_key, force_key = f"solution_{axis}", f"residual_{axis}"
        if coord_key not in ds.variables:
            continue
        if disp_key not in idx or force_key not in idx:
            continue
        coord = np.array(ds.variables[coord_key][:])
        span = coord.max() - coord.min()
        # Tolerance relative to the mesh extent; a degenerate (flat) direction
        # would select every node, so skip it rather than report nonsense.
        if span <= 0.0:
            continue
        face = np.flatnonzero(coord >= coord.max() - 1.0e-8 * span)
        out[f"displacement_{axis}"] = nodal(disp_key)[:, face].mean(axis=1)
        out[f"force_{axis}"] = nodal(force_key)[:, face].sum(axis=1)
    return out


def read_lcm_cap_exodus(file_path, file_type=None):
    """Read ``file_path`` (an LCM cap-model Exodus output) and return a MatCal
    ``Data`` with ``time`` plus whichever of ``stress_*``, ``strain_*``,
    ``displacement_*``, ``force_*``, ``kappa`` and ``evp`` the run wrote.
    ``file_type`` is accepted for MatCal's reader protocol and ignored (the
    format is always Exodus)."""
    ds = _open_exo(file_path)
    try:
        t, var, idx = _exo_accessor(ds)
        data = {"time": t}
        for field, exo_name in _FIELD_MAP.items():
            base = exo_name[:-2] if exo_name.endswith("_1") else exo_name
            if exo_name in idx or base in idx:
                data[field] = var(exo_name)
        data.update(_load_displacement(ds))
    finally:
        close = getattr(ds, "close", None)
        if callable(close):
            close()
    return convert_dictionary_to_data(data)
