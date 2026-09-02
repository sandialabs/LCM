"""Read a single-element LCM cap-model Exodus output into a MatCal ``Data``.

The cap verification decks run a single element with all DOFs prescribed, so
the FE problem *is* the material point: every element variable is a scalar
time series. This reader extracts those series and returns them as a MatCal
``Data`` object keyed by field name.

Several families of field come back, which is what lets the harness calibrate
against whichever curve the experiment actually produced:

  * ``displacement_x/y/z`` and ``force_x/y/z`` -- the displacement of, and the
    reaction force on, the loaded face. These are the raw measured
    quantities and mean the same thing under both kinematics.
  * ``strain_eng_x/y/z`` and ``stress_eng_x/y/z`` -- engineering (nominal)
    strain ``u/L0`` and stress ``force/A0``, both referred to the
    **undeformed** geometry.
  * ``strain_log_x/y/z`` -- logarithmic (true) strain ``ln(1 + u/L0)``, the
    measure work-conjugate to the Cauchy stress the model reports.
  * ``stress_xx/yy/zz/xy`` -- the material-point **Cauchy** (true) stress.
  * ``strain_xx/yy/zz/xy`` -- the model's own small-strain tensor. Present
    ONLY under ``Finite Deformation: false``; the finite-deformation kernel
    consumes ``F`` and ``Fp`` and never forms this field.
  * ``stress_dev_x/y/z`` -- the deviatoric (differential) Cauchy stress about
    that axis, ``sigma_aa - (sigma_bb + sigma_cc)/2``. On a triaxial path with
    equal lateral stresses this is exactly the ``sigma_1 - sigma_3`` a
    laboratory reports, negative in compression.
  * ``time``, ``kappa`` and ``evp`` -- the LOCA continuation parameter, the cap
    hardening parameter, and the volumetric plastic strain.

PRELOAD. ``preload_time`` (default 0.0, which changes nothing) is for load
paths that consolidate before they load, i.e. ``txc``. Steps before it are
dropped, and displacement, strain and the reference area are rebased to the
state at that time: the curve then starts at zero strain from the consolidated
configuration, which is where a laboratory triaxial test starts its axial
strain. Lengths and areas are the CONSOLIDATED ones, so ``strain_eng`` and
``stress_eng`` are referred to the specimen as it was when shearing began. At
``preload_time = 0.0`` the rebasing is the identity (the run starts at zero
displacement), so the other load paths are untouched.

REACTION FORCES. ``force_<axis>`` is the reaction only where that axis's face
is displacement-controlled. On the ``txc`` path the lateral faces carry
tractions and their displacements are free, so ``force_y``/``force_z`` are
converged residuals near zero and ``stress_eng_y``/``stress_eng_z`` are
meaningless. The axial direction is prescribed on every path, so
``force_x``/``stress_eng_x`` are always the reaction.

Displacement and force are read from the nodal fields ``solution_<axis>`` and
``residual_<axis>`` on the face at the maximum coordinate along that axis:
the displacement is the mean over the face nodes and the force is their sum.
Because every DOF is prescribed, that sum is the reaction force on the
**deformed** face.

The strain measures are reconstructed from that displacement rather than read
from the model. For these decks the deformation is exactly homogeneous (a
single element with every DOF prescribed), so ``u/L0`` is the engineering
strain to roundoff, and the reconstruction is therefore available under both
kinematics. That matters because the finite-deformation kernel writes no
strain field at all.

The distinction between the two stress measures only appears under finite
deformation, and only when the loaded face changes area. Confined compression
holds both lateral strains at zero, so its face area is preserved and
``force/A0`` equals the Cauchy stress exactly; hydrostatic compression shrinks
the face, and the two differ by the area ratio (measured 0.9604 = 0.98^2 at
2 per cent nominal strain). Under small strain they coincide on every path.

Signs follow the simulation: compression is negative in stress and strain
alike, and force and displacement share the sign of the stress and strain they
come from. Experimental data must use the same convention.

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


def _load_displacement(ds, start=0):
    """Return the nodal-derived fields for each axis whose nodal output and
    coordinates are present: ``displacement_<axis>``, ``force_<axis>``,
    ``strain_eng_<axis>``, ``strain_log_<axis>`` and ``stress_eng_<axis>``.

    The loaded face is the one at the maximum coordinate along the axis. Its
    displacement is the mean of ``solution_<axis>`` over the face nodes (they
    are equal for these homogeneous decks; the mean is simply robust) and its
    force is the sum of ``residual_<axis>``, which is the reaction wherever
    that face's DOFs are prescribed (see REACTION FORCES in the module
    docstring).

    ``start`` is the index of the step the strain measures are referred to:
    0 (the default) for a path that loads from the undeformed state, the end
    of the consolidation stage for one that preloads. The extent along each
    axis and the area normal to it are taken at that step, so they are the
    undeformed ones when ``start`` is 0 and the consolidated ones otherwise.
    Returns ``{}`` when the deck writes no nodal output.
    """
    idx = _names(ds, "name_nod_var")
    if not idx:
        return {}

    def nodal(name):
        return np.array(ds.variables[f"vals_nod_var{idx[name] + 1}"][:])

    extent = {}
    for axis in _AXES:
        key = f"coord{axis}"
        if key in ds.variables:
            coord = np.array(ds.variables[key][:])
            extent[axis] = (coord, coord.max() - coord.min())

    # Extent along each axis at the reference step: the undeformed extent plus
    # whatever that face had already moved by then. Collected first because the
    # area normal to one axis is built from the other two.
    ref_length = {}
    disp_of = {}
    for axis in _AXES:
        disp_key, force_key = f"solution_{axis}", f"residual_{axis}"
        if axis not in extent or disp_key not in idx or force_key not in idx:
            continue
        coord, length = extent[axis]
        # A degenerate (flat) direction would select every node, so skip it
        # rather than report nonsense.
        if length <= 0.0:
            continue
        face = np.flatnonzero(coord >= coord.max() - 1.0e-8 * length)
        disp = nodal(disp_key)[start:, face].mean(axis=1)
        disp_of[axis] = (disp, nodal(force_key)[start:, face].sum(axis=1))
        ref_length[axis] = length + disp[0]

    out = {}
    for axis, (disp, force) in disp_of.items():
        length = ref_length[axis]
        # Displacement and strain are measured from the reference step, so both
        # start at zero there whether or not the path consolidated first.
        moved = disp - disp[0]
        out[f"displacement_{axis}"] = moved
        out[f"force_{axis}"] = force

        stretch = 1.0 + moved / length
        out[f"strain_eng_{axis}"] = moved / length
        # Guard the log: a stretch at or below zero is a collapsed element,
        # which is a failed run rather than a curve worth reporting.
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"strain_log_{axis}"] = np.where(stretch > 0.0,
                                                 np.log(np.abs(stretch)), np.nan)

        area = 1.0
        for other in _AXES:
            if other != axis and other in ref_length:
                area *= ref_length[other]
        out[f"stress_eng_{axis}"] = force / area
    return out


class PreloadTrimmedReader:
    """``read_lcm_cap_exodus`` bound to a preload time, as a picklable object.

    MatCal ships the model (and with it the results reader) to worker
    processes, so the reader has to survive pickling. A closure or a lambda
    does not; a module-level class does.
    """

    def __init__(self, preload_time):
        self.preload_time = float(preload_time)

    def __call__(self, file_path, file_type=None):
        return read_lcm_cap_exodus(file_path, file_type,
                                   preload_time=self.preload_time)

    def __repr__(self):
        return f"PreloadTrimmedReader(preload_time={self.preload_time!r})"


def _add_deviatoric(data):
    """Add ``stress_dev_<axis>`` = ``sigma_aa - (sigma_bb + sigma_cc)/2`` for
    every axis whose three normal Cauchy components are present. On a triaxial
    path with equal lateral stresses this is ``sigma_1 - sigma_3``, the
    differential stress a laboratory reports, negative in compression."""
    normal = {a: f"stress_{a}{a}" for a in _AXES}
    if not all(n in data for n in normal.values()):
        return
    for axis in _AXES:
        others = [data[normal[a]] for a in _AXES if a != axis]
        data[f"stress_dev_{axis}"] = data[normal[axis]] - 0.5 * (others[0] + others[1])


def read_lcm_cap_exodus(file_path, file_type=None, preload_time=0.0):
    """Read ``file_path`` (an LCM cap-model Exodus output) and return a MatCal
    ``Data`` with ``time`` plus whichever of ``stress_*``, ``strain_*``,
    ``displacement_*``, ``force_*``, ``kappa`` and ``evp`` the run supports.
    ``file_type`` is accepted for MatCal's reader protocol and ignored (the
    format is always Exodus).

    ``preload_time`` is the continuation parameter at which loading proper
    begins on a path that consolidates first (``txc``). Steps before it are
    dropped and the strain measures are referred to that state; see PRELOAD in
    the module docstring. The default 0.0 keeps the whole run and refers the
    strains to the undeformed mesh, which is what every other path wants."""
    ds = _open_exo(file_path)
    try:
        t, var, idx = _exo_accessor(ds)
        start = 0
        if preload_time > 0.0:
            # Nearest written step rather than a threshold: the deck places the
            # end of consolidation on a step boundary, and a run that stopped
            # inside the consolidation stage is a failure worth naming.
            start = int(np.argmin(np.abs(t - preload_time)))
            if t[-1] < preload_time:
                raise RuntimeError(
                    f"{file_path}: the run ended at time {t[-1]:.6g}, before the "
                    f"end of the consolidation stage at {preload_time:.6g}; there "
                    f"is no loading curve to read.")
        data = {"time": t[start:]}
        for field, exo_name in _FIELD_MAP.items():
            base = exo_name[:-2] if exo_name.endswith("_1") else exo_name
            if exo_name in idx or base in idx:
                data[field] = var(exo_name)[start:]
        _add_deviatoric(data)
        data.update(_load_displacement(ds, start=start))
    finally:
        close = getattr(ds, "close", None)
        if callable(close):
            close()
    return convert_dictionary_to_data(data)
