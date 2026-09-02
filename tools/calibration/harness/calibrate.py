#!/usr/bin/env python3
"""Calibration harness for the LCM cap-plasticity model (MatCal + Dakota).

Runs inside the ``matcal`` conda env (which wires Dakota + PYTHONPATH):

    conda activate matcal

Three actions:

    # 0. check the environment before anything else (no Albany run)
    python calibrate.py check

    # 1. generate reference "experiments" at the default parameters
    python calibrate.py make-reference --load-path confined --load-path hydrostatic

    # 2. calibrate R and W against both paths at once, starting off-target
    #    (INIT defaults to the value the references were generated at)
    python calibrate.py calibrate --load-path confined --load-path hydrostatic --param R:20:35:22 --param W:0.02:0.15:0.05 --study gradient

KINEMATICS: the finite-deformation kernel is the default; --small-strain
selects the infinitesimal-strain one. They differ by 5 to 7 per cent in peak
stress on these decks, and the finite-deformation kernel writes no strain
field, so the harness reconstructs strain from nodal displacement. The
SALEM_LIMESTONE defaults are small-strain values from Sun, Chen & Ostien
(2014): a sound starting point under finite deformation, not the answer.

LOAD PATHS: hydrostatic, confined and triaxial prescribe every strain
component and are the verification paths. ``txc`` is the laboratory test:
consolidate to a confining pressure, then shear axially with that pressure
held on traction-loaded lateral faces. Select the pressure with
--set confining_pressure=... and the strain range with --set axial_strain=...
(negative). It is the path that takes measured triaxial data.

CURVES: --curve selects what is compared. ``true-stress-strain`` (default) uses
logarithmic strain against Cauchy stress, the pair the finite-deformation
kernel works in; ``eng-stress-strain`` uses engineering strain u/L0 against
engineering stress force/A0, which is what most laboratory reports contain;
``load-displacement`` uses the loaded face's displacement and reaction force,
which mean the same thing under both kinematics; ``dev-stress-strain`` uses
engineering strain against the deviatoric Cauchy stress sigma_1 - sigma_3,
which is what a triaxial laboratory reports and the natural choice for txc;
``time-stress`` uses the LOCA continuation parameter. Experimental CSVs must carry the matching column
names, or be told which of their columns to use:

    python calibrate.py calibrate --load-path confined --curve eng-stress-strain --data confined:oedometer.csv:Strain:Stress_Pa --param R:20:35:22

Data files default to ``<out-dir>/<load_path>_reference.csv``.

UNITS: base SI throughout -- stress in Pa, force in N, displacement in m,
magnitudes in scientific notation, never prefixed units. That applies to
--param bounds and initial values for stress-like parameters (A, C, N, kappa0,
calpha, elastic_modulus in Pa; D, L, D1 in 1/Pa; D2 in 1/Pa^2), to --set
overrides, and to the data columns of any --data file. R, W, psi, theta, phi,
Q and poissons_ratio are dimensionless. Compression is negative.

Platform: auto-detected (rigel, sirius, cee); force with --platform or
$LCM_MATCAL_PLATFORM. See the project README.

NOTE: launch only ONE calibration (one Dakota study) per Python process -- a
documented MatCal/Dakota-library limitation. Separate runs = separate
`python` invocations.
"""

import argparse
import csv
import os
import shutil
import sys

import numpy as np
import matcal as mc

from site_matcal import (make_lcm_cap_model, SALEM_LIMESTONE, get_platform,
                         get_albany, get_load_path, CURVES, DEFAULT_CURVE,
                         LOAD_PATHS, DECK_CONSTANTS, TEMPLATES_DIR,
                         DEFAULT_FINITE_DEFORMATION, DEFAULT_SETS, DEFAULT_SET)

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "examples")

# Below this a peak |stress| or |force| is physically implausible for the
# materials this harness targets, and almost always means the column was
# supplied in MPa (or kN) rather than base SI.
_SMALL_STRESS = 1.0e4


def _kinematics(finite_deformation):
    return "finite deformation" if finite_deformation else "small strain"


def _parse_param(spec):
    """NAME:LO:HI[:INIT] -> matcal.Parameter (INIT defaults to SALEM value).

    Bounds and INIT are in the harness's base-SI units (stress in Pa), matching
    ``SALEM_LIMESTONE``; e.g. ``A:5e8:8e8`` rather than ``A:500:800``.
    """
    parts = spec.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            f"bad --param {spec!r}; expected NAME:LO:HI[:INIT]")
    name = parts[0]
    if name not in SALEM_LIMESTONE:
        raise argparse.ArgumentTypeError(
            f"--param {name}: not a cap placeholder; known: "
            f"{' '.join(sorted(SALEM_LIMESTONE))}")
    try:
        lo, hi = float(parts[1]), float(parts[2])
        init_given = len(parts) == 4
        # Without an explicit INIT the starting value comes from the selected
        # --defaults set, which argparse has not read yet: park the parameter at
        # the midpoint and let _reinit replace and range-check it.
        init = float(parts[3]) if init_given else 0.5 * (lo + hi)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad --param {spec!r}: LO, HI and INIT must be numbers")
    if not lo < hi:
        raise argparse.ArgumentTypeError(
            f"--param {name}: LO ({lo:g}) must be below HI ({hi:g})")
    if init_given and not lo <= init <= hi:
        raise argparse.ArgumentTypeError(
            f"--param {name}: INIT ({init:g}) is outside [{lo:g}, {hi:g}]")
    param = mc.Parameter(name, lo, hi, init)
    # Remember whether the user pinned INIT, so that --defaults can supply it
    # from the selected set when they did not. Checked in _reinit.
    param._lcm_init_given = init_given
    return param


def _reinit(param, base):
    """Return ``param`` with its starting value taken from ``base`` when the
    user gave no INIT. ``_parse_param`` cannot do this itself: argparse types
    run before --defaults has been read."""
    if getattr(param, "_lcm_init_given", True):
        return param
    name = param.get_name()
    lo, hi = param.get_lower_bound(), param.get_upper_bound()
    init = base[name]
    if not lo <= init <= hi:
        which = "/".join(k for k, v in DEFAULT_SETS.items() if v is base)
        raise SystemExit(
            f"--param {name}: the {which} starting value {init:g} is outside "
            f"the bounds [{lo:g}, {hi:g}]; either widen them, pick another "
            f"--defaults set, or give an explicit INIT as "
            f"{name}:{lo:g}:{hi:g}:<init>")
    return mc.Parameter(name, lo, hi, init)


def _parse_kv(spec):
    """NAME=VALUE -> (name, float). ``NAME`` is either a cap parameter or one
    of the deck constants that describe the test itself (the txc confining
    pressure and the like)."""
    key, sep, val = spec.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"bad --set {spec!r}; expected NAME=VALUE")
    if key not in SALEM_LIMESTONE and key not in DECK_CONSTANTS:
        raise argparse.ArgumentTypeError(
            f"--set {key}: not a cap placeholder or deck constant; cap "
            f"placeholders: {' '.join(sorted(SALEM_LIMESTONE))}; deck "
            f"constants: {' '.join(sorted(DECK_CONSTANTS))}")
    try:
        return key, float(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--set {key}: {val!r} is not a number")


def _parse_data(spec):
    """LOADPATH:CSV[:XCOL:YCOL] -> (load_path, csv_path, xcol, ycol).

    ``XCOL``/``YCOL`` name the columns of the file that hold the independent
    and dependent quantities of the selected --curve. Omit them when the file
    already uses the harness's own field names.
    """
    parts = spec.split(":")
    if len(parts) not in (2, 4):
        raise argparse.ArgumentTypeError(
            f"bad --data {spec!r}; expected LOADPATH:CSV[:XCOL:YCOL]")
    lp_name, path = parts[0], parts[1]
    if lp_name not in LOAD_PATHS:
        raise argparse.ArgumentTypeError(
            f"--data {lp_name}: unknown load path; known: {' '.join(sorted(LOAD_PATHS))}")
    xcol, ycol = (parts[2], parts[3]) if len(parts) == 4 else (None, None)
    return lp_name, path, xcol, ycol


_STATE_LINE = "harness-state:"


def _read_state_constants(path, known):
    """Return the per-curve deck constants recorded in a CSV by prepare_data.py.

    ``prepare_data.py`` writes a ``# harness-state: name=value ...`` comment
    holding the constants that belong to that TEST rather than to the material,
    which is what lets several curves at different confining pressures be fitted
    together. Returns ``{}`` for a file that carries no such line, which is how
    a hand-made CSV keeps working.
    """
    constants = {}
    with open(path) as fh:
        for line in fh:
            if not line.lstrip().startswith("#"):
                break
            text = line.lstrip("# \t").rstrip()
            if not text.startswith(_STATE_LINE):
                continue
            for field in text[len(_STATE_LINE):].split():
                name, sep, value = field.partition("=")
                if not sep:
                    raise SystemExit(f"{path}: bad harness-state field {field!r}; "
                                     f"expected name=value")
                if name not in known:
                    raise SystemExit(
                        f"{path}: harness-state names {name!r}, which is not a "
                        f"deck constant; known: {' '.join(sorted(known))}")
                try:
                    constants[name] = float(value)
                except ValueError:
                    raise SystemExit(f"{path}: harness-state {name}={value!r} "
                                     f"is not a number")
    return constants


def _state_name(path):
    """A MatCal state name from a data file name: it becomes a directory."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in stem)


def _reference_csv(out_dir, load_path):
    return os.path.join(out_dir, f"{load_path}_reference.csv")


def _read_csv(path):
    """Return ``(header, values)`` for a CSV with one header row of names.

    Blank lines and ``#`` comment lines are skipped anywhere in the file.
    """
    header, rows = None, []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or not "".join(row).strip():
                continue
            if row[0].lstrip().startswith("#"):
                continue
            if header is None:
                header = [c.strip() for c in row]
                continue
            rows.append((reader.line_num, row))
    if header is None:
        raise SystemExit(f"{path}: no header row found")
    if not rows:
        raise SystemExit(f"{path}: header {header} but no data rows")
    for lineno, row in rows:
        if len(row) != len(header):
            raise SystemExit(f"{path}: line {lineno} has {len(row)} values but "
                             f"the header names {len(header)} columns")
        for cell in row:
            try:
                float(cell)
            except ValueError:
                raise SystemExit(f"{path}: line {lineno}: {cell!r} is not a number")
    values = np.array([[float(c) for c in row] for _, row in rows], dtype=float)
    return header, values


def _load_experiment(lp_name, path, indep, dep, xcol, ycol):
    """Read an experimental CSV into a MatCal ``Data`` with fields named
    ``indep``/``dep``, mapping from ``xcol``/``ycol`` when given, and check
    that the numbers look like base SI."""
    header, values = _read_csv(path)
    xcol = xcol or indep
    ycol = ycol or dep
    missing = [c for c in (xcol, ycol) if c not in header]
    if missing:
        raise SystemExit(
            f"[{lp_name}] {path}: column(s) {', '.join(missing)} not found.\n"
            f"  columns in the file: {', '.join(header)}\n"
            f"  columns needed for --curve: {indep}, {dep}\n"
            f"  Either rename the file's columns, or say which to use:\n"
            f"    --data {lp_name}:{path}:<x column>:<y column>\n"
            f"  If this is a generated reference in other coordinates, rerun\n"
            f"  make-reference with the same --curve.")
    x = values[:, header.index(xcol)]
    y = values[:, header.index(ycol)]

    peak_y = np.abs(y).max()
    if dep.startswith(("stress", "force")) and 0.0 < peak_y < _SMALL_STRESS:
        print(f"[{lp_name}] WARNING: peak |{ycol}| is {peak_y:.3e}, which is very "
              f"small for a base-SI {dep.split('_')[0]}. Is this column in MPa or "
              f"kN? The harness is in Pa and N; convert the data, not the harness.",
              file=sys.stderr)
    peak_x = np.abs(x).max()
    if indep.startswith("strain") and peak_x > 1.0:
        print(f"[{lp_name}] WARNING: peak |{xcol}| is {peak_x:.3e}; strain is "
              f"dimensionless here, so a value above 1 usually means the column "
              f"is in percent.", file=sys.stderr)
    if np.all(y >= 0.0) and dep.startswith(("stress", "force")):
        print(f"[{lp_name}] WARNING: {ycol} is never negative. These decks load "
              f"in compression, which is negative in this sign convention; a "
              f"compression-positive curve will not be matched.", file=sys.stderr)

    return mc.convert_dictionary_to_data({indep: x, dep: y})


def check(platform=None):
    """Report on everything the harness needs, without running Albany."""
    results = []

    def record(ok, label, detail):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")

    print("environment check")
    py = sys.version_info
    record(py[:2] == (3, 12), "python",
           f"{py.major}.{py.minor}.{py.micro} (Dakota's bindings are cpython-312)")
    record(True, "matcal", f"{getattr(mc, '__version__', 'unknown')}")

    try:
        import dakota.environment  # noqa: F401
        record(True, "dakota bindings", "import dakota.environment OK")
    except Exception as exc:
        record(False, "dakota bindings", f"{exc}; is DAKOTA_ROOT on PYTHONPATH?")

    dakota_cli = shutil.which("dakota")
    record(bool(dakota_cli), "dakota CLI", dakota_cli or "not on PATH")

    # The documented footgun: Dakota on LD_LIBRARY_PATH makes the Albany
    # subprocess load Dakota's bundled libmpi and segfault at MPI finalize.
    ldlp = os.environ.get("LD_LIBRARY_PATH", "")
    bad = [p for p in ldlp.split(":") if p and "dakota" in p.lower()]
    record(not bad, "LD_LIBRARY_PATH clean of Dakota",
           f"offending entries: {', '.join(bad)}" if bad else "yes")

    plat = get_platform(platform)
    record(True, "platform", repr(plat))

    albany = get_albany(platform)
    record(os.path.isfile(albany) and os.access(albany, os.X_OK), "Albany",
           albany if os.path.isfile(albany) else f"{albany} (not found; build it, "
           "or set $LCM_ALBANY)")

    missing = []
    for lp in LOAD_PATHS.values():
        for fname in (lp.deck, lp.materials):
            if not os.path.isfile(os.path.join(TEMPLATES_DIR, fname)):
                missing.append(fname)
    record(not missing, "templates",
           TEMPLATES_DIR if not missing else f"missing {sorted(set(missing))}")

    ok = all(results)
    print(f"\n{'all checks passed' if ok else 'SOME CHECKS FAILED'} "
          f"({sum(results)}/{len(results)})")
    return 0 if ok else 1


def make_reference(load_paths, defaults, out_dir, platform, curve, finite_deformation):
    os.makedirs(out_dir, exist_ok=True)
    for lp_name in load_paths:
        lp = get_load_path(lp_name)
        indep, dep = lp.fields(curve)
        model = make_lcm_cap_model(load_path=lp_name, defaults=defaults,
                                   platform=platform, name=f"ref_{lp_name}",
                                   finite_deformation=finite_deformation)
        run_dir = os.path.join(out_dir, f"reference_run_{lp_name}")
        os.makedirs(run_dir, exist_ok=True)
        results = model.run(mc.State(lp_name), mc.ParameterCollection("truth"),
                            target_directory=run_dir)
        data = results.results_data
        x = np.asarray(data[indep])
        y = np.asarray(data[dep])
        out = _reference_csv(out_dir, lp_name)
        np.savetxt(out, np.column_stack([x, y]), delimiter=",",
                   header=f"{indep},{dep}", comments="")
        print(f"[{lp_name}] wrote reference {out} ({len(x)} points, "
              f"{curve}, {_kinematics(finite_deformation)}, "
              f"peak |{dep}| = {np.abs(y).max():.6e})")


def calibrate(load_paths, params, data_map, defaults, out_dir, platform,
              study_type, core_limit, curve, finite_deformation):
    if not params:
        raise SystemExit("no --param given; nothing to calibrate")

    study_cls = {"gradient": mc.GradientCalibrationStudy,
                 "scipy": mc.ScipyMinimizeStudy}[study_type]
    study = study_cls(*params)

    for lp_name in load_paths:
        lp = get_load_path(lp_name)
        indep, dep = lp.fields(curve)
        entries = data_map.get(lp_name) or [(None, None, None)]
        model = make_lcm_cap_model(load_path=lp_name, defaults=defaults,
                                   platform=platform,
                                   finite_deformation=finite_deformation)
        datasets = []
        stateful = False
        for path, xcol, ycol in entries:
            data_path = os.path.abspath(path or _reference_csv(out_dir, lp_name))
            if not os.path.isfile(data_path):
                raise SystemExit(f"[{lp_name}] no data at {data_path}; run "
                                 f"make-reference or pass --data {lp_name}:<csv>")
            experiment = _load_experiment(lp_name, data_path, indep, dep,
                                          xcol, ycol)
            # Constants that belong to this curve rather than to the material:
            # the confining pressure and the strain range of that test. As
            # state constants they beat the deck defaults and the --set
            # overrides stay available for a single-curve run.
            per_curve = _read_state_constants(data_path, lp.constants)
            if per_curve or len(entries) > 1:
                stateful = True
                state = mc.State(_state_name(data_path), **per_curve)
                experiment.set_state(state)
                if per_curve:
                    model.add_state_constants(state, **per_curve)
                detail = " ".join(f"{k}={v:g}" for k, v in per_curve.items())
                print(f"[{lp_name}] state {state.name}: {data_path}"
                      + (f" ({detail})" if detail else ""))
            else:
                print(f"[{lp_name}] evaluation set: {data_path} "
                      f"({indep} vs {dep})")
            datasets.append(experiment)

        objective = mc.CurveBasedInterpolatedObjective(indep, dep)
        if stateful:
            study.add_evaluation_set(
                model, objective, mc.DataCollection(lp_name, *datasets))
        else:
            study.add_evaluation_set(model, objective, datasets[0])
        if len(datasets) > 1:
            print(f"[{lp_name}] {len(datasets)} curves fitted together "
                  f"({indep} vs {dep})")

    study.set_core_limit(core_limit)
    print(f"platform={get_platform(platform).name} study={study_type} "
          f"curve={curve} kinematics={_kinematics(finite_deformation)} "
          f"params={[p.get_name() for p in params]}")
    # MatCal writes its working dirs / Dakota files into the CWD; run inside a
    # dedicated (git-ignored) directory so the source tree stays clean.
    run_dir = os.path.join(out_dir, "calibration_run")
    os.makedirs(run_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        results = study.launch()
    finally:
        os.chdir(cwd)
    print("BEST:", results.best)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["check", "make-reference", "calibrate"])
    ap.add_argument("--load-path", action="append", dest="load_paths",
                    metavar="NAME",
                    help="hydrostatic|confined|triaxial|txc (repeatable)")
    ap.add_argument("--curve", choices=sorted(CURVES), default=DEFAULT_CURVE,
                    help=f"fields to compare (default: {DEFAULT_CURVE})")
    ap.add_argument("--param", action="append", type=_parse_param, dest="params",
                    default=[], metavar="NAME:LO:HI[:INIT]",
                    help="calibrated parameter, base SI (repeatable)")
    ap.add_argument("--data", action="append", type=_parse_data, dest="data",
                    default=[], metavar="LOADPATH:CSV[:XCOL:YCOL]",
                    help="experimental data for a load path, base SI. "
                         "Repeatable, INCLUDING several times for one load "
                         "path: each curve then becomes a MatCal state and all "
                         "of them are fitted with one parameter set.")
    ap.add_argument("--set", action="append", type=_parse_kv, dest="sets",
                    default=[], metavar="NAME=VALUE",
                    help="override a cap parameter or deck constant "
                         f"({'|'.join(sorted(DECK_CONSTANTS))}), base SI "
                         "(repeatable)")
    kin = ap.add_mutually_exclusive_group()
    kin.add_argument("--finite-deformation", dest="finite_deformation",
                     action="store_true", default=DEFAULT_FINITE_DEFORMATION,
                     help="exponential/logarithmic-map kinematics"
                          + (" (default)" if DEFAULT_FINITE_DEFORMATION else ""))
    kin.add_argument("--small-strain", dest="finite_deformation",
                     action="store_false",
                     help="infinitesimal-strain kinematics"
                          + ("" if DEFAULT_FINITE_DEFORMATION else " (default)"))
    ap.add_argument("--defaults", choices=sorted(DEFAULT_SETS), default=DEFAULT_SET,
                    help="starting parameter set that --param INIT and every "
                         "un-fitted placeholder come from "
                         f"(default: {DEFAULT_SET}). Use permafrost for txc "
                         "and for any frozen-soil calibration; salem is a rock "
                         "whose cap dominates every triaxial test of it.")
    ap.add_argument("--study", choices=["gradient", "scipy"], default="gradient")
    ap.add_argument("--platform", default=None, help="rigel|sirius|cee (default: auto)")
    ap.add_argument("--core-limit", type=int, default=4)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)

    if args.action == "check":
        return check(args.platform)

    load_paths = args.load_paths or ["confined"]
    for lp_name in load_paths:
        get_load_path(lp_name)          # fail early, with the list of known names
    # The named set first, then the individual --set overrides on top of it.
    base = DEFAULT_SETS[args.defaults]
    defaults = {**base, **dict(args.sets)}
    # --param INIT defaults to the named set's value, not always Salem's.
    params = [_reinit(p, base) for p in args.params]
    # Repeatable per load path: one --data per experimental curve. Several on
    # the same path become several MatCal states of one model.
    data_map = {}
    for lp_name, path, xcol, ycol in args.data:
        data_map.setdefault(lp_name, []).append((path, xcol, ycol))

    if args.action == "make-reference":
        make_reference(load_paths, defaults, args.out_dir, args.platform,
                       args.curve, args.finite_deformation)
    else:
        calibrate(load_paths, params, data_map, defaults, args.out_dir,
                  args.platform, args.study, args.core_limit, args.curve,
                  args.finite_deformation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
