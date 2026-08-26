#!/usr/bin/env python3
"""Calibration harness for the LCM cap-plasticity model (MatCal + Dakota).

Runs inside the ``matcal`` conda env (which wires Dakota + PYTHONPATH):

    conda activate matcal

Generate reference "experiments" at the default parameters, then calibrate a
parameter subset back to them (multi-load-path capable):

    # references for two paths
    python calibrate.py make-reference --load-path confined --load-path hydrostatic

    # calibrate R and W against both paths at once
    python calibrate.py calibrate --load-path confined --load-path hydrostatic --param R:20:35 --param W:0.02:0.15 --study gradient

Data files default to ``<out-dir>/<load_path>_reference.csv`` (columns
time,<dependent field>). Point ``--data confined:/path/to/expt.csv`` at real
measurements to calibrate against lab data.

UNITS: base SI throughout -- stress in Pa, magnitudes in scientific notation,
never prefixed units. That applies to --param bounds and initial values for
stress-like parameters (A, C, N, kappa0, calpha, elastic_modulus in Pa; D, L,
D1 in 1/Pa; D2 in 1/Pa^2), to --set overrides, and to the stress column of any
--data file. R, W, psi, theta, phi, Q and poissons_ratio are dimensionless.

Platform: auto-detected (rigel, sirius, cee); force with --platform or
$LCM_MATCAL_PLATFORM. See the project README.

NOTE: launch only ONE calibration (one Dakota study) per Python process -- a
documented MatCal/Dakota-library limitation. Separate runs = separate
`python` invocations.
"""

import argparse
import os
import sys

import numpy as np
import matcal as mc

from site_matcal import (make_lcm_cap_model, read_lcm_cap_exodus,
                         SALEM_LIMESTONE, get_platform, get_load_path)

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "examples")


def _parse_param(spec):
    """NAME:LO:HI[:INIT] -> matcal.Parameter (INIT defaults to SALEM value).

    Bounds and INIT are in the harness's base-SI units (stress in Pa), matching
    ``SALEM_LIMESTONE``; e.g. ``A:5e8:8e8`` rather than ``A:500:800``.
    """
    parts = spec.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            f"bad --param {spec!r}; expected NAME:LO:HI[:INIT]")
    name, lo, hi = parts[0], float(parts[1]), float(parts[2])
    init = float(parts[3]) if len(parts) == 4 else SALEM_LIMESTONE.get(name)
    if init is None:
        raise argparse.ArgumentTypeError(
            f"--param {name}: no default known; give an explicit INIT")
    return mc.Parameter(name, lo, hi, init)


def _parse_kv(spec):
    key, _, val = spec.partition("=")
    return key, float(val)


def _reference_csv(out_dir, load_path):
    return os.path.join(out_dir, f"{load_path}_reference.csv")


def make_reference(load_paths, defaults, out_dir, platform):
    os.makedirs(out_dir, exist_ok=True)
    for lp_name in load_paths:
        lp = get_load_path(lp_name)
        model = make_lcm_cap_model(load_path=lp_name, defaults=defaults,
                                   platform=platform, name=f"ref_{lp_name}")
        run_dir = os.path.join(out_dir, f"reference_run_{lp_name}")
        os.makedirs(run_dir, exist_ok=True)
        results = model.run(mc.State(lp_name), mc.ParameterCollection("truth"),
                            target_directory=run_dir)
        data = results.results_data
        t = np.asarray(data[lp.independent])
        y = np.asarray(data[lp.dependent])
        out = _reference_csv(out_dir, lp_name)
        np.savetxt(out, np.column_stack([t, y]), delimiter=",",
                   header=f"{lp.independent},{lp.dependent}", comments="")
        print(f"[{lp_name}] wrote reference {out} ({len(t)} points)")


def calibrate(load_paths, params, data_map, defaults, out_dir, platform,
              study_type, core_limit):
    if not params:
        raise SystemExit("no --param given; nothing to calibrate")

    study_cls = {"gradient": mc.GradientCalibrationStudy,
                 "scipy": mc.ScipyMinimizeStudy}[study_type]
    study = study_cls(*params)

    for lp_name in load_paths:
        lp = get_load_path(lp_name)
        data_path = os.path.abspath(data_map.get(lp_name, _reference_csv(out_dir, lp_name)))
        if not os.path.isfile(data_path):
            raise SystemExit(f"[{lp_name}] no data at {data_path}; run "
                             f"make-reference or pass --data {lp_name}:<csv>")
        experiment = mc.FileData(data_path)
        model = make_lcm_cap_model(load_path=lp_name, defaults=defaults,
                                   platform=platform)
        objective = mc.CurveBasedInterpolatedObjective(lp.independent, lp.dependent)
        study.add_evaluation_set(model, objective, experiment)
        print(f"[{lp_name}] evaluation set: {data_path}")

    study.set_core_limit(core_limit)
    print(f"platform={get_platform(platform).name} study={study_type} "
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
    ap.add_argument("action", choices=["make-reference", "calibrate"])
    ap.add_argument("--load-path", action="append", dest="load_paths",
                    metavar="NAME", help="hydrostatic|confined|triaxial (repeatable)")
    ap.add_argument("--param", action="append", type=_parse_param, dest="params",
                    default=[], metavar="NAME:LO:HI[:INIT]",
                    help="calibrated parameter, base SI (repeatable)")
    ap.add_argument("--data", action="append", dest="data", default=[],
                    metavar="LOADPATH:CSV",
                    help="experimental data for a load path, stress in Pa (repeatable)")
    ap.add_argument("--set", action="append", type=_parse_kv, dest="defaults",
                    default=[], metavar="NAME=VALUE",
                    help="override a constant default, base SI (repeatable)")
    ap.add_argument("--study", choices=["gradient", "scipy"], default="gradient")
    ap.add_argument("--platform", default=None, help="rigel|sirius|cee (default: auto)")
    ap.add_argument("--core-limit", type=int, default=4)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)

    load_paths = args.load_paths or ["confined"]
    defaults = dict(args.defaults)
    data_map = {}
    for d in args.data:
        lp_name, _, csv = d.partition(":")
        data_map[lp_name] = csv

    if args.action == "make-reference":
        make_reference(load_paths, defaults, args.out_dir, args.platform)
    else:
        calibrate(load_paths, args.params, data_map, defaults, args.out_dir,
                  args.platform, args.study, args.core_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
