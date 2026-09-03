#!/usr/bin/env python3
"""Convert digitized laboratory triaxial curves into harness CSVs.

Reads the format the Arctic team's digitized data arrives in (Engauge output
with a metadata row in curly brackets) and writes files the calibration harness
can consume directly on the ``txc`` load path:

    python prepare_data.py --out-dir data ~/test/calibration/Xu_Pc*_-4.csv

Each input becomes one output file with the harness's own column names,
``strain_eng_x`` and ``stress_dev_x``, so it needs no column mapping:

    python calibrate.py calibrate --load-path txc --curve dev-stress-strain \\
        --defaults permafrost --set confining_pressure=1.0e6 \\
        --data txc:data/Xu_Pc1e6_-4.csv --param A:1.5e6:2.5e6

The script prints the ``--set`` line each file needs, since the confining
pressure and the strain range are properties of the test rather than of the
material.

INPUT FORMAT. Row 1 is metadata, comma separated, each field ``{key: value}``
or ``{key: value [unit]}``; ``Pc`` (Pa) is the one the harness needs. Row 2
names the columns. The data columns share one uniform strain grid and each ends
where its own curve ended, so a row may carry a strain with no stress, or a
volumetric strain past the end of the strain column; both are dropped.

WHAT IT CHANGES, and why each is needed:

  * **Sign.** The harness works in compression negative. The digitized files
    have so far been compression positive, and the next batch is meant to be
    compression negative, so the convention is DETECTED from the axial strain
    column rather than assumed, and reported on every file.

    The volumetric column is never flipped. It is plotted positive upward as
    dilation in the source figures (Xu 2016 Figs. 3 and 7), and expansion
    positive already IS compression negative. Flipping it along with the other
    two would invert the dilatancy, which is the one thing this data is
    uniquely able to constrain (L, phi, Q).

  * **Zero offset.** Digitized curves start at a stress of up to 2.3e5 Pa at
    zero strain, an artifact of picking the curve off the axis. The value at
    zero strain is subtracted from the whole column.

  * **Truncation.** ``--max-strain`` drops everything past a given axial
    strain. Needed for the Yang curves, whose last few points are Engauge
    extrapolating past the plotted data (confirmed by Charles Choens): the
    3.0 MPa curve jumps from 7 to 28 MPa and the 0.5 MPa curve falls through
    zero to -1.2 MPa. Truncate those at 0.20.

The deviatoric stress ``q = sigma_1 - sigma_3`` is written as such, so nothing
here needs to know the confining pressure: it cancels out of the difference.
That is why ``--curve dev-stress-strain`` is the one to use with this data.

VOLUMETRIC COLUMN. Where the source carries a volumetric curve, it is written
as a third column, ``strain_vol``, linearly resampled onto the stress curve's
strain grid so that one objective can compare both against the same axial
strain (``--curve dev-stress-volumetric``). Resampling rather than assuming a
shared grid is deliberate: the two curves come from different figures and the
team has been asked to give the volumetric one its own strain column, so they
will not always align. No extrapolation, so a stress point outside the
volumetric curve's range is dropped and the count reported.
"""

import argparse
import csv
import os
import re
import sys

_META = re.compile(r"\{\s*([^:{}]+?)\s*:\s*(.*?)\s*\}")


def parse_metadata(row):
    """Return the ``{key: value}`` fields of a metadata row as a dict.

    Values keep any trailing ``[unit]``; callers that need a number strip it.
    """
    meta = {}
    for field in row:
        for key, value in _META.findall(field):
            meta[key.strip()] = value.strip()
    return meta


_LEADING_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def metadata_number(meta, key):
    """Return the number at the start of ``meta[key]``, or None.

    The unit is written three ways across the files received so far
    (``1e6 [Pa]``, ``1e6Pa``, and bare ``-6``), so the number is matched rather
    than the unit stripped.
    """
    if key not in meta:
        return None
    match = _LEADING_NUMBER.match(meta[key].strip())
    return float(match.group()) if match else None


def read_curves(path):
    """Return ``(metadata, [(strain, stress)], [(strain, volumetric)])``.

    Columns share a row index but not a length, so each pair is taken only
    where both cells of that row carry a number.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 3:
        raise SystemExit(f"{path}: expected a metadata row, a column row and data")
    meta = parse_metadata(rows[0])

    def pair(index):
        out = []
        for row in rows[2:]:
            if len(row) <= index:
                continue
            x, y = row[0].strip(), row[index].strip()
            if not x or not y:
                continue
            try:
                out.append((float(x), float(y)))
            except ValueError:
                continue
        return out

    return meta, pair(1), pair(2)


def resample_volumetric(strain_stress, strain_volumetric):
    """Return the volumetric strain sampled at the stress curve's strain grid.

    The two curves have so far arrived on one shared grid, but they need not:
    the volumetric curve is digitized from its own figure and the team has been
    asked to give it its own strain column. Interpolating onto the stress grid
    covers both cases and keeps one row per output line, which is what a single
    comparison file needs.

    Does NOT extrapolate. Returns ``(rows, n_dropped)`` where ``rows`` are the
    stress points that fall inside the volumetric curve's range, each paired
    with its interpolated volumetric strain.
    """
    if not strain_volumetric:
        return None, 0
    vx = [p[0] for p in strain_volumetric]
    vy = [p[1] for p in strain_volumetric]
    lo, hi = min(vx), max(vx)
    order = sorted(range(len(vx)), key=lambda i: vx[i])
    vx = [vx[i] for i in order]
    vy = [vy[i] for i in order]

    rows, dropped = [], 0
    for strain, stress in strain_stress:
        if strain < lo - 1.0e-12 or strain > hi + 1.0e-12:
            dropped += 1
            continue
        # Plain linear interpolation; the grids are dense enough that anything
        # cleverer would be inventing detail the digitizer did not capture.
        j = 0
        while j < len(vx) - 2 and vx[j + 1] < strain:
            j += 1
        x0, x1 = vx[j], vx[j + 1]
        y0, y1 = vy[j], vy[j + 1]
        t = 0.0 if x1 == x0 else (strain - x0) / (x1 - x0)
        rows.append((strain, stress, y0 + t * (y1 - y0)))
    return rows, dropped


def convert(strain_stress, max_strain=None):
    """Return ``(rows, report)``: the curve in the harness's convention.

    ``rows`` is a list of ``(strain_eng_x, stress_dev_x)``, compression
    negative and zero-offset removed. ``report`` describes what was done.
    """
    if not strain_stress:
        raise SystemExit("no stress-strain points found")

    # Detect the sign convention from the far end of the axial strain column,
    # which is the least ambiguous point on it.
    last_strain = strain_stress[-1][0]
    flip = last_strain > 0.0
    peak = max(strain_stress, key=lambda p: abs(p[1]))
    mixed = (peak[1] > 0.0) != (last_strain > 0.0)

    offset = strain_stress[0][1] if strain_stress[0][0] == 0.0 else 0.0
    sign = -1.0 if flip else 1.0

    rows = []
    for strain, stress in strain_stress:
        if max_strain is not None and abs(strain) > max_strain + 1.0e-12:
            continue
        rows.append((sign * strain, sign * (stress - offset)))

    report = {
        "convention": "compression positive (flipped)" if flip
                      else "compression negative (kept)",
        "offset": offset,
        "points_in": len(strain_stress),
        "points_out": len(rows),
        "max_strain": max(abs(r[0]) for r in rows),
        "peak_q": min(r[1] for r in rows),
        "mixed_signs": mixed,
    }
    return rows, report


def convert_volumetric(strain_volumetric, flip, max_strain=None):
    """Return ``[(strain_eng_x, strain_vol)]``.

    The axial strain follows the same convention as the stress-strain curve.
    The volumetric column does NOT: it is dilation positive in the source
    figures, which is already compression negative. See the module docstring.
    """
    sign = -1.0 if flip else 1.0
    rows = []
    for strain, volumetric in strain_volumetric:
        if max_strain is not None and abs(strain) > max_strain + 1.0e-12:
            continue
        rows.append((sign * strain, volumetric))
    return rows


def write_csv(path, header, rows, comments):
    with open(path, "w", newline="") as fh:
        for line in comments:
            fh.write(f"# {line}\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow([f"{v:.10e}" for v in row])


def suggest_axial_strain(max_strain):
    """A deck strain range that reaches past the data with a little headroom.

    MatCal interpolates the model onto the data's abscissa, so a model curve
    that stops short of the data leaves the far end unconstrained.
    """
    return -round(max_strain * 1.1 + 0.005, 2)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", metavar="CSV")
    ap.add_argument("--out-dir", required=True,
                    help="directory to write the converted curves into")
    ap.add_argument("--max-strain", type=float, default=None,
                    help="drop points past this axial strain (magnitude); use "
                         "0.20 for the Yang curves, whose tails are Engauge "
                         "extrapolation")
    ap.add_argument("--volumetric", action="store_true",
                    help="also write <name>-volumetric.csv, the volumetric "
                         "curve on its own unresampled grid. Not needed for "
                         "fitting (the main file carries a strain_vol column); "
                         "useful for plotting the measurement as digitized.")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    failed = False

    for path in args.inputs:
        name = os.path.splitext(os.path.basename(path))[0]
        meta, strain_stress, strain_volumetric = read_curves(path)
        rows, report = convert(strain_stress, args.max_strain)
        pressure = metadata_number(meta, "Pc")

        comments = [f"converted from {os.path.basename(path)} by prepare_data.py",
                    "base SI, compression negative; "
                    "stress_dev_x is q = sigma_1 - sigma_3"]
        # A machine-readable line the harness reads back: the deck constants
        # that belong to THIS test rather than to the material. calibrate.py
        # turns them into a MatCal state, which is what lets one run fit a
        # whole confining-pressure series with one parameter set.
        if pressure is not None:
            comments.append(
                f"harness-state: confining_pressure={pressure:.6e} "
                f"axial_strain={suggest_axial_strain(report['max_strain'])}")
        comments += [f"{k}: {v}" for k, v in meta.items()]
        comments.append(f"sign convention of the source: {report['convention']}")
        comments.append(f"stress offset removed at zero strain: {report['offset']:.6e} Pa")
        if args.max_strain is not None:
            comments.append(f"truncated at an axial strain of {args.max_strain}")

        # One file per test. The volumetric column is resampled onto the
        # stress curve's strain grid so a single objective can compare both
        # against the same axial strain; a curve that asks only for
        # stress_dev_x simply ignores the third column.
        flip = report["convention"].startswith("compression positive")
        volumetric = convert_volumetric(strain_volumetric, flip, args.max_strain)
        combined, dropped = resample_volumetric(rows, volumetric)
        if combined is not None:
            header = ["strain_eng_x", "stress_dev_x", "strain_vol"]
            body = combined
            comments.append(
                "strain_vol is the measured volumetric strain, linearly "
                "resampled onto this file's strain grid; positive is dilation")
            if dropped:
                comments.append(
                    f"{dropped} stress point(s) dropped: outside the "
                    f"volumetric curve's strain range (no extrapolation)")
        else:
            header = ["strain_eng_x", "stress_dev_x"]
            body = rows
        out = os.path.join(args.out_dir, f"{name}.csv")
        write_csv(out, header, body, comments)

        if report["mixed_signs"]:
            failed = True
            print(f"[{name}] ERROR: the stress and strain columns disagree on "
                  f"sign, so the convention cannot be read off the file. Look "
                  f"at it before using the output.", file=sys.stderr)
        if pressure is None:
            failed = True
            print(f"[{name}] ERROR: no numeric Pc in the metadata row; the "
                  f"confining pressure has to be passed by hand.", file=sys.stderr)

        print(f"[{name}] {report['points_in']} -> {report['points_out']} points, "
              f"{report['convention']}, offset {report['offset']:.3e} Pa, "
              f"peak q {report['peak_q']:.4e} Pa at |eps| <= {report['max_strain']:.3f}")
        if combined is not None:
            span = (min(r[2] for r in combined), max(r[2] for r in combined))
            print(f"    volumetric column: {len(combined)} rows, "
                  f"{span[0]:+.4f} to {span[1]:+.4f}"
                  + (f", {dropped} row(s) dropped outside its range" if dropped else ""))
        else:
            print(f"    no volumetric column: none in the source")
        if pressure is not None:
            print(f"    --data txc:{out} "
                  f"--set confining_pressure={pressure:.4e} "
                  f"--set axial_strain={suggest_axial_strain(report['max_strain'])}")

        if args.volumetric and volumetric:
            vout = os.path.join(args.out_dir, f"{name}-volumetric.csv")
            write_csv(vout, ["strain_eng_x", "strain_vol"], volumetric,
                      comments + ["the volumetric curve on ITS OWN grid, "
                                  "unresampled; kept for plotting, the "
                                  "harness reads the combined file above"])
            print(f"    also wrote the unresampled curve: {vout}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
