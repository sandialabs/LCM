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

SIGN CONVENTION FIELD. A metadata field ``SignConven`` (``CompressPos`` or
``CompressNeg``) states the convention of EVERY column, the volumetric one
included, and when present it replaces both the detection above and the rule
that the volumetric column is never flipped. The Moo Lee files carry it, and
there the volumetric strain is compaction positive: in the hydrostatic test
axial plus twice radial strain equals the volumetric strain to 8e-7, all
positive under compression. Applying the Xu rule to them would invert the
dilatancy.

MEASURED RECORDS. Laboratory records, as opposed to digitized curves, carry
thousands of noisy points whose axial strain goes back and forth. They are
recognized by that reversal and treated as follows, each step reported:

  * **Stage before shear.** Only rows at or past the zero of axial strain are
    kept. In the Lee triaxial files the rows before it are not consolidation
    (they disagree with the hydrostatic test at the same pressure by up to a
    factor of 40, and show no radial strain), so they are dropped.
  * **Noise.** A centered moving average over ``--window`` rows (7), then points
    are kept only where the axial strain has advanced and either the strain or
    the stress has changed by a set fraction of its range. The second criterion
    keeps the steep initial rise, which uniform strain bins would flatten.
  * **Seating.** ``--toe-correct`` moves the strain origin to where the
    steepest part of the initial rise, extrapolated linearly, reaches zero
    stress, as in ASTM toe compensation. A positive shift removes a seating
    toe; a negative one restores a rise that began before the recorded zero.
    The line is a least-squares fit to the unsmoothed rows between 5 and 30
    percent of the peak stress. The volumetric column is re-zeroed at the new
    origin.

HYDROSTATIC RECORDS. A file whose ``TestType`` is ``HS`` holds a pressure
history, with unloading and reloading, not a single curve. It is written for
the ``hydrostatic`` load path as ``time, stress_xx, strain_vol``: knots of the
history on a uniform time grid, chosen to include every reversal of the
pressure, with ``stress_xx = -P`` and ``strain_vol = J - 1`` the volume change
ratio, compression negative, where ``J = (1 - eps_a)(1 - eps_r)^2`` from the
compression-positive axial and radial strains. ``calibrate.py`` prescribes the
volume history on the element and compares the pressure at the same times
(``--curve time-stress``), which is well posed where a pressure against
volumetric strain comparison is not: that curve doubles back on itself.
``--max-pressure`` truncates the record; the Lee report attributes the change
in the hydrostatic test near 1e8 Pa to pressure melting of the ice, which the
model does not represent.
"""

import argparse
import csv
import os
import re
import sys

import numpy as np

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


def sign_convention(meta):
    """Return ``"pos"``, ``"neg"`` or None from the ``SignConven`` field."""
    value = meta.get("SignConven", "").replace(" ", "").lower()
    if not value:
        return None
    if value.startswith("compresspos"):
        return "pos"
    if value.startswith("compressneg"):
        return "neg"
    raise SystemExit(f"unrecognized SignConven {meta['SignConven']!r}; "
                     f"expected CompressPos or CompressNeg")


def test_type(meta):
    return meta.get("TestType", "").strip().split(",")[0].strip().upper()


def read_table(path):
    """Return ``(metadata, column names, array)`` with NaN for empty cells.

    Rows whose first cell is empty (the trailing blank rows of the measured
    files) are dropped.
    """
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(csv.reader(fh))
    # Some files wrap each whole line in quotes, which the reader returns as a
    # single cell; split those again.
    rows = [next(csv.reader([r[0]])) if len(r) == 1 and "," in r[0] else r
            for r in rows]
    meta = parse_metadata(rows[0])
    names = [c.strip() for c in rows[1]]
    data = []
    for row in rows[2:]:
        if not row or not row[0].strip():
            continue
        vals = []
        for cell in row[:len(names)]:
            cell = cell.strip()
            try:
                vals.append(float(cell) if cell else np.nan)
            except ValueError:
                vals.append(np.nan)
        vals += [np.nan] * (len(names) - len(vals))
        data.append(vals)
    return meta, names, np.array(data, dtype=float)


def reversals(x):
    """Number of sign changes in the nonzero increments of ``x``."""
    d = np.diff(x)
    d = d[d != 0.0]
    return int(np.sum(np.diff(np.sign(d)) != 0))


def is_measured(strains):
    """A measured record goes back and forth in strain; a digitized curve
    does not. More than 2 percent reversals separates the two cleanly."""
    return reversals(np.asarray(strains)) > 0.02 * len(strains)


def moving_average(x, window):
    if window <= 1:
        return np.asarray(x, dtype=float)
    kernel = np.ones(window) / window
    pad = window // 2
    xp = np.pad(np.asarray(x, dtype=float), (pad, window - 1 - pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def thin(e, q, de, dq):
    """Indices of points kept: strain strictly increasing, and either the
    strain or the stress changed by at least ``de`` or ``dq`` since the last
    kept point."""
    keep = [0]
    for i in range(1, len(e)):
        j = keep[-1]
        if e[i] <= e[j]:
            continue
        if e[i] - e[j] >= de or abs(q[i] - q[j]) >= dq:
            keep.append(i)
    return np.array(keep)


def toe_origin(e, q, lower=0.05, upper=0.3):
    """Strain at which the initial rise, extrapolated, reaches zero stress.

    The rise is taken as the points before the first to exceed ``upper`` of
    the peak stress and past the last below ``lower`` of it; a least-squares
    line through them gives the slope and the intercept. Pointwise slopes are
    not used: where the rise is steep, the thinned points are close in strain
    and their differences are noise.
    """
    qmax = np.max(q)
    end = int(np.argmax(q >= upper * qmax))
    begin = int(np.nonzero(q[:end] <= lower * qmax)[0][-1]) if np.any(q[:end] <= lower * qmax) else 0
    sel = slice(begin, max(end, begin + 2) + 1)
    slope, intercept = np.polyfit(e[sel], q[sel], 1)
    return -intercept / slope, slope


def convert_measured_triaxial(meta, names, table, max_strain, window, toe):
    """Measured triaxial record -> rows ``(strain_eng_x, stress_dev_x,
    strain_vol)`` in the harness convention, and a report. See MEASURED
    RECORDS in the module docstring."""
    conv = sign_convention(meta)
    if conv is None:
        raise SystemExit("a measured record needs a SignConven field: its "
                         "convention cannot be read reliably off noisy data")
    rows = table[~np.isnan(table[:, 0]) & ~np.isnan(table[:, 1])]
    e, q = rows[:, 0].copy(), rows[:, 1].copy()
    has_v = rows.shape[1] > 2 and not np.all(np.isnan(rows[:, 2]))
    v = rows[:, 2].copy() if has_v else np.zeros_like(e)
    # Compression positive from here on, so the shear stage is e >= 0.
    if conv == "neg":
        e, q, v = -e, -q, -v
    report = {"convention": "compression positive (flipped, stated in the file)"
              if conv == "pos" else "compression negative (stated in the file)",
              "points_in": len(e)}
    start = int(np.argmax(e >= 0.0))
    report["dropped_before_shear"] = start
    e, q, v = e[start:], q[start:], v[start:]
    if max_strain is not None:
        cut = e <= max_strain + 1.0e-12
        e, q, v = e[cut], q[cut], v[cut]
    # The toe is fitted on the unsmoothed rows: the initial rise can span only
    # a few rows, which the moving average would flatten.
    shift = 0.0
    if toe:
        shift, slope = toe_origin(e, q)
        report["toe_slope"] = slope
    es, qs, vs = (moving_average(x, window) for x in (e, q, v))
    idx = thin(es, qs, de=es.max() / 300.0, dq=np.abs(qs).max() / 100.0)
    e, q, v = es[idx], qs[idx], vs[idx]
    if toe:
        e = e - shift
        v0 = np.interp(0.0, e, v) if e[0] < 0.0 < e[-1] else v[0]
        keep = e > 0.0
        e, q, v = np.concatenate(([0.0], e[keep])), np.concatenate(([0.0], q[keep])), \
                  np.concatenate(([0.0], v[keep] - v0))
    else:
        v = v - v[0]
    report["toe_shift"] = shift
    # To the harness convention: compression negative, dilation positive.
    out = np.column_stack((-e, -q, -v))
    report.update(points_out=len(out), max_strain=float(np.abs(out[:, 0]).max()),
                  peak_q=float(out[:, 1].min()), offset=0.0, mixed_signs=False,
                  has_volumetric=has_v)
    return out, report


def zigzag_extrema(p, delta):
    """Indices of the reversals of ``p`` that exceed ``delta``."""
    ext = []
    i_ext, direction = 0, 0
    for i in range(1, len(p)):
        if direction >= 0:
            if p[i] > p[i_ext]:
                i_ext = i
            elif p[i_ext] - p[i] > delta:
                if direction == 1:
                    ext.append(i_ext)
                direction, i_ext = -1, i
                continue
            if direction == 0 and p[i] - p[0] > delta:
                direction = 1
        if direction < 0:
            if p[i] < p[i_ext]:
                i_ext = i
            elif p[i] - p[i_ext] > delta:
                ext.append(i_ext)
                direction, i_ext = 1, i
    return ext


def convert_hydrostatic(meta, names, table, max_pressure, knots):
    """Hydrostatic record -> ``(time, stress_xx, strain_vol)`` knots and a
    report. See HYDROSTATIC RECORDS in the module docstring."""
    conv = sign_convention(meta)
    if conv is None:
        raise SystemExit("a hydrostatic record needs a SignConven field")
    lower = [n.lower() for n in names]
    col = lambda *keys: next((i for i, n in enumerate(lower) if n in keys), None)
    ia, ip = col("engineering_strain", "axial_strain"), col("pc", "pressure")
    iv, ir = col("volume_strain", "volumetric_strain"), col("radial_strain", "lateral_strain")
    if ip is None or (ir is None and iv is None):
        raise SystemExit(f"hydrostatic record needs a pressure column and radial "
                         f"or volumetric strain; columns are {names}")
    rows = table[~np.isnan(table[:, ip])]
    s = 1.0 if conv == "pos" else -1.0
    p = s * rows[:, ip]
    if ir is not None and ia is not None:
        ea, er = s * rows[:, ia], s * rows[:, ir]
        jac = (1.0 - ea) * (1.0 - er) ** 2
        how = "J = (1 - eps_a)(1 - eps_r)^2 from the axial and radial columns"
    else:
        jac = 1.0 - s * rows[:, iv]
        how = "J = 1 - eps_v from the volumetric column"
    report = {"points_in": len(p), "how": how}
    if max_pressure is not None:
        over = np.nonzero(p > max_pressure)[0]
        if len(over):
            p, jac = p[:over[0]], jac[:over[0]]
    report["truncated_at"] = float(p.max())
    ps, js = moving_average(p, 5), moving_average(jac, 5)
    ext = zigzag_extrema(ps, 0.02 * ps.max())
    # Knots uniform in normalized arc length along (J, P), plus every reversal.
    dj, dp = np.diff(js) / np.ptp(js), np.diff(ps) / np.ptp(ps)
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(dj, dp))))
    uniform = np.searchsorted(arc, np.linspace(0.0, arc[-1], knots))
    idx = np.unique(np.concatenate(([0, len(ps) - 1], uniform.clip(0, len(ps) - 1), ext)))
    t = np.linspace(0.0, 1.0, len(idx))
    stress = -ps[idx]
    vol = js[idx] - 1.0
    vol[0] = 0.0
    report.update(knots=len(idx), reversals=len(ext),
                  reversal_pressures=[float(ps[i]) for i in ext],
                  final_strain_vol=float(vol[-1]))
    return np.column_stack((t, stress, vol)), report


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


def convert(strain_stress, max_strain=None, stated=None):
    """Return ``(rows, report)``: the curve in the harness's convention.

    ``rows`` is a list of ``(strain_eng_x, stress_dev_x)``, compression
    negative and zero-offset removed. ``report`` describes what was done.
    """
    if not strain_stress:
        raise SystemExit("no stress-strain points found")

    # Detect the sign convention from the far end of the axial strain column,
    # which is the least ambiguous point on it.
    last_strain = strain_stress[-1][0]
    flip = last_strain > 0.0 if stated is None else stated == "pos"
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
        "convention": ("compression positive (flipped)" if flip
                       else "compression negative (kept)")
                      + (", stated in the file" if stated else ""),
        "offset": offset,
        "points_in": len(strain_stress),
        "points_out": len(rows),
        "max_strain": max(abs(r[0]) for r in rows),
        "peak_q": min(r[1] for r in rows),
        "mixed_signs": mixed,
    }
    return rows, report


def convert_volumetric(strain_volumetric, flip, max_strain=None, stated=None):
    """Return ``[(strain_eng_x, strain_vol)]``.

    The axial strain follows the same convention as the stress-strain curve.
    The volumetric column does NOT: it is dilation positive in the source
    figures, which is already compression negative. See the module docstring.
    """
    sign = -1.0 if flip else 1.0
    # With no stated convention the volumetric column is kept as it is (the
    # Xu rule, see the module docstring); a stated compression-positive
    # convention covers it too.
    vsign = -1.0 if stated == "pos" else 1.0
    rows = []
    for strain, volumetric in strain_volumetric:
        if max_strain is not None and abs(strain) > max_strain + 1.0e-12:
            continue
        rows.append((sign * strain, vsign * volumetric))
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
    ap.add_argument("--window", type=int, default=7,
                    help="measured records: rows in the moving average "
                         "(default 7)")
    ap.add_argument("--toe-correct", action="store_true",
                    help="measured records: move the strain origin to where "
                         "the initial rise, extrapolated, reaches zero stress")
    ap.add_argument("--max-pressure", type=float, default=None,
                    help="hydrostatic records: drop everything from the first "
                         "point above this pressure, Pa")
    ap.add_argument("--knots", type=int, default=80,
                    help="hydrostatic records: knots uniform in arc length, "
                         "before adding the reversals (default 80)")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    failed = False

    for path in args.inputs:
        name = os.path.splitext(os.path.basename(path))[0]
        meta, names, table = read_table(path)
        if test_type(meta) == "HS":
            failed |= write_hydrostatic(path, name, meta, names, table, args)
            continue
        if is_measured(table[~np.isnan(table[:, 0]), 0]):
            failed |= write_measured_triaxial(path, name, meta, names, table, args)
            continue
        meta, strain_stress, strain_volumetric = read_curves(path)
        stated = sign_convention(meta)
        rows, report = convert(strain_stress, args.max_strain, stated)
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
        volumetric = convert_volumetric(strain_volumetric, flip, args.max_strain, stated)
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


def _common_comments(path, meta):
    return ([f"converted from {os.path.basename(path)} by prepare_data.py"],
            [f"{k}: {v}" for k, v in meta.items()])


def write_measured_triaxial(path, name, meta, names, table, args):
    out_rows, report = convert_measured_triaxial(
        meta, names, table, args.max_strain, args.window, args.toe_correct)
    pressure = metadata_number(meta, "Pc")
    head, tail = _common_comments(path, meta)
    comments = head + ["base SI, compression negative; "
                       "stress_dev_x is q = sigma_1 - sigma_3"]
    if pressure is not None:
        comments.append(f"harness-state: confining_pressure={pressure:.6e} "
                        f"axial_strain={suggest_axial_strain(report['max_strain'])}")
    comments += tail + [
        f"measured record: {report['convention']}",
        f"{report['dropped_before_shear']} rows before the zero of axial strain dropped",
        f"moving average over {args.window} rows, then thinned to "
        f"{report['points_out']} points",
        f"toe correction: strain origin moved by {report['toe_shift']:+.6e}"
        if args.toe_correct else "no toe correction",
    ]
    if args.max_strain is not None:
        comments.append(f"truncated at an axial strain of {args.max_strain}")
    if report["has_volumetric"]:
        header = ["strain_eng_x", "stress_dev_x", "strain_vol"]
        body = out_rows
        comments.append("strain_vol is the measured volumetric strain, zeroed at "
                        "the strain origin; positive is dilation")
    else:
        header, body = ["strain_eng_x", "stress_dev_x"], out_rows[:, :2]
    out = os.path.join(args.out_dir, f"{name}.csv")
    write_csv(out, header, body, comments)
    print(f"[{name}] measured record, {report['convention']}: "
          f"{report['points_in']} rows, {report['dropped_before_shear']} before "
          f"shear dropped, {report['points_out']} points written; peak q "
          f"{report['peak_q']:.4e} Pa at |eps| <= {report['max_strain']:.4f}")
    if args.toe_correct:
        print(f"    toe correction: origin moved by {report['toe_shift']:+.5f} "
              f"(steepest initial slope {report['toe_slope']:.3e} Pa)")
    if report["has_volumetric"]:
        print(f"    volumetric column: {out_rows[:, 2].min():+.4f} to "
              f"{out_rows[:, 2].max():+.4f}")
    if pressure is None:
        print(f"[{name}] ERROR: no numeric Pc in the metadata row", file=sys.stderr)
        return True
    print(f"    --data txc:{out} --set confining_pressure={pressure:.4e} "
          f"--set axial_strain={suggest_axial_strain(report['max_strain'])}")
    return False


def write_hydrostatic(path, name, meta, names, table, args):
    out_rows, report = convert_hydrostatic(meta, names, table,
                                           args.max_pressure, args.knots)
    head, tail = _common_comments(path, meta)
    comments = head + [
        "base SI, compression negative; a hydrostatic history for the "
        "hydrostatic load path (--curve time-stress)",
        "time is the continuation parameter at each knot; stress_xx = -P; "
        "strain_vol = J - 1, the volume change ratio",
    ] + tail + [
        report["how"],
        f"{report['points_in']} rows -> {report['knots']} knots including "
        f"{report['reversals']} pressure reversals",
        f"truncated at {args.max_pressure:.6e} Pa" if args.max_pressure
        else "not truncated",
    ]
    out = os.path.join(args.out_dir, f"{name}.csv")
    write_csv(out, ["time", "stress_xx", "strain_vol"], out_rows, comments)
    rev = ", ".join(f"{v:.3e}" for v in report["reversal_pressures"])
    print(f"[{name}] hydrostatic record: {report['points_in']} rows -> "
          f"{report['knots']} knots; pressure reversals at {rev or 'none'} Pa; "
          f"up to {report['truncated_at']:.4e} Pa, final J - 1 = "
          f"{report['final_strain_vol']:+.5f}")
    print(f"    --load-path hydrostatic --curve time-stress --data hydrostatic:{out}")
    return False


if __name__ == "__main__":
    sys.exit(main())
