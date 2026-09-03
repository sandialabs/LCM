#!/usr/bin/env python3
"""Run a txc deck twice so the confining pressure acts as a FOLLOWER load.

MatCal invokes this in place of Albany for the ``txc`` load path when a run
asks for the follower correction. It takes the rendered deck as its only
argument, leaves the same Exodus file behind, and exits with Albany's status,
so nothing downstream can tell the difference except the answer.

WHY. A triaxial cell surrounds the specimen with fluid at constant pressure,
which acts on the CURRENT membrane area: a follower load. Albany's ``P``
condition builds its normals and areas from ``coordVec``, the reference
coordinates (``PHAL_Neumann_Def.hpp``, ``calc_press``), so what it holds fixed
is force per UNDEFORMED area. As the specimen shortens axially and spreads
laterally the lateral face area changes and the Cauchy confining stress drifts
with it: measured on the fitted frozen-sand set, 3.5 per cent at 2 per cent
axial strain and 18.3 per cent by 28 per cent. The drift is in the direction
that HIDES softening (rising confinement strengthens the material), so a
softening parameter fitted against it comes out biased.

HOW. A dead load applied at magnitude P gives sigma_nn = -P/r, with r the face
area ratio. So applying P(t) = Pc * r(t) holds the Cauchy stress at -Pc, which
is what the fluid does. r is not known before the run, so:

  pass 1  run the deck as written (constant Pc) and read the strain history
  pass 2  rewrite the two lateral NBC arrays with P(t) = Pc * r(t) and rerun

That is one fixed-point iteration. It converges because r depends only weakly
on the small change in confinement: measured, pass 2 holds the lateral Cauchy
stress within 0.5 per cent of -Pc where the dead load had drifted to 18 per
cent. A third pass is not worth its cost.

The two lateral faces get their own array. The y+ face area is
(1 + eps_x)(1 + eps_z) and the z+ face area is (1 + eps_x)(1 + eps_y); on a
triaxial path the two are nearly equal by symmetry but they are not identical,
and using each face's own ratio costs nothing.

This is a stopgap for the real fix, a follower-pressure condition in Albany
(which would also serve the ACE wave-pressure path, whose bluff faces undergo
far larger displacement than a laboratory specimen). See the developers guide.
"""

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from site_matcal.exodus_reader import read_lcm_cap_exodus  # noqa: E402
from site_matcal.platforms import get_albany               # noqa: E402

# One array entry per this many output steps, so the pressure history is
# resolved without writing a 400-entry table into the deck.
NUM_PRESSURE_POINTS = 60

# The lateral side sets of the single-element mesh, with the two in-plane
# axes whose extents make up each face's area. SideSet3 is y+, SideSet5 z+.
LATERAL_FACES = {"SideSet3": ("x", "z"), "SideSet5": ("x", "y")}

NBC = (r"(Time Dependent NBC on SS {ss} for DOF all set P:\n)"
       r"[ \t]*Time Values: \[(?P<times>[^\]]*)\]\n"
       r"[ \t]*BC Values: \[(?P<values>.*)\]\n")


def run_albany(albany, deck, log):
    with open(log, "w") as fh:
        return subprocess.call([albany, deck], stdout=fh, stderr=subprocess.STDOUT)


def deck_field(text, name):
    """The value of a top-level ``name: value`` line, e.g. the Exodus file."""
    m = re.search(rf"^\s*{re.escape(name)}:\s*(\S+)\s*$", text, re.M)
    return m.group(1) if m else None


def read_pressure_block(text, side_set):
    """Return ``(times, values)`` of one lateral NBC as lists of float."""
    m = re.search(NBC.format(ss=side_set), text)
    if m is None:
        return None, None
    times = [float(v) for v in m.group("times").split(",")]
    values = [float(v) for v in re.findall(r"[-+0-9.eE]+", m.group("values"))]
    return times, values


def write_pressure_block(text, side_set, times, values):
    tv = "[" + ", ".join(f"{v:.8e}" for v in times) + "]"
    bv = "[" + ", ".join(f"[{v:.8e}]" for v in values) + "]"
    new, n = re.subn(NBC.format(ss=side_set),
                     lambda m: (m.group(1)
                                + f"        Time Values: {tv}\n"
                                + f"        BC Values: {bv}\n"),
                     text, count=1)
    if n != 1:
        sys.exit(f"txc_twopass: could not rewrite the {side_set} pressure block")
    return new


def interp(x, xs, ys):
    """Linear interpolation on an increasing abscissa, clamped at both ends."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo = 0
    while lo < len(xs) - 2 and xs[lo + 1] < x:
        lo += 1
    span = xs[lo + 1] - xs[lo]
    w = 0.0 if span == 0.0 else (x - xs[lo]) / span
    return ys[lo] + w * (ys[lo + 1] - ys[lo])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit("usage: txc_twopass.py <deck.yaml>")
    deck = argv[0]
    albany = get_albany()

    rc = run_albany(albany, deck, "pass1.log")
    if rc != 0:
        sys.stderr.write("txc_twopass: pass 1 failed; see pass1.log\n")
        return rc

    text = open(deck).read()
    exo = deck_field(text, "Exodus Output File Name")
    if exo is None or not os.path.isfile(exo):
        sys.stderr.write("txc_twopass: pass 1 wrote no Exodus file; "
                         "leaving the single-pass result in place\n")
        return 0

    # Full history from the undeformed state: preload_time 0 keeps every step
    # and refers the strains to the reference configuration, which is the
    # frame the reference-normal pressure condition works in.
    data = read_lcm_cap_exodus(exo, preload_time=0.0)
    time = list(data["time"])
    strain = {a: list(data[f"strain_eng_{a}"]) for a in ("x", "y", "z")}

    step = max(1, len(time) // NUM_PRESSURE_POINTS)
    grid = sorted(set(list(range(0, len(time), step)) + [len(time) - 1]))

    for side_set, (a, b) in LATERAL_FACES.items():
        times, values = read_pressure_block(text, side_set)
        if times is None:
            sys.stderr.write(f"txc_twopass: no {side_set} pressure block; "
                             f"leaving the single-pass result in place\n")
            return 0
        # The deck ramps 0 -> Pc over the consolidation stage and holds it,
        # so the held value is Pc and the ramp ends at the second time point.
        pressure = values[-1]
        preload_end = times[1] if len(times) > 2 else 0.0

        new_times, new_values = [], []
        for i in grid:
            t = time[i]
            ratio = (1.0 + strain[a][i]) * (1.0 + strain[b][i])
            p = pressure * ratio
            if preload_end > 0.0 and t < preload_end:
                p *= t / preload_end          # keep the consolidation ramp
            new_times.append(t)
            new_values.append(p)
        new_times[0], new_values[0] = 0.0, 0.0
        text = write_pressure_block(text, side_set, new_times, new_values)

    open(deck, "w").write(text)
    return run_albany(albany, deck, "pass2.log")


if __name__ == "__main__":
    sys.exit(main())
