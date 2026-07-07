#!/usr/bin/env python3
#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#
# Parallel-consistency check for ACE element erosion.
#
# Asserts that a 4-rank (epu-merged) run erodes the IDENTICAL set of cells as
# the serial reference, comparing by ELEMENT CENTROID POSITION rather than by
# frame index or element id. This is deliberate:
#
#   * The np4 run takes a partition-dependent solver cutback, so it produces a
#     DIFFERENT number of output frames and a slightly different end time than
#     serial even though the final death SET is identical. A frame-by-frame
#     exodiff therefore cannot be used -- we only compare the LAST time step of
#     each file.
#   * epu renumbers element ids when it stitches the per-rank pieces back
#     together, so comparing cell_death by element id is meaningless. Centroid
#     position is the partition-invariant identity of a cell.
#
# An exodus file is a netCDF file, so we read it directly with netCDF4:
# read the last-time-step `cell_death` element variable for every block, mark a
# cell dead when cell_death >= THRESHOLD, compute its centroid from
# connect<blk> + coordx/coordy/coordz, and require the two dead-centroid SETS to
# be equal.
#
# Usage:  compare_death_positions.py <serial.e> <parallel_epu.e>
# Exit 0 if the dead-cell sets match, non-zero (with a diff report) otherwise.

import sys
import numpy as np
import netCDF4

# A cell is "dead" once cell_death crosses this value. cell_death is written as
# a hard 0/1 flag in this problem, so 0.5 is a safe midpoint that also tolerates
# any future fractional (gradual) death ramp settling at 1.0.
THRESHOLD = 0.5

# Centroids are compared after rounding to this many decimals, to absorb the
# ~1e-12 reduction-order noise between a serial and a 4-rank coordinate read.
# The mesh spacing here is 0.25, so 6 decimals is enormous headroom.
ROUND_DECIMALS = 6


def dead_centroids(path):
    """Return the set of (x, y, z) centroids of all dead cells at the last step."""
    d = netCDF4.Dataset(path)
    try:
        names = [
            b"".join(r).decode("ascii", "ignore").strip().strip("\x00")
            for r in d.variables["name_elem_var"][:]
        ]
        if "cell_death" not in names:
            raise SystemExit("FATAL: 'cell_death' element variable not found in %s" % path)
        ci = names.index("cell_death") + 1  # exodus var arrays are 1-based

        x = np.asarray(d.variables["coordx"][:])
        y = np.asarray(d.variables["coordy"][:])
        z = np.asarray(d.variables["coordz"][:])

        nblk = d.dimensions["num_el_blk"].size
        dead = set()
        for b in range(1, nblk + 1):
            vname = "vals_elem_var%deb%d" % (ci, b)
            if vname not in d.variables:
                # Block may not carry this variable per the elem-var truth table.
                continue
            conn = np.asarray(d.variables["connect%d" % b][:])  # (nel, npe), 1-based
            vals = np.asarray(d.variables[vname][-1, :])        # last time step
            for e in range(conn.shape[0]):
                if vals[e] >= THRESHOLD:
                    nodes = conn[e, :] - 1
                    cx = round(float(x[nodes].mean()), ROUND_DECIMALS)
                    cy = round(float(y[nodes].mean()), ROUND_DECIMALS)
                    cz = round(float(z[nodes].mean()), ROUND_DECIMALS)
                    dead.add((cx, cy, cz))
        return dead
    finally:
        d.close()


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_death_positions.py <serial.e> <parallel_epu.e>")
    serial_path, parallel_path = sys.argv[1], sys.argv[2]

    serial = dead_centroids(serial_path)
    parallel = dead_centroids(parallel_path)

    print("serial   dead cells: %d  (%s)" % (len(serial), serial_path))
    print("parallel dead cells: %d  (%s)" % (len(parallel), parallel_path))

    if serial == parallel:
        print("PASS: serial and np4 erode the identical cell set (%d cells)." % len(serial))
        return 0

    only_serial = sorted(serial - parallel)
    only_parallel = sorted(parallel - serial)
    print("FAIL: erosion patterns differ.")
    print("  dead only in serial   (%d): %s" % (len(only_serial), only_serial))
    print("  dead only in parallel (%d): %s" % (len(only_parallel), only_parallel))
    return 1


if __name__ == "__main__":
    sys.exit(main())
