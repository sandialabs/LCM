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
# An exodus file is a netCDF file, so we read it directly: read the last-time-
# step `cell_death` element variable for every block, mark a cell dead when
# cell_death >= THRESHOLD, compute its centroid from connect<blk> +
# coordx/coordy/coordz, and require the two dead-centroid SETS to be equal.
#
# The exodus output is classic (64-bit-offset) netCDF, so we read it with
# whichever reader is available: netCDF4 if installed (developer machines), else
# scipy.io.netcdf_file, which ships with the numpy/scipy stack the nightly
# already has. Requiring netCDF4 made this test fail spuriously on hosts that
# lack it (the import error was reported as an erosion-set mismatch).
#
# Usage:  compare_death_positions.py <serial.e> <parallel_epu.e>
# Exit 0 if the dead-cell sets match, non-zero (with a diff report) otherwise.

import sys
import numpy as np


class _Exo:
    """Minimal read-only exodus/netCDF accessor over netCDF4 or scipy.

    Exposes .var(name) -> ndarray, .has(name) -> bool, and .num_el_blk -> int,
    normalizing the small API differences between the two backends.
    """

    def __init__(self, path):
        self._backend = None
        try:
            import netCDF4  # noqa: F401
            self._d = netCDF4.Dataset(path)
            self._backend = "netCDF4"
        except ImportError:
            from scipy.io import netcdf_file
            # mmap=False so arrays stay valid after the file is closed.
            self._d = netcdf_file(path, "r", mmap=False)
            self._backend = "scipy"

    def has(self, name):
        return name in self._d.variables

    def var(self, name):
        return np.asarray(self._d.variables[name][:])

    @property
    def num_el_blk(self):
        dim = self._d.dimensions["num_el_blk"]
        # netCDF4 returns a Dimension object; scipy returns a plain int.
        return dim.size if hasattr(dim, "size") else int(dim)

    def close(self):
        self._d.close()

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
    d = _Exo(path)
    try:
        names = [
            b"".join(r).decode("ascii", "ignore").strip().strip("\x00")
            for r in d.var("name_elem_var")
        ]
        if "cell_death" not in names:
            raise SystemExit("FATAL: 'cell_death' element variable not found in %s" % path)
        ci = names.index("cell_death") + 1  # exodus var arrays are 1-based

        x = d.var("coordx")
        y = d.var("coordy")
        z = d.var("coordz")

        nblk = d.num_el_blk
        dead = set()
        for b in range(1, nblk + 1):
            vname = "vals_elem_var%deb%d" % (ci, b)
            if not d.has(vname):
                # Block may not carry this variable per the elem-var truth table.
                continue
            conn = d.var("connect%d" % b)         # (nel, npe), 1-based
            vals = d.var(vname)[-1, :]             # last time step
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
