# LCM cap-plasticity calibration (MatCal + Dakota)

Fit the LCM (Albany) `CapModel` to a measured **stress-strain** or
**load-displacement** curve, using
[MatCal](https://github.com/sandialabs/matcal) and
[Dakota](https://github.com/snl-dakota/dakota).

MatCal renders a templated `materials.yaml`, runs `Albany` on an internally
generated single element with every degree of freedom prescribed (so the finite
element problem *is* a material point), reads the resulting curve back, and
compares it to yours. Dakota adjusts the cap parameters and repeats until the
two curves agree. One parameter set can be constrained against several load
paths at once, which is how the shear-surface and cap-surface parameters become
separately identifiable.

**New here? Read [`SETUP.md`](SETUP.md) first** and work through it to the end.
It installs everything and finishes with a calibration that recovers a known
answer. This document assumes that is done.

---

## Contents

- [Every session starts like this](#every-session-starts-like-this)
- [Quick start](#quick-start)
- [Calibrating against your own data](#calibrating-against-your-own-data)
- [Kinematics](#kinematics)
- [Units](#units)
- [Reference](#reference)
- [Layout](#layout)
- [Platforms](#platforms)
- [Notes and gotchas](#notes-and-gotchas)
- [Verification status](#verification-status)

---

## Every session starts like this

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate matcal
cd ~/LCM/LCM/tools/calibration/harness
```

If anything misbehaves later, come back and run this first. It reports on
everything the harness needs and runs no simulations:

```bash
python calibrate.py check
```

---

## Quick start

Three commands, about a minute. This manufactures a curve from known
parameters and then recovers one of them, which confirms the whole chain works:

```bash
python calibrate.py check
python calibrate.py make-reference --load-path confined
python calibrate.py calibrate --load-path confined --param R:20:35:22
```

The last command must end with:

```
BEST: R: 28.0

***** X-CONVERGENCE *****
```

`R = 28` is what the reference was generated at. The `:22` on the end of
`R:20:35:22` starts the search away from it, so the run has something to
recover. Leaving it off starts the search already at the answer, which
converges instantly and demonstrates nothing.

---

## Calibrating against your own data

This is what the harness is for. Five decisions, then one command.

### 1. Put your curve in a CSV

One header row naming the columns, then two columns of numbers:

```
Axial Strain,Axial Stress
0.0,0.0
-0.002,-15600000.0
-0.004,-31200000.0
```

Blank lines and lines starting with `#` are ignored. The column names can be
anything; you tell the harness which to use in step 5.

Two things about the numbers matter, and both are common mistakes:

- **Base SI.** Strain dimensionless (not percent), stress in Pa, displacement
  in m, force in N. See [Units](#units).
- **Compression is negative.** These are compression load paths, and the
  simulation reports compressive stress and strain as negative. A
  compression-positive curve will not be matched.

The harness warns on standard input if your data looks like it has either
problem, but it cannot be certain, so it warns rather than stopping.

### 2. Choose the curve type

| `--curve` | Independent | Dependent | Units |
|-----------|-------------|-----------|-------|
| `true-stress-strain` (default) | logarithmic strain `ln(1 + u/L0)` | Cauchy (true) stress | -, Pa |
| `eng-stress-strain` | engineering strain `u/L0` | engineering stress `force/A0` | -, Pa |
| `load-displacement` | face displacement | face reaction force | m, N |
| `time-stress` | LOCA continuation parameter | Cauchy stress | -, Pa |

Pick whichever your instrument actually recorded, and read
[Kinematics](#kinematics) before choosing: under finite deformation these are
genuinely different curves, differing by up to 7 per cent at the strain levels
these decks reach. `load-displacement` is the safest when you are unsure,
because displacement and force mean the same thing under both formulations.

`true-stress-strain` is the default because it is the pair the
finite-deformation kernel actually works in. If your laboratory reports
engineering stress and strain, as most do, either pass
`--curve eng-stress-strain` every run or change the default once: see
[Changing the defaults](#changing-the-defaults).

### 3. Choose the load path

| `--load-path` | Loading | Constrains |
|---------------|---------|------------|
| `hydrostatic` | equal compression on all three axes | the cap: `R`, `W`, `D1`, `kappa0`, `calpha` |
| `confined` | 1D (oedometric) compression, lateral strains held at zero | shear and cap together |
| `triaxial` | three unequal compressive strains | shear and non-associative terms: `A`, `C`, `theta`, `psi`, `L`, `phi`, `Q` |

`--load-path` is repeatable. Each one you give becomes a separate comparison
that the *same* parameter set has to satisfy, which is the whole point: cap
parameters are material constants, so one set must fit every path at once.

### 4. Choose which parameters to fit

`--param NAME:LO:HI[:INIT]`, repeatable. `LO` and `HI` bound the search, `INIT`
is where it starts (defaulting to the Salem limestone value in the table
[below](#cap-parameters)). Bounds are in base SI: `A:5e8:8e8`, not `A:500:800`.

Fit few parameters at a time. Every parameter you add costs Albany runs and
makes it easier for the optimizer to find a curve that matches for the wrong
reasons. Anything you do not name keeps its default; override a default without
fitting it using `--set NAME=VALUE`.

### 5. Run it

```bash
python calibrate.py calibrate --load-path confined --curve eng-stress-strain --data "confined:/path/to/oedometer.csv:Axial Strain:Axial Stress" --param A:3e8:9e8
```

The `--data` argument is `LOADPATH:CSV:XCOL:YCOL`. The last two name the
columns of *your* file holding the independent and dependent quantities. Drop
them if your file already uses the harness's own names (`strain_xx`,
`stress_xx`, `displacement_x`, `force_x`). Give one `--data` per load path.
Wrap the whole argument in double quotes, as above, whenever a column name
contains a space.

The run prints which file it read and what it compared, then Dakota's progress,
then:

```
BEST: A: 681275829.17

***** RELATIVE FUNCTION CONVERGENCE *****
```

### Reading the result

- **A convergence message** ending in `CONVERGENCE` means Dakota stopped on
  purpose. `X-CONVERGENCE` and `ABSOLUTE FUNCTION CONVERGENCE` are the clean
  ones.
- **`SINGULAR CONVERGENCE`, or a result sitting exactly at its starting value,**
  means the objective did not respond to the parameter. Either the data is in
  the wrong units or sign (check the warnings), or that parameter is not
  identifiable from that load path (see the table in step 3).
- **A result pinned to `LO` or `HI`** means the true value is probably outside
  your bounds, or the data is off by a factor of `1e6` because it is in MPa.

Everything MatCal and Dakota wrote is left in `examples/calibration_run/`,
including one working directory per evaluation, so you can look at any
individual Albany run.

---

## Kinematics

**The harness runs the finite-deformation cap model by default.** The materials
file's `Finite Deformation` flag is templated, `--small-strain` flips it for one
run, and `DEFAULT_FINITE_DEFORMATION` in `site_matcal/lcm_model.py` flips it
permanently (see [Changing the defaults](#changing-the-defaults)).

The two kernels are not two discretizations of the same thing:

| | `--finite-deformation` (default) | `--small-strain` |
|---|---|---|
| Kinematics | multiplicative, `be = F Cp^-1 F^T` | additive |
| Integrated in | logarithmic elastic strain, Kirchhoff stress | infinitesimal strain |
| Consumes | `F`, `J`, `Fp` | `Strain` |
| Writes a strain field | **no** | yes (`Strain_i_j`) |
| Reports | Cauchy stress (converted by `1/J`) | Cauchy stress |

The same verified integrator runs in both; only the kinematics wrap around it.

Two consequences matter in practice.

**The model writes no strain field under finite deformation.** The harness
therefore reconstructs strain from the nodal displacement of the loaded face,
`u/L0` and `ln(1 + u/L0)`. That is exact here, because a single element with
every degree of freedom prescribed deforms homogeneously, and it means every
curve is available under both formulations.

**The strain and stress measures separate.** At the strains these decks reach,
logarithmic strain differs from engineering strain by 1.0 per cent at 2 per
cent nominal and 2.0 per cent at 4 per cent. Engineering stress `force/A0`
differs from Cauchy stress by the area change of the loaded face: nothing at
all on the confined path, whose lateral strains are held at zero so its area is
preserved, but a factor of `0.98^2 = 0.9604` on the hydrostatic path. Under
`--small-strain` all of this collapses and the measures coincide.

The response itself moves by more than any of that:

| Load path | peak Cauchy stress, small strain | finite deformation | difference |
|-----------|----------------------------------|--------------------|------------|
| confined | `3.106368e+08` Pa | `3.270823e+08` Pa | +5.3% |
| hydrostatic | `3.094229e+08` Pa | `3.313530e+08` Pa | +7.1% |
| triaxial | `3.085363e+08` Pa | `3.250095e+08` Pa | +5.3% |

### The Salem limestone defaults are small-strain values

The `SALEM_LIMESTONE` constants come from Sun, Chen and Ostien (2014), who
identified them under a small-strain formulation. They remain a sound **starting
point** under finite deformation, and that is how the harness uses them: as
`--param` defaults for `INIT`. They are not the finite-deformation answer to the
same data.

How much that matters is measurable. Generating the reference under
`--small-strain` and then fitting it with the finite-deformation model, starting
from the Salem values themselves, moves them:

| Parameter | Salem (small strain) | refit under finite deformation | shift |
|-----------|----------------------|-------------------------------|-------|
| `R` | `28.0` | `27.731329063` | -1.0% |
| `W` | `0.08` | `0.090377625831` | +13.0% |

So `R` is nearly formulation-independent and `W` is not. Do not quote a
published small-strain parameter set as a finite-deformation result, and do not
treat agreement with one as a validation.

Note that a **round trip is self-consistent either way**: `make-reference`
generates its reference with the same flag the subsequent `calibrate` uses, so
the recovery tests below return the exact input parameters under both
formulations. That checks the machinery, not the physics.

## Units

Everything here is **base SI**, with magnitudes written in scientific notation
rather than as prefixed units: `2.2547e10` Pa, never `22547` MPa. That covers
the defaults, the templated materials files, `--param` bounds and initial
values, `--set` overrides, and both columns of any `--data` file.

| Quantity | Unit | Where |
|----------|------|-------|
| stress | Pa | `A` `C` `N` `kappa0` `calpha` `elastic_modulus`, `stress_*` columns |
| 1/stress | 1/Pa | `D` `L` `D1` |
| 1/stress^2 | 1/Pa^2 | `D2` |
| dimensionless | - | `poissons_ratio` `theta` `R` `W` `psi` `phi` `Q`, `strain_*` columns |
| force | N | `force_*` columns |
| length | m | `displacement_*` columns |

The cap model is unit-agnostic and needs only one consistent system, so this is
a convention rather than a constraint. It matches the ACE permafrost production
decks, which are already in Pa and scientific notation
(`tests/LCM/ACE/MiniErosionPermafrost/materials_mechanical_permafrost.yaml`:
`A: 2.0e+06`, `kappa0: -1.0e+07`, `D1: 1.0e-08`), so a parameter set calibrated
here drops into one without rescaling. The verification decks in
`tests/LCM/CapModelPlasticity3D` that these templates came from are in base SI
too. It does mean the values differ from the MPa table of Sun, Chen and Ostien
(2014) by the appropriate power of `1e6`.

A curve supplied in MPa is fit by stress-like parameters `1e6` too small.
**Convert the data, not the harness.**

---

## Reference

### Actions

| Command | What it does |
|---------|--------------|
| `python calibrate.py check` | Reports on the environment. Runs no simulations. Exits nonzero if anything fails. |
| `python calibrate.py make-reference` | Runs Albany once per load path at the default parameters and writes the resulting curve to `examples/<load_path>_reference.csv`. Use it to generate practice data. |
| `python calibrate.py calibrate` | Runs the Dakota study. |

### Options

| Option | Meaning |
|--------|---------|
| `--load-path NAME` | `hydrostatic`, `confined` or `triaxial`. Repeatable. Default `confined`. |
| `--curve NAME` | `true-stress-strain` (default), `eng-stress-strain`, `load-displacement` or `time-stress`. |
| `--finite-deformation` / `--small-strain` | Kinematics. Finite deformation is the default; see [Kinematics](#kinematics). |
| `--param NAME:LO:HI[:INIT]` | Parameter to fit, base SI. Repeatable. |
| `--data LOADPATH:CSV[:XCOL:YCOL]` | Experimental data for one load path, base SI. Repeatable. Defaults to `examples/<load_path>_reference.csv`. |
| `--set NAME=VALUE` | Override a default without fitting it, base SI. Repeatable. |
| `--study gradient\|scipy` | Dakota gradient study (default) or SciPy. |
| `--platform rigel\|sirius\|cee` | Force a platform. Default: detected from the hostname. |
| `--core-limit N` | Concurrent Albany evaluations. Default 4. |
| `--out-dir DIR` | Where references and run directories go. Default `../examples`. |

Bad input is rejected before any simulation runs: unknown parameter or load
path names, bounds that are not increasing, an `INIT` outside its bounds, a
data file whose columns do not match the chosen curve.

### Changing the defaults

Two choices are made for you, and both are stated on every run: the header line
of `calibrate` reads `curve=... kinematics=...`, and `make-reference` records
both in the line it prints and nothing else changes silently.

| Default | Value | Override for one run | Change permanently |
|---------|-------|----------------------|--------------------|
| Curve | `true-stress-strain` | `--curve eng-stress-strain` (or any other) | `DEFAULT_CURVE` in `site_matcal/load_paths.py` |
| Kinematics | finite deformation | `--small-strain` | `DEFAULT_FINITE_DEFORMATION` in `site_matcal/lcm_model.py` |
| Load path | `confined` | `--load-path NAME`, repeatable | - |
| Cores | 4 | `--core-limit N` | - |

Each permanent default is a single named constant, read by both the library and
the command line, so editing it is the whole change. For example, to make
engineering stress-strain the site default:

```python
DEFAULT_CURVE = "eng-stress-strain"    # site_matcal/load_paths.py
```

`--help` picks the change up automatically: whichever kinematics constant is
set gets the `(default)` label, and the `--curve` help text names the current
default. Nothing else needs editing, and no deck or template is touched.

Changing a default does **not** rewrite existing reference CSVs. Those carry
the column names of the curve they were generated with, so a stale one is
caught by the column check rather than silently misread; regenerate with
`make-reference`.

`--curve time-stress` compares the LOCA continuation parameter (which runs over
`[0, 1]` and is affine in applied displacement) against axial Cauchy stress. It
is kept for regression checks that want the abscissa the deck stepped on rather
than a measured one.

### Cap parameters

Every name below is a placeholder in the templated materials file and can be
given to `--param` or `--set`. Defaults are Salem limestone, Table 1 of Sun,
Chen and Ostien, *Acta Geotechnica* **9** (2014) 903-934, converted to base SI.

| Name | Default | Unit | Name | Default | Unit |
|------|---------|------|------|---------|------|
| `elastic_modulus` | `2.2547e10` | Pa | `W` | `0.08` | - |
| `poissons_ratio` | `0.2524` | - | `D1` | `1.47e-9` | 1/Pa |
| `A` | `6.892e8` | Pa | `D2` | `0.0` | 1/Pa^2 |
| `C` | `6.752e8` | Pa | `calpha` | `1.0e11` | Pa |
| `D` | `3.94e-10` | 1/Pa | `psi` | `1.0` | - |
| `theta` | `0.0` | - | `N` | `6.0e6` | Pa |
| `R` | `28.0` | - | `L` | `3.94e-10` | 1/Pa |
| `kappa0` | `-8.05e6` | Pa | `phi` | `0.0` | - |
| | | | `Q` | `28.0` | - |

### Fields the simulation reports

Available as `--curve` components and as CSV column names:

| Field | Meaning |
|-------|---------|
| `time` | LOCA continuation parameter, `[0, 1]` |
| `stress_xx/yy/zz/xy` | Cauchy (true) stress |
| `stress_eng_x/y/z` | engineering stress, `force/A0` |
| `strain_eng_x/y/z` | engineering strain, `u/L0` |
| `strain_log_x/y/z` | logarithmic (true) strain, `ln(1 + u/L0)` |
| `displacement_x/y/z` | loaded-face displacement |
| `force_x/y/z` | loaded-face reaction force |
| `kappa`, `evp` | cap hardening parameter, volumetric plastic strain |
| `strain_xx/yy/zz/xy` | the model's own small-strain tensor. **`--small-strain` only**; the finite-deformation kernel never forms it. |

---

## Layout

```
tools/calibration/
  README.md                  this file: how to calibrate
  SETUP.md                   how to install everything, from scratch
  site_matcal/               top-level package MatCal auto-imports
    __init__.py              exposes helpers (also on the matcal namespace)
    register_factories.py    clean-shell env + jinja templating (on import)
    platforms.py             platform registry: rigel, sirius, cee
    load_paths.py            load-path and curve registries
    lcm_model.py             make_lcm_cap_model(...); SALEM_LIMESTONE defaults
    exodus_reader.py         read_lcm_cap_exodus() -> MatCal Data
  templates/
    materials.yaml           jinja cap params (CapModel)         hydrostatic, confined
    materials_triaxial.yaml  jinja cap params (CapModelTriaxial)  triaxial
    input_{hydrostatic,confined,triaxial}.yaml   Albany decks (single element)
  harness/
    calibrate.py             the CLI: check / make-reference / calibrate
  examples/                  generated references and run output (git-ignored)
```

The decks are copies of the verification decks in
`tests/LCM/CapModelPlasticity3D`, with the materials file jinja-templated.

---

## Platforms

Platform specifics live in `site_matcal/platforms.py`. All three are working.

| Platform | Albany | Dakota |
|----------|--------|--------|
| `cee` (`hpws*`) | `~/LCM/lcm-build-serial-gcc-release/src/Albany` | `/projects/dakota/install/rhel8/6.24.0` (on disk) |
| `rigel` | same | `~/dakota/6.24.0` (downloaded) |
| `sirius` | same | `~/dakota/6.24.0` (downloaded) |

None of them needs a runtime environment: the serial Albany build resolves its
Trilinos libraries through a baked-in RUNPATH. Selection is
`$LCM_MATCAL_PLATFORM`, then hostname, then rigel as the fallback. Override
just the executable with `$LCM_ALBANY`.

To add a platform, add a `Platform` entry in `site_matcal/platforms.py`: the
Albany path, any environment variables the build needs at run time (or a small
wrapper script that does the `module load`s and then `exec Albany "$@"`), and
hostname substrings for auto-detection. Nothing else changes; the model,
harness and readers are platform-agnostic.

---

## Notes and gotchas

- **One calibration per `python` command.** Dakota-as-a-library cannot run two
  studies in one interpreter; it segfaults on the second. This is a documented
  MatCal limitation. Separate calibrations means separate `python` commands.
- **Keep Dakota off `LD_LIBRARY_PATH`.** Dakota's programs and bindings resolve
  their own libraries. If Dakota's `bin` is on `LD_LIBRARY_PATH`, the Albany
  subprocess loads Dakota's bundled `libmpi.so.40` ahead of its own and
  segfaults at MPI finalize. `check` tests for this.
- **Identifiability depends on the load path.** The confined path is
  cap-dominated, so `A` and `C` are weakly constrained there: a fit can match
  the curve at the wrong `A` and `C`. Calibrate cap-active parameters (`R`,
  `W`, `D1`, `kappa0`) on confined and hydrostatic, and add the triaxial path
  to pin the shear and non-associative terms.
- **Templates end with a blank line on purpose.** jinja2 strips one trailing
  newline, and Albany's YAML parser fails at end-of-file without one.
- **The reference CSV is overwritten by `make-reference`,** including when you
  change `--curve`. If a later `calibrate` complains that a column is missing,
  regenerate the reference with the curve you actually want.

---

## Verification status

### Finite deformation (the default), all three platforms, 2026-08-26

sirius, rigel and CEE (`hpws00344`) each pulled the commit and returned
identical values to every printed digit. sirius ran the full table below;
rigel and CEE ran the three forward runs, the `true-stress-strain` and
`load-displacement` round trips, the `kappa0` round trip, and the
`--small-strain` peak.

Forward runs, peak Cauchy stress at the default parameters:

| Load path | peak `stress_xx` |
|-----------|------------------|
| confined | `3.270823e+08` Pa |
| hydrostatic | `3.313530e+08` Pa |
| triaxial | `3.250095e+08` Pa |

Round trips, each recovering the parameter its reference was generated at:

| Study | Result | Convergence | Cost |
|-------|--------|-------------|------|
| `R` from 22, confined, `true-stress-strain` | `R: 28.0` | X-CONVERGENCE | 15 s |
| `R` from 22, confined, `eng-stress-strain` | `R: 28.0` | X-CONVERGENCE | 15 s |
| `R` from 22, confined, `load-displacement` | `R: 28.0` | X-CONVERGENCE | 15 s |
| `R` from 22, confined, `time-stress` | `R: 28.0` | X-CONVERGENCE | 15 s |
| `kappa0` from `-1.2e7`, hydrostatic | `kappa0: -8050000.0` | X-CONVERGENCE | 12 s |
| `R`, `W` from (22, 0.05), confined + hydrostatic | `R: 28.0`, `W: 0.08` | X-CONVERGENCE | 28 s on 4 cores |
| `R` from 22, confined, `--study scipy` | `R: 27.983` | (looser tolerance by design) | 17 s |

All four curves recover the same answer exactly, which is the point: they are
four reductions of one simulation, and the objective is conditioned onto the
data range either way.

Cross-formulation, the check behind the
[Salem caveat](#the-salem-limestone-defaults-are-small-strain-values):
a reference generated under `--small-strain` and then fitted with the
finite-deformation model, starting from the Salem values, returns
`R: 27.731329063`, `W: 0.090377625831` (RELATIVE FUNCTION CONVERGENCE).

### Small strain (`--small-strain`), all three platforms, 2026-08-26

| Platform | OS | MatCal | Dakota |
|----------|----|--------|--------|
| sirius | Fedora 44 | 1.4.28 | 6.24.0, `~/dakota` |
| rigel | RHEL 9.8 | 1.4.27 | 6.24.0, `~/dakota` |
| cee (`hpws00344`) | RHEL 9.7 | 1.4.27 | 6.24.0, `/projects` |

All three confirmed the confined small-strain peak of `3.106368e+08` Pa under
the current code, which is what pins the two kinematics apart. The fuller
small-strain sweep below predates the finite-deformation default and was run
when small strain was the only option.

All three return identical values to every printed digit. Forward-run peaks are
`3.106368e+08`, `3.094229e+08` and `3.085363e+08` Pa for confined, hydrostatic
and triaxial. The round trips recover `R: 28.0` (from 22, both under
stress-strain and load-displacement), `kappa0: -8050000.0` (from `-1.2e7`) and
`R: 28.0`, `W: 0.08` across two paths, all X-CONVERGENCE.

Coverage of that multi-platform run, all matching exactly where run: sirius ran
everything; rigel ran all three forward runs and the first four round trips;
CEE ran the confined and hydrostatic forward runs and three round trips.
`--study scipy` was run on sirius only.

Only timings differ between platforms: rigel takes about twice as long per
study as sirius (31 s against 15 s for the single-parameter confined round
trip). That is per-core speed on what is a serial workload, not a configuration
problem; pinning `OMP_NUM_THREADS=1` on rigel's 336-core node changes nothing
(30.3 s against 31.6 s).

### Notes on both

`kappa0` is the case the unit convention actually touches: Dakota normalizes
each parameter onto `[0, 1]` over its bounds, so a stress-like parameter is
searched no differently from a dimensionless one.

**Unit invariance**, checked when the harness moved to base SI: each load path
rerun with the original MPa parameter set reproduces the Pa run to a maximum
relative difference of `3e-15` in stress (confined `2.4e-15`, hydrostatic
`2.9e-15`, triaxial `2.4e-15`), which is the roundoff floor. The drift
tolerance is scaled by `E^2` in `CapModel_Def.hpp` and the Newton test acts on
prescribed displacements, so nothing in the solve carries an absolute stress
scale. Converged calibration results are likewise unaffected, because
`CurveBasedInterpolatedObjective` conditions each field onto a fixed range
before differencing (MatCal's `RangeDataConditioner`), leaving the objective
dimensionless.
