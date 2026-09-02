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
- [Triaxial compression: the `txc` path](#triaxial-compression-the-txc-path)
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
| `dev-stress-strain` | engineering strain `u/L0` | deviatoric Cauchy stress `sigma_1 - sigma_3` | -, Pa |
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
| `txc` | consolidate to a confining pressure, then shear at constant pressure | the same, and this is the one that takes measured data |

The first three prescribe every strain component and are the verification
paths, taken from `tests/LCM/CapModelPlasticity3D`. `txc` is the laboratory
test: see [Triaxial compression](#triaxial-compression-the-txc-path) before
using it.

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

## Triaxial compression: the `txc` path

The other three paths prescribe all 24 degrees of freedom of the element, so
the finite element problem is a material point with no solve in it. `txc` is
the test a triaxial cell actually runs, and it is different in kind: only the
axial direction is prescribed. The two lateral faces carry a constant normal
traction and their displacement is **predicted**. That is what lets the path
see dilatancy, because the specimen is free to change volume against the
confinement, and it is why this is the path that takes measured triaxial data.

### What the deck does

One continuation run, two stages:

| `Time` | stage | axial face | lateral faces |
|--------|-------|-----------|---------------|
| `0` to `preload_fraction` | consolidation | displaced to `-Pc(1-2nu)/E` | traction ramps `0` to `-Pc` |
| `preload_fraction` to `1` | shear | displaced by `axial_strain` further | traction held at `-Pc` |

The consolidation stage is exact, and not by luck: under lateral tractions
`-Pc`, prescribing the axial strain at the elastic hydrostatic value returns an
axial stress of exactly `-Pc`, so the state at the end of the stage is
hydrostatic. Measured: the residual deviator is `5.4e-4` of `Pc` under finite
deformation and `9e-15` of it under small strain. It is exact only while the
response to `Pc` is elastic, which holds whenever `Pc` is inside the initial
cap, `3 Pc < |kappa0|`.

The elastic hydrostatic strain is computed **in the deck**, from
`elastic_modulus` and `poissons_ratio`, and rendered on every evaluation. So it
tracks those two if you calibrate them rather than going stale.

The reader then trims the consolidation stage off the curve it returns and
refers displacement, strain and the reference area to the consolidated state.
The curve you compare against therefore starts at zero strain under a
hydrostatic stress, which is exactly where a laboratory starts measuring.

### Three decisions it adds

```bash
python calibrate.py calibrate --load-path txc --curve dev-stress-strain \
    --defaults permafrost \
    --set confining_pressure=1.0e6 --set axial_strain=-0.20 \
    --data "txc:/path/to/test.csv:Axial Strain:q" --param A:1.5e6:2.5e6
```

- **`--set confining_pressure`** (Pa, a positive magnitude) is the cell
  pressure. One per curve.
- **`--set axial_strain`** (negative) is how far the shear stage goes. Make it
  reach past the end of your data: MatCal interpolates the model onto the
  data's abscissa, and a model curve that stops short leaves the far end
  unconstrained.
- **`--curve dev-stress-strain`** compares `q = sigma_1 - sigma_3` against
  engineering axial strain, which is what a triaxial laboratory reports. The
  confining pressure cancels out of the difference, so the data file needs only
  the two columns the laboratory already has. Negative in compression, like
  everything else here.

### Use the permafrost defaults, not Salem

`--defaults permafrost` is not a convenience on this path, it is close to a
requirement. Salem limestone has its cap at `kappa0 = -8.05e6` Pa and an
unconfined deviatoric strength of `3.5e7` Pa, so **every** triaxial test of it
at a confinement inside its own cap is cap-dominated rather than a test of the
shear envelope. Worse, its strength comes from `A - C = 1.4e7` Pa, a difference
of two numbers near `6.8e8`: a 4 per cent change in `A` alone swings `A - C`
through zero. Fitting `A` on `txc` with the Salem set and any honest bounds
gives a response that is not even monotonic in `A` (peak `q` measured at
`-49.8`, `-19.5`, `-24.1`, `-17.9`, `-46.8` MPa for `A` from `6.4e8` to
`6.892e8`). That is the parameter set, not the deck. With
`--defaults permafrost` the same sweep is a straight line.

### What it costs in fidelity

Three things are worth knowing before quoting a number from this path.

- **The confining pressure is a dead load, not a follower load.** Albany's `P`
  condition acts on the *reference* normal, so what is held constant is the
  force on the undeformed face, not the Cauchy stress on the deformed one. The
  lateral stress therefore drifts as the specimen dilates: measured `-1.0012e6`
  at the start of shear and `-1.0679e6` at 20 per cent axial strain, so 0.2 per
  cent at the peak and 6.7 per cent at the end of a long test. A follower
  pressure condition would remove this and does not exist in Albany today.
- **`force_y` and `force_z` are not reaction forces here,** because those faces
  are free. `stress_eng_y` and `stress_eng_z` are meaningless on this path.
  Everything axial is fine: the axial face is prescribed on every path.
- **The area convention for `q` is yours to match.** The model reports the true
  deviatoric Cauchy stress. A laboratory that applied its own area correction,
  or none, is reporting something else at large strain. Convert the data.

### One confining pressure per run

`--set` is global to the run, so a single `calibrate` fits one confining
pressure. A confining-pressure series is what pins `theta` and separates it
from `A`, so fitting the series simultaneously is the obvious next step; it
wants a MatCal *state* per pressure rather than a `--set`. Not built yet.

---

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
| `--set NAME=VALUE` | Override a cap parameter or a deck constant (`confining_pressure`, `axial_strain`, `preload_fraction`) without fitting it, base SI. Repeatable. |
| `--defaults salem\|permafrost` | Starting parameter set: where `--param` `INIT` and every un-fitted placeholder come from. Default `salem`. |
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
| Parameter set | Salem limestone | `--defaults permafrost` | `DEFAULT_SET` in `site_matcal/lcm_model.py` |
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

A second named set, `PERMAFROST_FROZEN`, holds the frozen end member of
`tests/LCM/ACE/MiniErosionPermafrost/materials_mechanical_permafrost.yaml`
(`E = 1.0e9`, `nu = 0.2`, `A = 2.0e6`, `C = 2.679e5`, `theta = 0.10`,
`kappa0 = -1.0e7`, `W = 0.60`, `R = Q = 5.0`, `phi = 0.08`). Select it with
`--defaults permafrost`. It is the set to start a frozen-soil calibration from
and the one the `txc` path needs.

### Fields the simulation reports

Available as `--curve` components and as CSV column names:

| Field | Meaning |
|-------|---------|
| `time` | LOCA continuation parameter, `[0, 1]` |
| `stress_xx/yy/zz/xy` | Cauchy (true) stress |
| `stress_dev_x/y/z` | deviatoric Cauchy stress about that axis, `sigma_aa - (sigma_bb + sigma_cc)/2`; `sigma_1 - sigma_3` on a triaxial path |
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
    lcm_model.py             make_lcm_cap_model(...); SALEM_LIMESTONE and
                             PERMAFROST_FROZEN parameter sets
    exodus_reader.py         read_lcm_cap_exodus() -> MatCal Data
  templates/
    materials.yaml           jinja cap params (CapModel)         hydrostatic, confined, txc
    materials_triaxial.yaml  jinja cap params (CapModelTriaxial)  triaxial
    input_{hydrostatic,confined,triaxial}.yaml   Albany decks (single element)
    input_txc.yaml           Albany deck, triaxial cell (jinja: confining
                             pressure, strain range, consolidation strain)
  harness/
    calibrate.py             the CLI: check / make-reference / calibrate
  examples/                  generated references and run output (git-ignored)
```

The first three decks are copies of the verification decks in
`tests/LCM/CapModelPlasticity3D`, with the materials file jinja-templated.
`input_txc.yaml` has no counterpart there: it is the laboratory test, and its
own boundary conditions are templated as well.

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
- **On `txc`, keep `3 * confining_pressure` inside `|kappa0|`.** The exact
  consolidation stage assumes the response to the cell pressure is elastic. Past
  the cap it is not, and the state at the end of the stage stops being
  hydrostatic, silently.
- **`txc` steps adaptively, and the other three do not.** With a constant step
  LOCA reports a step whose Newton solve failed and carries straight on, and
  Albany writes the unconverged state to Exodus like any other point. The curve
  then contains points that are not solutions, and since which steps fail
  depends on the parameters, the objective stops being a smooth function of
  them. Adaptive stepping cuts the step and retries. Failures are worth
  grepping for in `simulation.out` when a `txc` fit misbehaves.
- **The reference CSV is overwritten by `make-reference`,** including when you
  change `--curve`. If a later `calibrate` complains that a column is missing,
  regenerate the reference with the curve you actually want.

---

## Verification status

### Triaxial compression (`txc`), sirius, 2026-09-02

Added with the path. Everything below is `--defaults permafrost`,
`--curve dev-stress-strain`, finite deformation, `confining_pressure = 1.0e6`,
`axial_strain = -0.20`, and zero LOCA convergence failures in every run.

Forward run, the shape a triaxial test should have:

| quantity | value |
|----------|-------|
| deviator at the end of consolidation | `-5.4e+02` Pa, against `Pc = 1.0e+06` |
| peak `q` | `-4.272999e+06` Pa at 0.67 per cent axial strain |
| `q` at 20 per cent axial strain | `-3.98e+06` Pa |
| volumetric strain | `-0.0019` (compaction) then `+0.144` (dilation) |
| lateral Cauchy stress | `-1.0012e+06` Pa at the peak, `-1.0679e+06` Pa at the end |

The same run under `--small-strain`: deviator at the end of consolidation
`-9.1e-09` Pa, so the consolidation stage is exact to roundoff once the finite
deformation of the dead pressure load is out of the way; peak `q`
`-4.258566e+06` Pa, 0.34 per cent from the finite-deformation value.

Sweeping `A` about the truth, to check the objective is well conditioned:

| `A` | `1.6e6` | `1.8e6` | `2.0e6` | `2.2e6` | `2.4e6` |
|-----|---------|---------|---------|---------|---------|
| peak `q` (Pa) | `-3.4304e+06` | `-3.8515e+06` | `-4.2730e+06` | `-4.6946e+06` | `-5.1164e+06` |

Straight, to three digits, in equal steps. Round trips against a reference
generated at the values above:

| Study | Result | Convergence |
|-------|--------|-------------|
| `A` from `1.7e6` | `A: 2000000.0` | X-CONVERGENCE |
| `A` with no `INIT` (taken from `--defaults permafrost`) | `A: 2000000.0` | X- AND RELATIVE FUNCTION CONVERGENCE |
| `phi` from `0.02` | `phi: 0.079999999895` | X-CONVERGENCE |
| `A`, `theta` from `(1.6e6, 0.20)` | `A: 2000000.0003`, `theta: 0.099999999957` | X-CONVERGENCE |

`phi` is the point of the path: it is a non-associative *flow* parameter, and it
is recoverable here only because the lateral faces are free, so the dilatancy it
controls feeds back into the axial response. The last row says `A` and `theta`
separate even at a single confining pressure, because the confinement drifts
under the dead load and the envelope is therefore sampled over a range of `I1`.
Do not read that as a reason to skip the pressure series: it is a weak
separation resting on an artifact.

The three verification paths were rerun after this change and reproduce every
value in the table below to the digit, including all six round trips.

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
