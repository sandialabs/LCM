# LCM cap-plasticity calibration (MatCal + Dakota)

Calibrate the LCM (Albany) `CapModel` against material-point (single-element)
experiments using [MatCal](https://github.com/sandialabs/matcal) and
[Dakota](https://github.com/snl-dakota/dakota).

MatCal drives a `UserExecutableModel`: it renders a jinja-templated
`materials.yaml`, runs `Albany <deck>` on an internally generated single
element (all DOFs prescribed → a true material point), and reads the resulting
Exodus stress/strain series back for the objective. The same cap parameter set
can be constrained against several load paths at once (hydrostatic, confined,
triaxial), which is how the shear- and cap-surface parameters become
identifiable.

## Units

Everything in this harness is in **base SI**, with magnitudes written in
scientific notation rather than as prefixed units: stress in Pa (`2.2547e10`,
not `22547` MPa), never kPa/MPa/GPa. That covers the `SALEM_LIMESTONE`
defaults, the templated materials files, `--param` bounds and initial values,
`--set` overrides, the stress column of any `--data` file, and the stresses the
Exodus reader returns.

| Dimension | Parameters |
|-----------|------------|
| stress, Pa | `A` `C` `N` `kappa0` `calpha` `elastic_modulus` |
| 1/stress, 1/Pa | `D` `L` `D1` |
| 1/stress^2, 1/Pa^2 | `D2` |
| dimensionless | `poissons_ratio` `theta` `R` `W` `psi` `phi` `Q` |

The cap model itself is unit-agnostic and only requires one consistent system,
so this is a convention, not a constraint. It is chosen to match the ACE
permafrost production decks, which are already in Pa and scientific notation
(`tests/LCM/ACE/MiniErosionPermafrost/materials_mechanical_permafrost.yaml`:
`A: 2.0e+06`, `kappa0: -1.0e+07`, `D1: 1.0e-08`): a parameter set calibrated
here drops into one without rescaling. The verification decks in
`tests/LCM/CapModelPlasticity3D` that these templates were copied from are in
base SI too, so the two carry identical numbers. It does mean the values here
differ from the MPa table of Sun, Chen & Ostien (2014) by the appropriate power
of 1e6; the two parameter sets agree to roundoff, see "Verification status".

A curve supplied in MPa would be fit by stress-like parameters 1e6 too small,
and would not be detected: the objective compares raw values. Convert the data,
not the harness.

## Layout

```
tools/calibration/
  site_matcal/               top-level package MatCal auto-imports (from site_matcal import *)
    __init__.py              exposes helpers (also on the matcal namespace)
    register_factories.py    clean-shell env + jinja templating (side-effect on import)
    platforms.py             platform registry: rigel, sirius (local) + cee
    load_paths.py            hydrostatic / confined / triaxial deck registry
    lcm_model.py             make_lcm_cap_model(load_path=...); SALEM_LIMESTONE defaults
    exodus_reader.py         read_lcm_cap_exodus() -> MatCal Data (time, stress_*, kappa, evp)
  templates/
    materials.yaml           jinja cap params (CapModel)        -- hydrostatic, confined
    materials_triaxial.yaml  jinja cap params (CapModelTriaxial) -- triaxial (same placeholders)
    input_{hydrostatic,confined,triaxial}.yaml   Albany decks (single element)
  harness/
    calibrate.py             CLI: make-reference / calibrate; multi-load-path, platform-aware
  examples/
    *_reference.csv          demo "experiments" generated at the defaults
  README.md
```

The decks are copies of the verification decks in
`tests/LCM/CapModelPlasticity3D` (same single-element material-point setup),
with the materials file jinja-templated for calibration.

## Environment

Setup is a one-time task done outside the repo (Miniforge + a `matcal` conda
env with Python 3.12 + Dakota 6.24). The full, reproducible bring-up for each
platform is in **[`docs/SETUP.md`](docs/SETUP.md)**. Once done, the env's
activate hook puts this directory on `PYTHONPATH`, so:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate matcal
```

makes `import site_matcal` work (the "no site matcal" warning disappears) and
puts the Dakota CLI + bindings on the path.

## Usage

```bash
conda activate matcal
cd tools/calibration/harness

# 1. Synthetic "experiments" at the default parameters (one Albany run each):
python calibrate.py make-reference --load-path confined --load-path hydrostatic

# 2. Calibrate cap-active parameters against those paths (one Dakota study):
python calibrate.py calibrate --load-path confined --load-path hydrostatic --param R:20:35 --param W:0.02:0.15 --study gradient --core-limit 4

# Stress-like parameters take base-SI bounds (Pa), e.g. the cap branch point:
python calibrate.py calibrate --load-path hydrostatic --param kappa0:-2.0e7:-2.0e6:-1.2e7
```

- `--param NAME:LO:HI[:INIT]` — repeatable; `NAME` must match a jinja
  placeholder. `INIT` defaults to the Salem-limestone value. Bounds are in base
  SI (see "Units"): `A:5e8:8e8`, not `A:500:800`.
- `--load-path` — repeatable; each becomes a MatCal evaluation set.
- `--data LOADPATH:CSV` — real experimental data (columns `time,stress_zz`
  with stress in Pa; `time`∈[0,1] maps to applied strain for these decks).
  Defaults to `examples/<load_path>_reference.csv`.
- `--set NAME=VALUE` — override a fixed (non-calibrated) parameter default,
  in base SI.
- `--platform rigel|sirius|cee` — force a platform (default: auto by hostname).
- `--study gradient|scipy` — Dakota gradient (default) or SciPy.

Calibratable placeholders: `A C R W D1 D2 kappa0 calpha N theta psi L phi Q D
elastic_modulus poissons_ratio` (dimensions in the table above). Anything not
calibrated keeps its `SALEM_LIMESTONE` default (MatCal precedence: study params
> model constants).

## Platforms

Platform specifics (Albany path, environment) live in `site_matcal/platforms.py`.

| Platform | Status | Albany | Dakota | Environment |
|----------|--------|--------|--------|-------------|
| `rigel`  | working | `~/LCM/lcm-build-serial-gcc-release/src/Albany` | `~/dakota/6.24.0` (downloaded) | none (serial build resolves Trilinos via RUNPATH) |
| `sirius` | working | `~/LCM/lcm-build-serial-gcc-release/src/Albany` | `~/dakota/6.24.0` (downloaded) | none (serial build resolves Trilinos via RUNPATH) |
| `cee` (hpws\*) | working | `~/LCM/lcm-build-serial-gcc-release/src/Albany` | `/projects/dakota/install/rhel8/6.24.0` (on disk) | none (serial build resolves Trilinos via RUNPATH) |

`sirius` is off-SRN (direct internet, Fedora): its bring-up is the same as
rigel's minus every proxy/CA step. See `docs/SETUP.md`.

Full environment bring-up (Miniforge, conda env, MatCal, Dakota, activate hook)
for every platform is documented in [`docs/SETUP.md`](docs/SETUP.md). MatCal is
installed via conda+pip everywhere; the CEE `matcal` module is not used (it is
pinned to the older rhel8 analyst stack; see `docs/SETUP.md`).

Selection: `$LCM_MATCAL_PLATFORM` → hostname match → local default (rigel).
Override just the executable with `$LCM_ALBANY`.

### Adding another platform

rigel, sirius and CEE are set up (see `docs/SETUP.md`). To add a new platform,
add a `Platform` entry in `site_matcal/platforms.py`:

1. **Albany** — set `albany` to the build/install path (or leave a name and rely
   on `$LCM_ALBANY` / PATH).
2. **Environment** — if the platform needs extra library paths at run time, add
   them to `env` (applied to the Albany subprocess via `Platform.apply_env`). If
   it needs `module load` commands, prefer a small wrapper script as the
   executable (`albany="/path/to/run_albany.sh"` that does the loads then
   `exec Albany "$@"`).
3. **Hostnames** — add substrings to the `hostnames` tuple for auto-detection
   (or force with `$LCM_MATCAL_PLATFORM`).
4. **HPC queue** — to submit models through a scheduler, call
   `model.run_in_queue(...)` and register the computing-platform factories; out
   of scope here.

Nothing else changes — the model, harness, and readers are platform-agnostic.

## Notes / gotchas

- **One Dakota study per Python process.** Dakota-as-a-library cannot run
  multiple studies in one interpreter (documented MatCal limitation; it
  segfaults on the second). Separate calibrations = separate `python` runs.
- **Do not put Dakota's `bin`/`lib` on `LD_LIBRARY_PATH`.** The Dakota CLI and
  bindings self-resolve via RPATH/RUNPATH; if Dakota's `bin` is on
  `LD_LIBRARY_PATH`, the Albany subprocess loads Dakota's bundled
  `libmpi.so.40` ahead of its own RUNPATH and segfaults at MPI finalize. The
  activate hook is set up accordingly.
- **Templates end with a blank line on purpose.** jinja2 strips one trailing
  newline; Albany's YAML parser fails at EOF without a final newline.
- **Identifiability depends on the load path.** The confined path is
  cap-dominated, so shear-surface parameters `A`/`C` are weakly constrained
  there (a fit can match the curve at the wrong `A`/`C`). Calibrate cap-active
  parameters (`R`, `W`, `D1`, `kappa0`) on confined/hydrostatic, and add the
  triaxial path to constrain the shear and non-associative terms.

## Verification status

All three platforms, rerun 2026-08-25 after the switch to base SI. The full
list below is the sirius run. rigel and cee (hpws00344) were rerun on the
forward runs and the two single-parameter studies, and return **identical**
values: the same `max|stress_zz|` on every load path to all printed digits, the
same converged parameters, and the same Dakota convergence messages. Timings
are the only thing that differs between machines.

sirius (Fedora 44, MatCal 1.4.28, Dakota 6.24.0):

- Forward runs on all three load paths (about 2.5 s each; `1.585615e+08`,
  `3.094229e+08` and `2.265800e+08` Pa peak `stress_zz` for confined,
  hydrostatic and triaxial).
- **Unit invariance.** Each load path rerun with the original MPa parameter set
  reproduces the Pa run to a maximum relative difference of 3e-15 in
  `stress_zz` (confined 2.4e-15, hydrostatic 2.9e-15, triaxial 2.4e-15), which
  is the roundoff floor. The model is unit-agnostic as documented: the drift
  tolerance is scaled by `E^2` in `CapModel_Def.hpp` and the Newton test acts on
  prescribed displacements, so nothing in the solve carries an absolute stress
  scale.
- `GradientCalibrationStudy`, one dimensionless parameter, one path: `R`
  recovered as `28.000000001` from an initial 22 (ABSOLUTE FUNCTION
  CONVERGENCE, ten Albany evaluations, 12 s).
- `GradientCalibrationStudy`, one stress-like parameter, one path:
  `kappa0 = -8050000.0` recovered exactly from an initial `-1.2e7` with bounds
  `[-2.0e7, -2.0e6]` (X-CONVERGENCE, ten evaluations, 13 s). This is the case
  the unit convention actually touches: Dakota normalizes each parameter onto
  [0, 1] over its bounds, so a stress-like parameter is searched no differently
  from a dimensionless one.
- `GradientCalibrationStudy`, two parameters, two paths: `R = 28.0`,
  `W = 0.080000000001` from (22, 0.05) (X-CONVERGENCE, 18 evaluations, 28 s on
  four cores).
- `ScipyMinimizeStudy` (`--study scipy`), one parameter: `R = 27.995`.

The gradient-study results are unchanged by the switch to Pa, as expected:
`CurveBasedInterpolatedObjective` conditions each field onto a fixed range
before differencing (MatCal's `RangeDataConditioner`), so the objective is
dimensionless and Dakota's convergence path does not carry the stress scale
either. The `--study scipy` value moves in the fourth digit only because that
study stops at a much looser tolerance.
