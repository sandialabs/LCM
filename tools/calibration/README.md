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
python calibrate.py calibrate \
    --load-path confined --load-path hydrostatic \
    --param R:20:35 --param W:0.02:0.15 \
    --study gradient --core-limit 4
```

- `--param NAME:LO:HI[:INIT]` — repeatable; `NAME` must match a jinja
  placeholder. `INIT` defaults to the Salem-limestone value.
- `--load-path` — repeatable; each becomes a MatCal evaluation set.
- `--data LOADPATH:CSV` — real experimental data (columns `time,stress_zz`;
  `time`∈[0,1] maps to applied strain for these decks). Defaults to
  `examples/<load_path>_reference.csv`.
- `--set NAME=VALUE` — override a fixed (non-calibrated) parameter default.
- `--platform rigel|sirius|cee` — force a platform (default: auto by hostname).
- `--study gradient|scipy` — Dakota gradient (default) or SciPy.

Calibratable placeholders: `A C R W D1 D2 kappa0 calpha N theta psi L phi Q D
elastic_modulus poissons_ratio`. Anything not calibrated keeps its
`SALEM_LIMESTONE` default (MatCal precedence: study params > model constants).

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

rigel:

- MatCal → Albany → Exodus forward runs on all three load paths.
- End-to-end Dakota `GradientCalibrationStudy` converges through the harness.

sirius (Fedora 44, MatCal 1.4.28, Dakota 6.24.0, 2026-08-25):

- Forward runs on all three load paths (about 3 s each).
- `GradientCalibrationStudy`, one parameter, one path: `R` recovered as
  `28.000000001` from an initial 22 (ABSOLUTE FUNCTION CONVERGENCE, ten Albany
  evaluations, 12 s).
- `GradientCalibrationStudy`, two parameters, two paths: `R = 28.0`,
  `W = 0.080000000001` from (22, 0.05) (X-CONVERGENCE, 18 evaluations, 28 s on
  four cores).
- `ScipyMinimizeStudy` (`--study scipy`), one parameter: `R = 27.9934`.
