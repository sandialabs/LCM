# Environment setup for LCM cap calibration (MatCal + Dakota)

This documents the **exact, reproducible route** to stand up the calibration
environment on each platform. The harness code (`tools/calibration`) is
platform-agnostic; only the environment bring-up differs.

The approach on **every** platform is the same:

- **MatCal**: installed from the public repo (github.com/sandialabs/matcal,
  v1.4.27) into a dedicated **conda env on Python 3.12**, via `pip install -r
  matcal/requirements.txt`. (Python 3.12 — not 3.11 — because Dakota's prebuilt
  Python bindings are compiled `cpython-312`.)
- **Dakota**: a **6.24.0** build whose `dakota.environment` bindings are
  `cpython-312` (matches the env). Reused from disk where available, downloaded
  otherwise.
- A conda **activate hook** wires the CA bundle, Dakota, and this repo onto the
  environment automatically.

We deliberately **do not** use the CEE `matcal` module: `matcal/stable` is
version 1.3.6 and pinned to the CEE **rhel8** analyst stack (`dakota/6.20.0-aue`,
`aue/anaconda3/2023.09`, `seacas`), which does not load on a rhel9 workstation
and is an older API than the harness targets. conda+pip keeps both platforms on
the same MatCal version and identical harness code.

---

## Common prerequisites

- **Network.** On the SRN platforms (rigel, cee) outbound HTTPS goes through the
  Sandia proxy `proxy.sandia.gov:80` (already in the environment as
  `http_proxy`/`https_proxy`); those machines sit behind a TLS-inspecting proxy,
  so conda/pip must be pointed at the **system CA bundle**
  `/etc/pki/tls/certs/ca-bundle.crt` (curl/git already trust it). That is why
  the steps below set `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` and
  `conda config --set ssl_verify`. Off-SRN workstations (sirius) reach GitHub
  and PyPI directly: **skip every CA/proxy step**, and in particular do not
  export `SSL_CERT_FILE`. On Fedora the RHEL path
  `/etc/pki/tls/certs/ca-bundle.crt` does not exist (the bundle lives at
  `/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem`), and pointing
  `SSL_CERT_FILE` at a missing file breaks every TLS call made by pip and
  `requests`. The steps below are annotated "(SRN only)" where they apply.
- The LCM Albany serial binary must exist to run calibrations
  (`~/LCM/lcm-build-serial-gcc-release/src/Albany`); building it (Trilinos +
  LCM) is a separate, long step, see "Building Albany" at the end.

---

## 1. Miniforge (all platforms)

```bash
# (SRN only) trust the inspecting proxy's CA
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
cd /tmp
curl -fsSL -o miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash miniforge.sh -b -p "$HOME/miniforge3"
# (SRN only)
"$HOME/miniforge3/bin/conda" config --set ssl_verify /etc/pki/tls/certs/ca-bundle.crt
```

(On SRN add `--cacert /etc/pki/tls/certs/ca-bundle.crt` to the `curl` call.)

## 2. The `matcal` conda env (all platforms)

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda create -y -n matcal python=3.12
conda activate matcal
```

## 3. MatCal (all platforms)

```bash
# (SRN only) export SSL_CERT_FILE=... REQUESTS_CA_BUNDLE=... and add
#            --cert /etc/pki/tls/certs/ca-bundle.crt to pip
mkdir -p ~/Repos && cd ~/Repos
git clone https://github.com/sandialabs/matcal.git      # v1.4.27+ (1.4.28 tested)
pip install -r matcal/matcal/requirements.txt
# put the matcal package on the env's path (conda develop was removed in conda 26):
SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$HOME/Repos/matcal" > "$SP/matcal.pth"
```

Install the requirements **before** importing anything from `matcal`: the
package's `__init__` pulls in numpy at import time, so even a version query
fails on a bare env.

## 4. Dakota

Dakota 6.24.0 with `cpython-312` bindings, plus its `share/dakota/Python`
(the `dakota.environment` module MatCal imports).

- **CEE (hpws\*)** — already on disk, no download:
  `DAKOTA_ROOT=/projects/dakota/install/rhel8/6.24.0`
  (verified: `share/dakota/Python/dakota/environment/environment.cpython-312-*.so`).
- **rigel, sirius (local workstations)** - downloaded once to `~/dakota/6.24.0`:
  ```bash
  cd /tmp
  curl -fsSL -o dakota.tgz https://github.com/snl-dakota/dakota/releases/download/v6.24.0/dakota-6.24.0-public-rhel8.Linux.x86_64-cli.tar.gz
  mkdir -p ~/dakota && tar xzf dakota.tgz -C ~/dakota
  mv ~/dakota/dakota-6.24.0-public-rhel8.Linux.x86_64-cli ~/dakota/6.24.0
  ```
  The rhel8 tarball runs unmodified on Fedora (verified on sirius, Fedora 44):
  both the CLI and the `cpython-312` bindings load without any compatibility
  shims.

## 5. Activate hook (per platform)

Create `~/miniforge3/envs/matcal/etc/conda/activate.d/zz_matcal_dakota.sh` so a
plain `conda activate matcal` wires everything. **Only `DAKOTA_ROOT` and
`LCM_MATCAL_PLATFORM` differ between platforms.**

```sh
#!/bin/sh
# CA bundle (SRN only: TLS-inspecting proxy proxy.sandia.gov).
# Omit these two lines off-SRN; the file does not exist on Fedora and an
# SSL_CERT_FILE pointing at a missing file breaks pip and requests.
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

# --- Dakota ---
# NOTE: do NOT add Dakota's bin/lib to LD_LIBRARY_PATH. The CLI ($ORIGIN RPATH)
# and the python bindings (relative RUNPATH) self-resolve; if Dakota's bin is on
# LD_LIBRARY_PATH the Albany subprocess loads Dakota's bundled libmpi.so.40
# ahead of its own RUNPATH and segfaults at MPI finalize.
export DAKOTA_ROOT=/projects/dakota/install/rhel8/6.24.0   # CEE
export PATH="$DAKOTA_ROOT/bin:$PATH"
# Default PYTHONPATH to empty: it is usually unset, and an unguarded $PYTHONPATH
# aborts the hook (and so the whole `conda activate`) under `set -u`.
export PYTHONPATH="$DAKOTA_ROOT/share/dakota/Python:${PYTHONPATH:-}"

# --- site_matcal harness + platform selection ---
export PYTHONPATH="${LCM_DIR:-$HOME/LCM}/LCM/tools/calibration:${PYTHONPATH:-}"
export LCM_MATCAL_PLATFORM=cee
```

Per-platform values of the two variables that differ:

| Platform | `DAKOTA_ROOT` | `LCM_MATCAL_PLATFORM` | CA lines |
|----------|---------------|-----------------------|----------|
| `rigel`  | `$HOME/dakota/6.24.0` | `rigel` (or omit: auto) | keep |
| `sirius` | `$HOME/dakota/6.24.0` | `sirius` (or omit: auto) | drop |
| `cee`    | `/projects/dakota/install/rhel8/6.24.0` | `cee` | keep |

Auto-detection is by hostname, so `LCM_MATCAL_PLATFORM` is optional on any
machine whose hostname is registered in `site_matcal/platforms.py`.

## 6. Verify (no Albany needed)

```bash
conda activate matcal
python -c "import matcal, site_matcal; print(site_matcal.get_platform())"   # -> Platform(name='cee'...)
python -c "import dakota.environment; print('dakota bindings OK')"
dakota -version
```

Full end-to-end verification (once Albany is built):

```bash
cd "$LCM_DIR/LCM/tools/calibration/harness"
python calibrate.py make-reference --load-path confined
python calibrate.py calibrate --load-path confined --param R:20:35:22
```

The second command must recover the parameter the reference was generated at
(`BEST: R: 28.000000001`, ABSOLUTE FUNCTION CONVERGENCE; about 12 s and ten
Albany evaluations on a workstation). A two-parameter, two-path round trip is
the next check:

```bash
python calibrate.py make-reference --load-path confined --load-path hydrostatic
python calibrate.py calibrate --load-path confined --load-path hydrostatic \
    --param R:20:35:22 --param W:0.02:0.15:0.05 --core-limit 4
```

which returns `R: 28.0`, `W: 0.080000000001` (X-CONVERGENCE, 18 evaluations,
about 28 s). Both of those parameters are dimensionless; add one stress-like
parameter to check that base-SI bounds are being entered correctly (the harness
is in Pa, see the README's "Units" section):

```bash
python calibrate.py calibrate --load-path hydrostatic \
    --param kappa0:-2.0e7:-2.0e6:-1.2e7
```

which returns `kappa0: -8050000.0` (X-CONVERGENCE, ten evaluations, about 13
s). A result stuck at a bound, or off by a clean factor of 1e6, means the
bounds or the experimental data were given in MPa.

---

## Building Albany (all platforms, required to run calibrations)

The calibration drives `~/LCM/lcm-build-serial-gcc-release/src/Albany`. On a
fresh checkout, build Trilinos then LCM (long):

```bash
cd ~/LCM
./lcm config trilinos && ./lcm build trilinos 32
./lcm config lcm       && ./lcm build lcm 32
```

(Always build via `./lcm`, never bare cmake.) The serial build resolves its
Trilinos libraries via a baked-in RUNPATH, so no module load is needed to run
Albany from MatCal. If a given CEE build turns out to need runtime module
libraries, add them to the `cee` entry's `env` in `site_matcal/platforms.py`.

---

## Gotchas (all platforms)

- **Base SI everywhere** - the harness is in Pa, with magnitudes in scientific
  notation and no prefixed units (`2.2547e10`, not `22547` MPa). `--param`
  bounds, `--set` overrides and the stress column of a `--data` file must match.
  A curve supplied in MPa is not detected; it is simply fit by stress-like
  parameters 1e6 too small. See the README's "Units" section.
- **One Dakota study per Python process** — Dakota-as-a-library cannot run
  multiple studies in one interpreter (segfaults on the second). Separate
  calibrations = separate `python` invocations.
- **Keep Dakota off `LD_LIBRARY_PATH`** — see the hook note above.
- **Templates end with a trailing blank line** — jinja2 strips one trailing
  newline and Albany's YAML parser fails at EOF without it.
- **Proxy CA (SRN only)** - conda/pip fail with `CERTIFICATE_VERIFY_FAILED`
  unless pointed at `/etc/pki/tls/certs/ca-bundle.crt`. Off-SRN, leave
  `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` unset: that RHEL path does not exist on
  Fedora, and a stale value breaks TLS for pip and `requests`.
- **New machine** - a host whose name is not in any `Platform.hostnames` falls
  back to the rigel entry. That happens to be right when the Albany build sits
  at the standard path, but add an entry (one four-line block in
  `site_matcal/platforms.py`) so `get_platform()` reports the truth and
  `$LCM_MATCAL_PLATFORM=<host>` works in the activate hook.
