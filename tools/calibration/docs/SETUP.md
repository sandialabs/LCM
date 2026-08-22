# Environment setup for LCM cap calibration (MatCal + Dakota)

This documents the **exact, reproducible route** to stand up the calibration
environment on each platform. The harness code (`tools/calibration`) is
platform-agnostic; only the environment bring-up differs.

The approach on **both** platforms is the same:

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

- Outbound HTTPS works through the Sandia proxy `proxy.sandia.gov:80` (already
  in the environment as `http_proxy`/`https_proxy`).
- The machine is behind a TLS-inspecting proxy, so conda/pip must trust the
  **system CA bundle** `/etc/pki/tls/certs/ca-bundle.crt` (curl/git already do).
  This is why every step below sets `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` and
  `conda config --set ssl_verify` to that bundle.
- The LCM Albany serial binary must exist to run calibrations
  (`~/LCM/lcm-build-serial-gcc-release/src/Albany`); building it (Trilinos +
  LCM) is a separate, long step — see "Building Albany" at the end.

---

## 1. Miniforge (both platforms)

```bash
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
cd /tmp
curl -fsSL --cacert /etc/pki/tls/certs/ca-bundle.crt -o miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash miniforge.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" config --set ssl_verify /etc/pki/tls/certs/ca-bundle.crt
```

## 2. The `matcal` conda env (both platforms)

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda create -y -n matcal python=3.12
conda activate matcal
```

## 3. MatCal (both platforms)

```bash
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
mkdir -p ~/Repos && cd ~/Repos
git clone https://github.com/sandialabs/matcal.git      # v1.4.27+
pip install --cert /etc/pki/tls/certs/ca-bundle.crt -r matcal/matcal/requirements.txt
# put the matcal package on the env's path (conda develop was removed in conda 26):
SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$HOME/Repos/matcal" > "$SP/matcal.pth"
```

## 4. Dakota

Dakota 6.24.0 with `cpython-312` bindings, plus its `share/dakota/Python`
(the `dakota.environment` module MatCal imports).

- **CEE (hpws\*)** — already on disk, no download:
  `DAKOTA_ROOT=/projects/dakota/install/rhel8/6.24.0`
  (verified: `share/dakota/Python/dakota/environment/environment.cpython-312-*.so`).
- **rigel (local)** — downloaded once to `~/dakota/6.24.0`:
  ```bash
  cd /tmp
  curl -fsSL -o dakota.tgz https://github.com/snl-dakota/dakota/releases/download/v6.24.0/dakota-6.24.0-public-rhel8.Linux.x86_64-cli.tar.gz
  mkdir -p ~/dakota && tar xzf dakota.tgz -C ~/dakota
  mv ~/dakota/dakota-6.24.0-public-rhel8.Linux.x86_64-cli ~/dakota/6.24.0
  ```

## 5. Activate hook (per platform)

Create `~/miniforge3/envs/matcal/etc/conda/activate.d/zz_matcal_dakota.sh` so a
plain `conda activate matcal` wires everything. **Only `DAKOTA_ROOT` and
`LCM_MATCAL_PLATFORM` differ between platforms.**

```sh
#!/bin/sh
# CA bundle (TLS-inspecting proxy proxy.sandia.gov)
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

# --- Dakota ---
# NOTE: do NOT add Dakota's bin/lib to LD_LIBRARY_PATH. The CLI ($ORIGIN RPATH)
# and the python bindings (relative RUNPATH) self-resolve; if Dakota's bin is on
# LD_LIBRARY_PATH the Albany subprocess loads Dakota's bundled libmpi.so.40
# ahead of its own RUNPATH and segfaults at MPI finalize.
export DAKOTA_ROOT=/projects/dakota/install/rhel8/6.24.0   # CEE  (rigel: $HOME/dakota/6.24.0)
export PATH="$DAKOTA_ROOT/bin:$PATH"
export PYTHONPATH="$DAKOTA_ROOT/share/dakota/Python:$PYTHONPATH"

# --- site_matcal harness + platform selection ---
export PYTHONPATH="${LCM_DIR:-$HOME/LCM}/LCM/tools/calibration:$PYTHONPATH"
export LCM_MATCAL_PLATFORM=cee              # rigel: omit (auto-detected) or set to rigel
```

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

---

## Building Albany (both platforms, required to run calibrations)

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

## Gotchas (both platforms)

- **One Dakota study per Python process** — Dakota-as-a-library cannot run
  multiple studies in one interpreter (segfaults on the second). Separate
  calibrations = separate `python` invocations.
- **Keep Dakota off `LD_LIBRARY_PATH`** — see the hook note above.
- **Templates end with a trailing blank line** — jinja2 strips one trailing
  newline and Albany's YAML parser fails at EOF without it.
- **Proxy CA** — conda/pip fail with `CERTIFICATE_VERIFY_FAILED` unless pointed
  at `/etc/pki/tls/certs/ca-bundle.crt`.
