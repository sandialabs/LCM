# Setting up the LCM cap-plasticity calibration

This walks you through installing everything needed to calibrate the LCM cap
plasticity model, from a machine where nothing is installed yet, to a finished
calibration that recovers a known answer.

It is written for a **CEE workstation** (`hpws*`), which is where these
calibrations are meant to run, and assumes **no prior command-line
experience**. Follow the steps in order. Other platforms are covered in
[Appendix A](#appendix-a-other-platforms).

Once you are set up, [`README.md`](README.md) explains how to calibrate
against your own data.

---

## How to read this document

- Every grey box is a list of commands, **one command per line**. Copy and run
  them one line at a time, waiting for the prompt to come back between them.
  Nothing here needs to be typed on more than one line.
- `~` means your home directory (`/ascldap/users/<your username>` on CEE).
  Leave it as `~`; the shell expands it for you.
- Lines beginning with `#` inside a box are comments. They do nothing, and you
  can paste them along with the commands.
- Most successful commands print little or nothing. **Silence usually means
  success.** After every stage there is a **Check it worked** box: run it, and
  compare against the output shown. Do not move on until it matches.
- If something fails, look in [Troubleshooting](#troubleshooting) before
  retrying.

---

## Dependencies

Four things get installed. Steps 1 to 5 install them in this order.

| What | Why it is needed | Where it comes from | Step |
|------|------------------|---------------------|------|
| **LCM** (with the Albany executable) | Runs the actual cap-plasticity simulation that gets fitted | Built from source; separate instructions | [1](#step-1-lcm-and-albany) |
| **Miniforge** | Provides `conda`, which isolates the Python environment | Downloaded installer | [2](#step-2-miniforge) |
| **MatCal** | Drives the calibration: renders inputs, runs Albany, builds the objective | **Cloned from GitHub** (not the CEE module) | [4](#step-4-matcal) |
| **Dakota** | The optimizer MatCal calls to search parameter space | On-disk CEE install (no download needed) | [5](#step-5-dakota) |

Two rules that the rest of this document depends on:

> **Clone MatCal from its repository. Do not use the CEE `matcal` module.**
> See [Step 4](#step-4-matcal) for why, and what goes wrong if you do.

> **Do not put Dakota's `bin` or `lib` on `LD_LIBRARY_PATH`.** See
> [Step 5](#step-5-dakota).

---

## Step 1: LCM and Albany

The calibration works by running the LCM `Albany` executable a few dozen
times, so that executable has to exist before anything else is useful.

Building it (Trilinos, then LCM) is a long job and is documented separately,
in the LCM repository's own README:

- In this checkout: [`../../README.md`](../../README.md)
- On the web: <https://github.com/sandialabs/LCM#quick-start>

Follow it as far as building LCM. The short version, once Trilinos and LCM are
cloned into `~/LCM` and the `lcm` symlink exists, is:

```bash
cd ~/LCM
./lcm config trilinos 32 && ./lcm build trilinos 32
./lcm config lcm 32 && ./lcm build lcm 32
```

Always build through `./lcm`, never by calling `cmake` yourself. The `lcm`
script loads the right CEE compilers and libraries for you, so no `.bashrc`
changes are needed.

The serial build bakes the paths to its Trilinos libraries into the
executable, so you do **not** need to load any modules to run `Albany` later.
That is what lets the calibration launch it as a plain subprocess.

**Check it worked.** This must print a path and a file size:

```bash
ls -la ~/LCM/lcm-build-serial-gcc-release/src/Albany
```

If it says "No such file or directory", the build has not finished. Go back to
the LCM README; nothing below will work until this file exists.

---

## Step 2: Miniforge

Miniforge gives you the `conda` command, which keeps this project's Python
packages separate from the system Python and from every other project.

```bash
cd /tmp
curl -fsSL --cacert /etc/pki/tls/certs/ca-bundle.crt -o miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash miniforge.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" config --set ssl_verify /etc/pki/tls/certs/ca-bundle.crt
```

The `--cacert` and `ssl_verify` bits are needed **on the SRN only**. CEE sits
behind a proxy that inspects encrypted traffic, so `curl`, `conda` and `pip`
have to be told to trust the Sandia certificate at
`/etc/pki/tls/certs/ca-bundle.crt`. Without them you get
`CERTIFICATE_VERIFY_FAILED`.

**Check it worked.** This must print a version number such as `conda 25.x.x`:

```bash
~/miniforge3/bin/conda --version
```

---

## Step 3: The `matcal` conda environment

Everything from here on lives in one conda environment named `matcal`.

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda create -y -n matcal python=3.12
conda activate matcal
```

**Python 3.12 exactly.** Not 3.11, not 3.13. Dakota ships pre-compiled Python
bindings built for 3.12 (`cpython-312`), and they will not import into any
other version.

**Check it worked.** Your prompt should now start with `(matcal)`, and this
must print `3.12` followed by a third number:

```bash
python --version
```

From now on, **every session** in which you want to calibrate must start with
these two lines:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate matcal
```

---

## Step 4: MatCal

> **Install MatCal by cloning it. Do not `module load matcal`.**

The CEE `matcal` module is not an option on a modern CEE workstation, and
trying to use it wastes time in confusing ways:

- On a RHEL 9 workstation (which `hpws*` machines now are) the module tree it
  belongs to does not load at all. `module load sems-env` and
  `module load dakota-env` both fail with
  `ERROR: Unrecognized cluster linux_rh9`, and `module avail` lists no
  `matcal` module whatsoever.
- Where it does load, `matcal/stable` is version 1.3.6, pinned to the older
  RHEL 8 analyst stack (`dakota/6.20.0-aue`, `aue/anaconda3/2023.09`,
  `seacas`). That is an older API than this harness targets.

Cloning keeps every platform on the same MatCal version running identical
harness code.

```bash
mkdir -p ~/Repos
cd ~/Repos
git clone https://github.com/sandialabs/matcal.git
pip install --cert /etc/pki/tls/certs/ca-bundle.crt -r ~/Repos/matcal/matcal/requirements.txt
```

(Off the SRN, drop the `--cert` option.)

Now put the cloned package where Python can find it. `conda develop` was
removed in conda 26, so write the path file directly:

```bash
SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$HOME/Repos/matcal" > "$SP/matcal.pth"
```

Install the requirements **before** importing anything from MatCal: its
`__init__` imports numpy at import time, so even asking for the version fails
on an empty environment.

**Check it worked.** This must print a version of 1.4.27 or newer:

```bash
python -c "import matcal; print(matcal.__version__)"
```

---

## Step 5: Dakota

Dakota is the optimizer. MatCal calls it as a library, so you need both the
Dakota programs and its Python bindings, and the bindings must be the
`cpython-312` ones.

**On CEE, Dakota is already on disk. Nothing to download or build:**

```bash
ls /projects/dakota/install/rhel8/6.24.0
```

That directory is used directly; Step 6 points the environment at it. Despite
the `rhel8` in the path, it runs correctly on the RHEL 9 workstations.

> **Never add Dakota's `bin` or `lib` to `LD_LIBRARY_PATH`.** Dakota's programs
> and bindings find their own libraries. If Dakota's `bin` is on
> `LD_LIBRARY_PATH`, the Albany subprocess picks up Dakota's bundled
> `libmpi.so.40` instead of its own and crashes at the end of every run. The
> hook in Step 6 is written to avoid this, and Step 7 checks for it.

**Check it worked.** This must list a file containing `cpython-312`:

```bash
ls /projects/dakota/install/rhel8/6.24.0/share/dakota/Python/dakota/environment/
```

---

## Step 6: The activate hook

Rather than remember four `export` lines every session, put them in a script
that conda runs automatically whenever you activate the environment.

Create the file:

```bash
mkdir -p ~/miniforge3/envs/matcal/etc/conda/activate.d
```

Then write this into
`~/miniforge3/envs/matcal/etc/conda/activate.d/zz_matcal_dakota.sh`. Use any
editor you like; if you have no preference, `nano <filename>` is the simplest
(save with Ctrl-O, exit with Ctrl-X).

```sh
#!/bin/sh
# CA bundle for the SRN's TLS-inspecting proxy. Keep these two lines on CEE
# and rigel; delete them off the SRN, where this file does not exist and
# pointing at a missing file breaks pip.
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

# --- Dakota ---
# NOTE: do NOT add Dakota's bin/lib to LD_LIBRARY_PATH. The programs and the
# python bindings find their own libraries; if Dakota's bin is on
# LD_LIBRARY_PATH the Albany subprocess loads Dakota's bundled libmpi.so.40
# ahead of its own and segfaults at MPI finalize.
export DAKOTA_ROOT=/projects/dakota/install/rhel8/6.24.0
export PATH="$DAKOTA_ROOT/bin:$PATH"
# Default PYTHONPATH to empty: it is usually unset, and an unguarded
# $PYTHONPATH aborts the hook (and the whole `conda activate`) under `set -u`.
export PYTHONPATH="$DAKOTA_ROOT/share/dakota/Python:${PYTHONPATH:-}"

# --- calibration harness + platform selection ---
export PYTHONPATH="${LCM_DIR:-$HOME/LCM}/LCM/tools/calibration:${PYTHONPATH:-}"
export LCM_MATCAL_PLATFORM=cee
```

The last two lines are what make `import site_matcal` work and tell the
harness which machine it is on.

Make the hook take effect by leaving and re-entering the environment:

```bash
conda deactivate
conda activate matcal
```

**Check it worked.** This must print the Dakota directory from Step 5:

```bash
echo $DAKOTA_ROOT
```

---

## Step 7: Check the environment

The harness can inspect its own environment and tell you what is missing. This
runs no simulations and takes about a second.

```bash
cd ~/LCM/LCM/tools/calibration/harness
python calibrate.py check
```

**Check it worked.** Every line must say `PASS`, and the last line must say
`all checks passed`:

```
environment check
  [PASS] python: 3.12.14 (Dakota's bindings are cpython-312)
  [PASS] matcal: 1.4.27
  [PASS] dakota bindings: import dakota.environment OK
  [PASS] dakota CLI: /projects/dakota/install/rhel8/6.24.0/bin/dakota
  [PASS] LD_LIBRARY_PATH clean of Dakota: yes
  [PASS] platform: Platform(name='cee', albany='/ascldap/users/<you>/LCM/lcm-build-serial-gcc-release/src/Albany')
  [PASS] Albany: /ascldap/users/<you>/LCM/lcm-build-serial-gcc-release/src/Albany
  [PASS] templates: /ascldap/users/<you>/LCM/LCM/tools/calibration/templates

all checks passed (8/8)
```

`<you>` is your username, and the exact home-directory prefix may differ. Your
MatCal and Python patch versions may be newer than those shown; what matters is
that every line says `PASS`.

Any `FAIL` line names the thing that is wrong. Match it against
[Troubleshooting](#troubleshooting), fix it, and run `check` again.

---

## Step 8: Run your first calibration

This is the end-to-end test: manufacture a stress-strain curve from known
parameters, then throw one of those parameters away and let the optimizer find
it again. It takes about a minute.

First, generate the curve. This runs Albany once:

```bash
python calibrate.py make-reference --load-path confined
```

**Check it worked.** The last line must read (the numbers matter):

```
[confined] wrote reference .../examples/confined_reference.csv (201 points, true-stress-strain, finite deformation, peak |stress_xx| = 3.270823e+08)
```

Two defaults are being applied here, and both are named in that line:
`true-stress-strain` is the curve being compared, and `finite deformation` is
the kinematics. Every run states them, so you never have to guess which was
used. Adding `--small-strain` selects the infinitesimal-strain kernel instead
and gives `3.106368e+08`.

The README explains both: "Kinematics" for why finite deformation is the
default and when it matters, and "Changing the defaults" for how to override
either one for a single run or change it permanently (one constant each).

Now recover `R`, one of the cap-shape parameters. The curve was made at
`R = 28`; `20:35:22` tells the optimizer to search between 20 and 35 **and to
start from 22**, so that it has something real to find:

```bash
python calibrate.py calibrate --load-path confined --param R:20:35:22
```

**Check it worked.** After a page of MatCal output, the last lines must read:

```
BEST: R: 28.0

***** X-CONVERGENCE *****
```

That is a complete, verified installation: `28.0` is the value the reference
was generated at, recovered from a starting guess of 22.

You are done. Read [`README.md`](README.md) next to calibrate against your own
data. Start with its "Kinematics" section, which explains the finite-deformation
default and, in particular, why the Salem limestone numbers this test recovers
are a starting point rather than an answer.

---

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `CERTIFICATE_VERIFY_FAILED` from `conda` or `pip` | The SRN proxy's certificate is not trusted. Add `--cert /etc/pki/tls/certs/ca-bundle.crt` to `pip`, and rerun the `conda config --set ssl_verify` line from Step 2. |
| `ModuleNotFoundError: No module named 'matcal'` | The `.pth` file in Step 4 was not written, or was written into a different environment. Re-run Step 4's last two lines with `(matcal)` showing in your prompt. |
| `ModuleNotFoundError: No module named 'dakota'` | `DAKOTA_ROOT` is unset or not on `PYTHONPATH`. Check Step 6, then `conda deactivate && conda activate matcal`. |
| `check` says `FAIL` on Albany | The path in the message does not exist. Finish Step 1, or point at a build elsewhere with `export LCM_ALBANY=/path/to/Albany`. |
| `check` says `FAIL` on `LD_LIBRARY_PATH` | Something (often a `.bashrc` line) put Dakota on `LD_LIBRARY_PATH`. Remove it; see Step 5. |
| Albany crashes at the end of every evaluation | Same cause as above: Dakota's `libmpi` is being loaded instead of Albany's. |
| `ERROR: Unrecognized cluster linux_rh9` | You are trying to `module load` the SEMS/Dakota module tree on a RHEL 9 workstation. Do not; this setup does not use modules at all. |
| `conda activate` itself fails with `PYTHONPATH: unbound variable` | An older hook that used `$PYTHONPATH` unguarded. Use `${PYTHONPATH:-}` as shown in Step 6. |
| Everything passes, but a calibration ends at its starting value | Usually the experimental data is not in base SI, or is compression-positive. The harness warns about both; see the README's "Units" section. |
| A column such as `strain_xx` is reported missing | The finite-deformation kernel writes no strain field. Use `--curve true-stress-strain` or `eng-stress-strain`, whose strain is reconstructed from displacement, or run `--small-strain`. |
| The second calibration in one script crashes | Only one calibration may run per `python` process. This is a Dakota-as-a-library limitation. Use a separate `python` command for each. |

---

## Appendix A: Other platforms

The harness itself is platform-agnostic; only Steps 5 and 6 differ.

| Platform | Dakota | `LCM_MATCAL_PLATFORM` | Proxy/CA lines |
|----------|--------|-----------------------|----------------|
| `cee` (`hpws*`) | `/projects/dakota/install/rhel8/6.24.0` (on disk) | `cee` | keep |
| `rigel` | `$HOME/dakota/6.24.0` (download) | `rigel`, or omit | keep |
| `sirius` | `$HOME/dakota/6.24.0` (download) | `sirius`, or omit | **delete** |

`sirius` is off the SRN, on Fedora. Skip every proxy and certificate step: the
path `/etc/pki/tls/certs/ca-bundle.crt` does not exist there (Fedora keeps its
bundle at `/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem`), and pointing
`SSL_CERT_FILE` at a missing file breaks every encrypted connection `pip` and
`requests` make.

On `rigel` and `sirius`, Dakota is not on disk and must be downloaded once:

```bash
cd /tmp
curl -fsSL -o dakota.tgz https://github.com/snl-dakota/dakota/releases/download/v6.24.0/dakota-6.24.0-public-rhel8.Linux.x86_64-cli.tar.gz
mkdir -p ~/dakota && tar xzf dakota.tgz -C ~/dakota
mv ~/dakota/dakota-6.24.0-public-rhel8.Linux.x86_64-cli ~/dakota/6.24.0
```

The RHEL 8 tarball runs unmodified on Fedora 44 (verified on sirius): the
programs and the `cpython-312` bindings both load with no compatibility shims.

`LCM_MATCAL_PLATFORM` is optional on any machine whose hostname is registered
in `site_matcal/platforms.py`; auto-detection is by hostname. A machine that
matches nothing falls back to the `rigel` entry, which happens to be correct
whenever the Albany build sits at the standard path. Add an entry anyway (a
four-line block) so `check` reports the truth.

---

## Appendix B: What the activate hook actually does

Four things, and nothing else:

1. Points `pip` and `requests` at the Sandia CA bundle, so they work through
   the inspecting proxy (SRN only).
2. Sets `DAKOTA_ROOT` and puts Dakota's programs on `PATH`, **without**
   touching `LD_LIBRARY_PATH`.
3. Puts Dakota's Python bindings on `PYTHONPATH`.
4. Puts `tools/calibration` on `PYTHONPATH` and names the platform, which is
   what makes `import site_matcal` succeed.

It deliberately loads no modules. The serial Albany build resolves its own
Trilinos libraries, MatCal and Dakota come from the conda environment and from
`/projects`, and keeping the shell clean is what makes the same harness code
run identically on CEE, rigel and sirius.
