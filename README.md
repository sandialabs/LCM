<img src="https://github.com/sandialabs/LCM/blob/main/wiki/lcm_logo.png" width="300">

# LCM

LCM (Laboratory for Computational Mechanics) is a fork of [Albany](https://github.com/sandialabs/Albany), an implicit, unstructured grid, finite element code for the solution and analysis of multiphysics problems. LCM focuses on computational solid mechanics and thermo-mechanics, with nearly 200 regression tests demonstrating capabilities across a wide variety of problems.

## Features

### Analysis of complex multiphysics problems

![Notched Cylinder Multi-Scale Simulation](https://github.com/sandialabs/LCM/blob/main/wiki/lcm_image_notched_cylinder.png)

Solid mechanics simulation of a notched cylinder using the [alternating Schwarz-based multi-scale coupling capability](https://onlinelibrary.wiley.com/doi/10.1002/nme.6982) in LCM.

### Capabilities

- Large-scale parallel simulations (2.1+ billion DOFs) using MPI
- Automatic differentiation via [Sacado](https://trilinos.github.io/sacado.html)
- Constitutive models: linear elasticity, J2 plasticity, crystal plasticity, hyperelasticity, damage models
- [Schwarz alternating method](https://onlinelibrary.wiley.com/doi/10.1002/nme.6982) for multi-scale coupling
- [Arctic Coastal Erosion (ACE) model](https://www.sciencedirect.com/science/article/pii/S0377042721001527): coupled thermo-mechanical model with permafrost constitutive models, part of the [InterFACE project](https://climatemodeling.science.energy.gov/projects/interface-interdisciplinary-research-arctic-coastal-environments)
- Data transfer between coupled domains via DataTransferKit (DTK), bundled in-tree

## Prerequisites

LCM requires [Trilinos](https://trilinos.org) and the following system packages.

### RHEL / Fedora

```bash
sudo dnf install \
  blas blas-devel boost boost-devel boost-openmpi boost-openmpi-devel \
  cmake gcc-c++ gcc-gfortran gtest-devel git \
  hdf5 hdf5-devel hdf5-openmpi hdf5-openmpi-devel \
  lapack lapack-devel netcdf netcdf-devel netcdf-openmpi netcdf-openmpi-devel \
  openblas openblas-devel openmpi openmpi-devel environment-modules
```

Optional (for clang builds):
```bash
sudo dnf install clang clang-devel
```

CMake 3.27 or newer (including 4.x) is required. If your system cmake is older:
```bash
spack install cmake@3.27
```

### Ubuntu

```bash
sudo apt install \
  libblas-dev libboost-dev libboost-program-options-dev \
  cmake g++ gfortran git \
  libhdf5-openmpi-dev liblapack-dev libnetcdf-dev \
  libopenmpi-dev mpi-default-bin environment-modules
```

### Sandia CEE LAN (hpws / cee-* machines)

No system packages need to be installed. The `lcm` build script automatically
detects CEE LAN hosts by hostname and loads all required compilers and TPLs
from the SEMS/AUE module system (`/projects/sems`).

**No `.bashrc` edits are needed.** The `lcm` script sources the SEMS module
init, loads AUE/SEMS modules (GCC 12.3, OpenMPI 4.1.6, CMake 3.31, Boost,
HDF5, NetCDF, binutils), and exports all necessary environment variables
internally. You only need `LCM_DIR` and the `lcm` symlink.

**Potential conflicts:** If your `.bashrc` already sources the SEMS init
script or loads AUE/sierra-devel modules, those may conflict with the versions
the `lcm` script loads. Either remove those lines from `.bashrc` or start
from a clean shell (`env -i bash --login`). In particular, pre-loaded
`sierra-devel` modules set compiler and MPI paths that will clash with the
AUE modules that `lcm` expects.

Currently only GCC builds are supported on CEE (module `serial-gcc-release`).

**If a configure fails with a nonsense compiler or TPL path** (for example
`CMAKE_CXX_COMPILER: /mpicxx is not a full path to an existing compiler tool`),
run `lcm env` first. It loads the module and prints the resolved compilers,
MPI and TPL paths without running anything else, so you can see which variable
is empty. The usual cause is a module loaded by your `.bashrc` that owns one of
the names the build uses (`CC`, `MPI_BIN`, `BOOST_INC`, ...); `lcm` now
restores its own values after loading the LCM module, and refuses to configure
if any of them is still missing.

## Quick Start

### 1. Clone repositories

```bash
mkdir ~/LCM && cd ~/LCM

git clone -b develop https://github.com/trilinos/Trilinos.git
git clone https://github.com/sandialabs/LCM.git
```

LCM builds against the `develop` branch of Trilinos, tracking its head
rather than any pinned commit.

### 2. Set up environment (optional, for interactive use only)

The `lcm` build script is self-contained: it auto-detects `LCM_DIR`, sets
`MODULEPATH`, sources the Environment Modules init script, and loads the
requested build module itself (default: `release`). **No `.bashrc` changes
are needed to build.**

Environment Modules setup is only needed for interactive use — that is,
running `module load` yourself so that `Albany` and the Trilinos tools
(`decomp`, `epu`, `exodiff`) are on your `PATH`. For that, add to
`~/.bashrc`:
```bash
export LCM_DIR=~/LCM
module use $LCM_DIR/LCM/doc/lcm/modulefiles
```

Log out and back in, or `source ~/.bashrc`.

### 3. Create the `lcm` symlink

```bash
cd ~/LCM
ln -s LCM/doc/lcm/build/lcm .
```

### 4. Build and test

```bash
cd ~/LCM

./lcm clean trilinos          # clean previous builds
./lcm config trilinos 16      # configure Trilinos (16 = parallel threads)
./lcm build trilinos 16       # build Trilinos

./lcm clean lcm               # clean previous LCM build
./lcm config lcm 16           # configure LCM
./lcm build lcm 16            # build LCM
./lcm test lcm                # run test suite
```

Or do everything at once:
```bash
./lcm all 16
```

### 5. Available commands

```
lcm clean   <package>        Clean build (and install for trilinos)
lcm config  <package> [N]    Configure package
lcm build   <package> [N]    Build package with N threads (default: nproc)
lcm test    <package>        Run tests
lcm all     [N]              Full pipeline: clean + config + build + test
lcm help                     Show help
```

## Module System

Modules configure the compiler, architecture, and build type. Available configurations:

| Module | Compiler | Build Type |
|--------|----------|------------|
| `serial-gcc-release` | GCC | Release (optimized) |
| `serial-gcc-debug` | GCC | Debug (symbols) |
| `serial-clang-release` | Clang | Release |
| `serial-clang-debug` | Clang | Debug |

The `release` module is an alias for `serial-gcc-release`.

Select a configuration for a build with the `--module` flag (the `lcm`
script loads the module itself):
```bash
./lcm all 16 --module=serial-clang-release
```

For interactive use (running `Albany` and Trilinos tools from `PATH`), load
the module in your shell instead:
```bash
module load serial-clang-release
```

Build directories are named by configuration, e.g. `lcm-build-serial-gcc-release`.
A corresponding Trilinos installation must exist for each configuration.

## Running

After building, the Albany executable is in the build directory:
```bash
cd ~/LCM/lcm-build-serial-gcc-release/tests/LCM/Pressure
mpiexec -np 4 ~/LCM/lcm-build-serial-gcc-release/src/Albany input_tetra4.yaml
```

The module system adds build and Trilinos install directories to `PATH`, so after loading a module, `Albany` and Trilinos tools (`decomp`, `epu`, `exodiff`) are available directly.

## Testing

The test suite is in `tests/` and runs via CTest:
```bash
./lcm test lcm
```

Individual tests can be run from their subdirectory:
```bash
cd ~/LCM/lcm-build-serial-gcc-release/tests/LCM/Pressure
ctest
```

Many tests run in parallel using up to 4 MPI ranks.

## Nightly Tests

The `clone-build-test-dash.sh` script clones Trilinos and LCM from scratch,
builds with each supported compiler, and submits results to CDash. It lives
at `LCM/doc/lcm/test/clone-build-test-dash.sh` but must be run from a copy
in the workspace root: it sets `LCM_DIR` to the current directory and
deletes and re-clones the `LCM` repository, so it cannot run from inside
the repository itself. Copy it into place once:

```bash
cp LCM/doc/lcm/test/clone-build-test-dash.sh ~/LCM/
```

It is typically run via cron:

```bash
# In crontab:
00 00 * * 1-5 cd /home/lcm/LCM; bash -l -c "./clone-build-test-dash.sh"
```

## Contributing

Development discussion: https://github.com/sandialabs/LCM/issues

We follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). Use `clang-format` with the `.clang-format` file in `src/LCM/`:
```bash
clang-format -i <source file>
```

Please ensure all tests pass before pushing changes.

## Contact

- Alejandro Mota (amota@sandia.gov)
- Irina Tezaur (ikalash@sandia.gov)
