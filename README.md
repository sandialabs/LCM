<img src="https://github.com/sandialabs/LCM/blob/main/wiki/albany_lcm.png" width="300">

# Albany-LCM

Albany-LCM (Laboratory for Computational Mechanics) is a fork of [Albany](https://github.com/sandialabs/Albany), an implicit, unstructured grid, finite element code for the solution and analysis of multiphysics problems. Albany-LCM focuses on computational solid mechanics and thermo-mechanics, with nearly 200 regression tests demonstrating capabilities across a wide variety of problems.

## Features

### Analysis of complex multiphysics problems

![Notched Cylinder Multi-Scale Simulation](https://github.com/sandialabs/LCM/blob/main/wiki/lcm_image_notched_cylinder.png)

Solid mechanics simulation of a notched cylinder using the [alternating Schwarz-based multi-scale coupling capability](https://onlinelibrary.wiley.com/doi/10.1002/nme.6982) in Albany-LCM.

### Capabilities

- Large-scale parallel simulations (2.1+ billion DOFs) using MPI
- Automatic differentiation via [Sacado](https://trilinos.github.io/sacado.html)
- Constitutive models: linear elasticity, J2 plasticity, crystal plasticity, hyperelasticity, damage models
- [Schwarz alternating method](https://onlinelibrary.wiley.com/doi/10.1002/nme.6982) for multi-scale coupling
- [Arctic Coastal Erosion (ACE) model](https://www.sciencedirect.com/science/article/pii/S0377042721001527): coupled thermo-mechanical model with permafrost constitutive models, part of the [InterFACE project](https://climatemodeling.science.energy.gov/projects/interface-interdisciplinary-research-arctic-coastal-environments)
- Data transfer between coupled domains via [DataTransferKit (DTK)](https://github.com/ikalash/DataTransferKit)

## Prerequisites

Albany-LCM requires [Trilinos](https://trilinos.org) and the following system packages.

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

CMake 3.27+ is required. If your system cmake is older:
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

## Quick Start

### 1. Clone repositories

```bash
mkdir ~/LCM && cd ~/LCM

git clone git@github.com:trilinos/Trilinos.git
git clone git@github.com:sandialabs/LCM.git
git clone -b dtk-2.0-tpetra-static-graph git@github.com:ikalash/DataTransferKit.git
cp -r DataTransferKit Trilinos/
```

### 2. Set up environment

Add to `~/.bashrc`:
```bash
export LCM_DIR=~/LCM
module use $LCM_DIR/LCM/doc/LCM/modulefiles
```

Log out and back in, or `source ~/.bashrc`.

### 3. Create the `lcm` symlink

```bash
cd ~/LCM
ln -s LCM/doc/LCM/build/lcm .
```

### 4. Build and test

```bash
cd ~/LCM
module load release           # loads serial-gcc-release environment

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

Load a module before building:
```bash
module load serial-clang-release
./lcm all 16 --module=serial-clang-release
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

The `doc/LCM/build/clone-build-test.sh` script clones all repositories from scratch and runs the full build and test suite. It is typically run via cron:

```bash
# In crontab:
00 00 * * 1-5 cd /home/lcm/LCM; bash -l -c "./clone-build-test.sh"
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
