<img src="https://github.com/sandialabs/LCM/blob/main/wiki/albany_lcm.png" width="300">

# Albany-LCM

Albany-LCM (Laboratory for Computational Mechanics) is spin-off (fork) from <a href = "https://github.com/sandialabs/Albany">Albany</a>, an implicit, unstructured grid, finite element code for the solution and analysis of multiphysics problems. The Albany-LCM repository 
on the GitHub site contains almost 200 regression tests and examples
that demonstrate the code's capabilities on a wide variety of problems, with a strong focus on computational 
solid mechanics and thermo-mechanics.

## Features

### Analysis of complex multiphysics problems

![von Karman Vortex Street](https://drive.google.com/file/d/1HThIIwik3xljfcKgrmwp8CxKlFa7hcur/view?usp=sharing)

Illustration of a von Karman vortex street that forms around a heated tube bundle under certain conditions

### Software architecture

Albany-LCM heavily leverages the [Trilinos](https://trilinos.org) Framework, available at:

	git clone https://github.com/trilinos/Trilinos.git

Albany-LCM supports the solution of very large problems (those over 2.1 billion degrees of freedom) using MPI.
It relies on automatic differentiation from the Sacado library of Trilinos, which makes it straightforward to add
new PDEs/physics to the code.  Albany-LCM contains a wide variety of constitutive models for solid mechanics, 
ranging from simple linear elasticity to sophisticated nonlinear micro-structure models with plasticity (e.g., J2 plasticity, crystal plasticity), and the Schwarz alternating method for multi-scale coupling in solid mechanics.  It also houses the terrestrial component of the Arctic Coastal Erosion (ACE) model, a coupled thermo-mechanical model with 
some novel permafrost constitutive models currently under 
development as part of the InterFACE project.

## Building Albany

To get started with Albany-LCM it is helpful to consult the
build instructions for both Trilinos and Albany located on the Albany wiki at
[https://github.com/sandialabs/Albany/wiki/Building-Albany-and-supporting-tools](https://github.com/sandialabs/Albany/wiki/Building-Albany-and-supporting-tools).
For help with building the code, please contact <a href = www.sandia.gov/~ikalash>Irina Tezaur</a> (ikalash@sandia.gov) or Alejandro 
Mota (amota@sandia.gov).  

## Nightly Build and Test Results

The Albany-LCM repository is tested nightly on several CPU-based architectures, with the 
results posted to an internal-to-Sandia CDash site.

The regression test suite is contained within the Albany-LCM repository in the directories:

	/tests

These tests are stand-alone and also serve as nice examples about how to describe various multiphysics problems.
They also serve as a template for developing new simulations.

Once Albany-LCM is built, the default test suite is executed by typing `ctest`
within the build directory. Any individual test can be executed by
moving into its sub-directory, and executing `ctest` in that
sub-directory. Many Albany-LCM tests run in parallel using up to 4 MPI ranks.

## Documentation

Unfortunately, we do not have up-to-date documentation of Albany-LCM; 
the interested user may wish to consult the (out-of-date) [HTML user guide](http://sandialabs.github.io/Albany/user-guide/guide.html) inside the Albany-LCM repository at:

	/doc/user-guide/guide.html

The LaTeX Developer's Guide (also out-of-date) is located at:

	/doc/developersGuide


## Note on unsupported code

When Albany-LCM was first created by forking from the main Albany branch, the idea
was to use this code primarily for analyses involving mechanics and thermo-mechanics, 
as the name of the code suggests.  To facilitate development of the code, a decision
was made to remove PDEs and capabilities that were no longer funded, including 
application-specific PDEs, Kokkos kernels and mesh adaptation.  Users interested in this capability
should check out the main Albany repository, which houses a performance-portable land-ice model
known as MPAS-Albany Land Ice (MALI) and the Albany-SCOREC repository, which focuses on 
developing capabilities for additive manufacturing and includes adaptive mesh refinement (AMR) via 
the Parallel Unstructured Mesh Interface (PUMI).  
