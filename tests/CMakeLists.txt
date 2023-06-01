#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

# Setup paths for SEACAS tools needed to run some of the tests
if(DEFINED ALBANY_SEACAS_PATH)
  find_program(
    SEACAS_EPU
    NAMES epu
    PATHS ${ALBANY_SEACAS_PATH})
  find_program(
    SEACAS_EXODIFF
    NAMES exodiff
    PATHS ${ALBANY_SEACAS_PATH})
  find_program(
    SEACAS_ALGEBRA
    NAMES algebra
    PATHS ${ALBANY_SEACAS_PATH})
  find_program(
    SEACAS_DECOMP
    NAMES decomp
    PATHS ${ALBANY_SEACAS_PATH})
else()
  find_program(
    SEACAS_EPU
    NAMES epu
    PATHS ${Trilinos_BIN_DIRS} ENV PATH)
  find_program(
    SEACAS_EXODIFF
    NAMES exodiff
    PATHS ${Trilinos_BIN_DIRS} ENV PATH)
  find_program(
    SEACAS_ALGEBRA
    NAMES algebra
    PATHS ${Trilinos_BIN_DIRS} ENV PATH)
  find_program(
    SEACAS_DECOMP
    NAMES decomp
    PATHS ${Trilinos_BIN_DIRS} ENV PATH)
endif()
message("-- SEACAS_ALGEBRA = ${SEACAS_ALGEBRA}")
message("-- SEACAS_DECOMP  = ${SEACAS_DECOMP}")
message("-- SEACAS_EPU     = ${SEACAS_EPU}")
message("-- SEACAS_EXODIFF = ${SEACAS_EXODIFF}")

# Paths to the actual executables
set(AlbanyPath ${Albany_BINARY_DIR}/src/Albany)

set(SerialAlbany.exe ${SERIAL_CALL} ${AlbanyPath})
# Do not test on greater than Trilinos_MPI_EXEC_MAX_NUMPROCS configured in
# Trilinos build or explicity given at Albany configure time -D
# ALBANY_MPI_EXEC_MAX_NUMPROCS
if(DEFINED MPIMNP)
  if(MPIMNP LESS 8)
    set(MAX_MPI_RANKS ${MPIMNP})
  else(MPIMNP LESS 8)
    set(MAX_MPI_RANKS 8)
  endif(MPIMNP LESS 8)
else(DEFINED MPIMNP)
  set(MAX_MPI_RANKS 4)
endif(DEFINED MPIMNP)
set(Albany.exe ${PARALLEL_CALL} ${AlbanyPath})
set(Albany8.exe ${MPIEX} ${MPIPRE} ${MPINPF} ${MAX_MPI_RANKS} ${MPIPOST}
                ${AlbanyPath})

set(PartitionTest.exe ${Albany_BINARY_DIR}/src/LCM/PartitionTest)
set(Subdivision.exe ${Albany_BINARY_DIR}/src/LCM/Subdivision)
set(MPS.exe ${Albany_BINARY_DIR}/src/LCM/MaterialPointSimulator)
set(MPST.exe ${Albany_BINARY_DIR}/src/LCM/MaterialPointSimulatorT)
set(DTK_Interp_and_Error.exe ${Albany_BINARY_DIR}/src/LCM/DTK_Interp_and_Error)
set(DTK_Interp_Volume_to_NS.exe
    ${Albany_BINARY_DIR}/src/LCM/DTK_Interp_Volume_to_NS)
set(Parallel_DTK_Interp_and_Error.exe ${PARALLEL_CALL}
                                      ${DTK_Interp_and_Error.exe})
set(Parallel_DTK_Interp_Volume_to_NS.exe ${PARALLEL_CALL}
                                         ${DTK_Interp_Volume_to_NS.exe})

#
# The tests
#

# Solid Mechanics problems
add_subdirectory(LCM)

# Heat Transfer Problems
add_subdirectory(ContinuationHeat1D)
add_subdirectory(Heat2DTriangles)
add_subdirectory(HeatEigenvalues)
add_subdirectory(SideSetLaplacian)
add_subdirectory(SteadyHeat1D)
add_subdirectory(SteadyHeat1DEB)
add_subdirectory(SteadyHeat2D)
add_subdirectory(SteadyHeat2DEB)
add_subdirectory(SteadyHeat3D)
add_subdirectory(SteadyHeat3DTest)
add_subdirectory(TransientHeat1D)
add_subdirectory(TransientHeat2D)
add_subdirectory(Thermal1D)
add_subdirectory(Thermal2D)
add_subdirectory(Thermal3D)

if(ALBANY_PAMGEN)
  add_subdirectory(Heat3DPamgen)
endif()
add_subdirectory(Heat2DMMCylWithSource)
add_subdirectory(HeatQuadTri)
add_subdirectory(TransientHeat2DTableSource)
add_subdirectory(Ioss2D)
add_subdirectory(Ioss3D)
add_subdirectory(IossRestart)
add_subdirectory(SteadyHeat2DInternalNeumann)
add_subdirectory(SteadyHeat2DRobin)
add_subdirectory(SteadyHeat2DSS)
add_subdirectory(Utils)

# DEMO PDES
if(ALBANY_DEMO_PDES)
  add_subdirectory(AdvDiff)
  add_subdirectory(CahnHillElast2D)
  add_subdirectory(ComprNS)
  add_subdirectory(Helmholtz2D)
  add_subdirectory(NSPoiseuille2D)
  add_subdirectory(NSRayleighBernard2D)
  add_subdirectory(NSVortexShedding2D)
  add_subdirectory(ODE)
  add_subdirectory(PNP)
  add_subdirectory(ReactDiffSystem)
  add_subdirectory(ThermoElectrostatics2D)
endif(ALBANY_DEMO_PDES)
