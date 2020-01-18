#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

# Setup paths for SEACAS tools needed to run some of the tests
if(ALBANY_SEACAS)
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
endif()

# Paths to the actual executables
set(AlbanyPath ${Albany_BINARY_DIR}/src/Albany)

if(ALBANY_MPI)
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
  if(ALBANY_ENABLE_CUDA)
    set(Albany.exe ${PARALLEL_CALL} ${AlbanyPath} ${KOKKOS_NDEVICES})
    set(Albany8.exe
        ${MPIEX}
        ${MPIPRE}
        ${MPINPF}
        ${MAX_MPI_RANKS}
        ${MPIPOST}
        ${AlbanyPath}
        ${KOKKOS_NDEVICES})
  else()
    set(Albany.exe ${PARALLEL_CALL} ${AlbanyPath})
    set(Albany8.exe ${MPIEX} ${MPIPRE} ${MPINPF} ${MAX_MPI_RANKS} ${MPIPOST}
                    ${AlbanyPath})
  endif()
else()
  set(SerialAlbany.exe ${AlbanyPath})
  set(Albany.exe ${AlbanyPath})
endif()

if(ALBANY_LCM)
  set(PartitionTest.exe ${Albany_BINARY_DIR}/src/LCM/PartitionTest)
  set(Subdivision.exe ${Albany_BINARY_DIR}/src/LCM/Subdivision)
  set(MPS.exe ${Albany_BINARY_DIR}/src/LCM/MaterialPointSimulator)
  set(MPST.exe ${Albany_BINARY_DIR}/src/LCM/MaterialPointSimulatorT)
  set(DTK_Interp_and_Error.exe
      ${Albany_BINARY_DIR}/src/LCM/DTK_Interp_and_Error)
  set(DTK_Interp_Volume_to_NS.exe
      ${Albany_BINARY_DIR}/src/LCM/DTK_Interp_Volume_to_NS)
  if(ALBANY_MPI)
    set(Parallel_DTK_Interp_and_Error.exe ${PARALLEL_CALL}
                                          ${DTK_Interp_and_Error.exe})
    set(Parallel_DTK_Interp_Volume_to_NS.exe ${PARALLEL_CALL}
                                             ${DTK_Interp_Volume_to_NS.exe})
  endif()
endif(ALBANY_LCM)

add_subdirectory(small)
if(ALBANY_LARGE_TESTS)
  add_subdirectory(large)
endif()
