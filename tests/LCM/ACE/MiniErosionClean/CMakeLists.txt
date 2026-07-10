#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#
# MiniErosionClean -- a calibrated ACE Sequential Thermo-Mechanical J2Erosion
# case built specifically to guard element-death PARALLEL CONSISTENCY.
#
# A staggered top-down thermal thaw front erodes the top 5 of 8 element layers
# (80 of 128 cells) as a receding front, leaving the bottom 3 layers frozen and
# intact. The run reaches Final Time, does NOT whole-mesh-die, and -- the point
# of the test -- serial and np4 erode the IDENTICAL final death set.
#
# Two tests are registered:
#
#   1. ..._Denudation_Serial  -- serial regression: exodiff vs denudation_gold.e.
#      Guards run-to-run stability.
#
#   2. ..._Denudation_np4_Consistency -- parallel consistency: run on 4 ranks,
#      epu-merge, then assert (compare_death_positions.py) that the FINAL eroded
#      cell set matches the serial gold, compared BY CELL CENTROID POSITION.
#      This is robust to the np4 run's partition-dependent solver cutback, which
#      gives it a different frame count and end time (a plain exodiff cannot be
#      used), and to epu's element renumbering.
#
# The element-death C++ algorithm may soon gain an outer re-solve loop, which
# will shift erosion TIMING -- and therefore the serial gold and the frame
# counts -- but NOT the final death SET. So the position-based consistency test
# is the durable invariant; the serial gold is the fragile piece.
#
# REGENERATING THE SERIAL GOLD (do this if the death-timing algorithm changes):
#   cd <build>/tests/LCM/ACE/MiniErosionClean
#   <build>/src/Albany coupled_denudation.yaml
#   cp denudation.e <source>/tests/LCM/ACE/MiniErosionClean/denudation_gold.e
# (denudation.e is the single coupled Exodus file the ACE solver writes.)

add_subdirectory(grid)
add_subdirectory(external_files)

# python3 for the position-based comparison. The comparison reads the exodus
# (classic netCDF) output with netCDF4 if present, else scipy.io.netcdf_file, so
# it only needs the numpy/scipy stack -- not netCDF4 -- to be available.
find_program(PYTHON3_EXE NAMES python3 python)

# Stage input decks, comparison assets, and the golds into the build dir.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/coupled_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_thermal_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_thermal_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_mechanical_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_mechanical_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/denudation.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/denudation.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/denudation_gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/denudation_gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compare_death_positions.py
               ${CMAKE_CURRENT_BINARY_DIR}/compare_death_positions.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_serial.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_np4_consistency.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_np4_consistency.cmake COPYONLY)

# Name the tests with the directory name.
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(OUTFILE "denudation.e")
set(REF_FILE "denudation_gold.e")

# --- Test 1: serial regression (exodiff vs serial gold) -------------------
set(runtest_serial.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake)
add_test(
  NAME ACE_${testName}_Denudation_Serial
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME=denudation
    -DTEST_ARGS=coupled_denudation.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE} -P
    ${runtest_serial.cmake})
set_tests_properties(ACE_${testName}_Denudation_Serial
                     PROPERTIES LABELS "LCM;Tpetra;Forward")

# --- Test 2: parallel consistency (position-based dead-set match) ---------
set(runtest_np4_consistency.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/runtest_np4_consistency.cmake)
add_test(
  NAME ACE_${testName}_Denudation_np4_Consistency
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}" -DTEST_NAME=denudation
    -DTEST_ARGS=coupled_denudation.yaml -DMPIMNP=${MPIMNP}
    -DSEACAS_EPU=${SEACAS_EPU}
    -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
    "-DPYTHON_EXE=${PYTHON3_EXE}"
    "-DCOMPARE_SCRIPT=${CMAKE_CURRENT_BINARY_DIR}/compare_death_positions.py" -P
    ${runtest_np4_consistency.cmake})
set_tests_properties(ACE_${testName}_Denudation_np4_Consistency
                     PROPERTIES LABELS "LCM;Tpetra;Forward"
                                PROCESSORS ${MPIMNP})
