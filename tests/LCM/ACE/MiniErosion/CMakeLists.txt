#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

add_subdirectory(bcs)
add_subdirectory(grid)
add_subdirectory(external_files)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_thermal_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_thermal_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_serial.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/coupled_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_mechanical_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_mechanical_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/denudation.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/denudation.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/denudation_gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/denudation_gold.e COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Serial test - denudation with element death (short run: 20 steps).
# M4: the ACE thermo-mechanical solver writes a single coupled Exodus
# output file (denudation.e) containing both physics' fields on the
# shared mesh, so the test does one exodiff against denudation_gold.e.
set(OUTFILE "denudation.e")
set(REF_FILE "denudation_gold.e")
set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake)
add_test(
  NAME ACE_${testName}_Denudation_Serial
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME=denudation
    -DTEST_ARGS=coupled_denudation.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE} -P
    ${runtest.cmake})
set_tests_properties(ACE_${testName}_Denudation_Serial PROPERTIES LABELS "LCM;Tpetra;Forward")
