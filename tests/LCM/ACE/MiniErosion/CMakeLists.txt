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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_denudation_im.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/coupled_denudation_im.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation_im.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation_im.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_mechanical_denudation_im.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_mechanical_denudation_im.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_im.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_im.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_im.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_im.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation_im_gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation_im_gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation_im_gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation_im_gold.e COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Serial test - denudation with element death (short run: 20 steps)
set(OUTFILE1 "thermal_denudation.e")
set(REF_FILE1 "thermal_denudation_im_gold.e")
set(OUTFILE2 "mechanical_denudation.e")
set(REF_FILE2 "mechanical_denudation_im_gold.e")
set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake)
add_test(
  NAME ACE_${testName}_Denudation_IM_Serial
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME1=thermal_im
    -DTEST_NAME2=mechanical_im
    -DTEST_ARGS=coupled_denudation_im.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1}
    -DREF_FILENAME2=${REF_FILE2} -DOUTPUT_FILENAME2=${OUTFILE2} -P
    ${runtest.cmake})
set_tests_properties(ACE_${testName}_Denudation_IM_Serial PROPERTIES LABELS "LCM;Tpetra;Forward")
