#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

add_subdirectory(bcs)
add_subdirectory(grid)
add_subdirectory(external_files)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/coupled.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/thermal.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/mechanical.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_thermal.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/materials_thermal.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_mechanical.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/materials_mechanical.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_gold.e-s.79
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_gold.e-s.79 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_gold.e-s.79
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_gold.e-s.79 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/thermal.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical.exodiff COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
set(OUTFILE1 "thermal.e-s.79")
#set(REF_FILE1 "${CMAKE_CURRENT_BINARY_DIR}/thermal_gold.e-s.79")
set(REF_FILE1 "thermal_gold.e-s.79")
set(OUTFILE2 "mechanical.e-s.79")
#set(REF_FILE2 "${CMAKE_CURRENT_BINARY_DIR}/mechanical_gold.e-s.79")
set(REF_FILE2 "mechanical_gold.e-s.79")
add_test(
  NAME ACE_${testName}
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME1=thermal
    -DTEST_NAME2=mechanical
    -DTEST_ARGS=coupled.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1} 
    -DREF_FILENAME2=${REF_FILE2} -DOUTPUT_FILENAME2=${OUTFILE2} -P
    ${runtest.cmake})
set_tests_properties(ACE_${testName} PROPERTIES LABELS "LCM;Tpetra;Forward")
  
