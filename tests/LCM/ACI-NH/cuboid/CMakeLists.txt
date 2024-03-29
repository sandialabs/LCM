#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

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
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_serial.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_parallel.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_parallel.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.g
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_gold.e-s.23
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_gold.e-s.23 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_gold.e-s.23.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_gold.e-s.23.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_gold.e-s.23.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_gold.e-s.23.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_gold.e-s.23.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_gold.e-s.23.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_gold.e-s.23.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_gold.e-s.23.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_gold.e-s.23
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_gold.e-s.23 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_gold.e-s.23.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_gold.e-s.23.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_gold.e-s.23.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_gold.e-s.23.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_gold.e-s.23.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_gold.e-s.23.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_gold.e-s.23.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_gold.e-s.23.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/thermal.exodiff COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests

#Serial test - cuboid
set(OUTFILE1 "thermal.e-s.23")
set(REF_FILE1 "thermal_gold.e-s.23")
set(OUTFILE2 "mechanical.e-s.23")
set(REF_FILE2 "mechanical_gold.e-s.23")
set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake)
add_test(
NAME ACI-NH_${testName}_serial
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME1=thermal
    -DTEST_NAME2=mechanical
    -DTEST_ARGS=coupled.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1} 
    -DREF_FILENAME2=${REF_FILE2} -DOUTPUT_FILENAME2=${OUTFILE2} -P
    ${runtest.cmake})
set_tests_properties(ACI-NH_${testName}_serial PROPERTIES LABELS "LCM;Tpetra;Forward")

#Parallel test - cuboid
set(OUTFILE1 "thermal.e-s.23")
set(REF_FILE1 "thermal_gold.e-s.23")
set(OUTFILE2 "mechanical.e-s.23")
set(REF_FILE2 "mechanical_gold.e-s.23")
set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest_parallel.cmake)
add_test(
NAME ACI-NH_${testName}_parallel
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}" -DTEST_NAME1=thermal
    -DTEST_NAME2=mechanical
    -DTEST_ARGS=coupled.yaml -DMPIMNP=4 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1} 
    -DREF_FILENAME2=${REF_FILE2} -DOUTPUT_FILENAME2=${OUTFILE2} -P
    ${runtest.cmake})
set_tests_properties(ACI-NH_${testName}_parallel PROPERTIES LABELS "LCM;Tpetra;Forward")
