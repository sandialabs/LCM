#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

add_subdirectory(bcs)
#add_subdirectory(bcs-orig)
add_subdirectory(grid)
add_subdirectory(external_files)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_old_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/coupled_old_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_new_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/coupled_new_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_old_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/thermal_old_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_new_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/thermal_new_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_old_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/mechanical_old_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_new_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/mechanical_new_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_thermal_old_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/materials_thermal_old_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_thermal_new_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/materials_thermal_new_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_mechanical_old_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/materials_mechanical_old_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_mechanical_new_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/materials_mechanical_new_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_serial.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_new_wave_press_nbc_gold.e-s.27
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_new_wave_press_nbc_gold.e-s.27 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_new_wave_press_nbc_gold.e-s.27
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_new_wave_press_nbc_gold.e-s.27 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_old_wave_press_nbc_gold.e-s.50
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_old_wave_press_nbc_gold.e-s.50 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_old_wave_press_nbc_gold.e-s.50
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_old_wave_press_nbc_gold.e-s.50 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/thermal.exodiff COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests

#Serial test - new ACE wave pressure NBC
set(OUTFILE1 "thermal_new_wave_press_nbc.e-s.27")
set(REF_FILE1 "thermal_new_wave_press_nbc_gold.e-s.27")
set(OUTFILE2 "mechanical_new_wave_press_nbc.e-s.27")
set(REF_FILE2 "mechanical_new_wave_press_nbc_gold.e-s.27")
set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake)
add_test(
  NAME ACE_${testName}_NewWavePressNBC_Serial
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME1=thermal
    -DTEST_NAME2=mechanical
    -DTEST_ARGS=coupled_new_wave_press_nbc.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1} 
    -DREF_FILENAME2=${REF_FILE2} -DOUTPUT_FILENAME2=${OUTFILE2} -P
    ${runtest.cmake})
set_tests_properties(ACE_${testName}_NewWavePressNBC_Serial PROPERTIES LABELS "LCM;Tpetra;Forward")

#Serial test - old ACE wave pressure NBC
set(OUTFILE1 "thermal_old_wave_press_nbc.e-s.50")
set(REF_FILE1 "thermal_old_wave_press_nbc_gold.e-s.50")
set(OUTFILE2 "mechanical_old_wave_press_nbc.e-s.50")
set(REF_FILE2 "mechanical_old_wave_press_nbc_gold.e-s.50")
set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake)
add_test(
  NAME ACE_${testName}_OldWavePressNBC_Serial
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME1=thermal
    -DTEST_NAME2=mechanical
    -DTEST_ARGS=coupled_old_wave_press_nbc.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1} 
    -DREF_FILENAME2=${REF_FILE2} -DOUTPUT_FILENAME2=${OUTFILE2} -P
    ${runtest.cmake})
set_tests_properties(ACE_${testName}_OldWavePressNBC_Serial PROPERTIES LABELS "LCM;Tpetra;Forward")
  
#Parallel test
#set(OUTFILE1 "thermal.e-s.78")
#set(REF_FILE1 "thermal_gold.e-s.78")
#set(OUTFILE2 "mechanical.e-s.78")
#set(REF_FILE2 "mechanical_gold.e-s.78")
#set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
#add_test(
#  NAME ACE_${testName}_Parallel
#  COMMAND
#  ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}" -DTEST_NAME1=thermal
#    -DTEST_NAME2=mechanical
#    -DTEST_ARGS=coupled_old_wave_press_nbc.yaml -DMPIMNP=4 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
#    -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1} 
#    -DREF_FILENAME2=${REF_FILE2} -DOUTPUT_FILENAME2=${OUTFILE2} -P
#    ${runtest.cmake})
#set_tests_properties(ACE_${testName}_Parallel PROPERTIES LABELS "LCM;Tpetra;Forward")
  
