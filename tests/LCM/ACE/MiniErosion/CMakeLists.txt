#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

add_subdirectory(bcs)
#add_subdirectory(bcs-orig)
add_subdirectory(grid)
add_subdirectory(external_files)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_new_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/coupled_new_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_hs_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/coupled_hs_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_new_wave_press_nbc_implicit.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/coupled_new_wave_press_nbc_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_denudation.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/coupled_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_denudation_tempus_noerosion.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/coupled_denudation_tempus_noerosion.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_denudation_tempus_noerosion_all_implicit.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/coupled_denudation_tempus_noerosion_all_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_new_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/thermal_new_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_hs_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/thermal_hs_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_new_wave_press_nbc_implicit.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/thermal_new_wave_press_nbc_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation_tempus_noerosion.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation_tempus_noerosion.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation_tempus_noerosion_implicit.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation_tempus_noerosion_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_new_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/mechanical_new_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_hs_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/mechanical_hs_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_new_wave_press_nbc_implicit.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/mechanical_new_wave_press_nbc_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation_tempus_noerosion.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation_tempus_noerosion.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation_tempus_noerosion_implicit.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation_tempus_noerosion_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_thermal_new_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/materials_thermal_new_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_thermal_denudation.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/materials_thermal_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_mechanical_new_wave_press_nbc.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/materials_mechanical_new_wave_press_nbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_mechanical_denudation.yaml
        	${CMAKE_CURRENT_BINARY_DIR}/materials_mechanical_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_serial.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_new_wave_press_nbc_gold.e-s.6
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_new_wave_press_nbc_gold.e-s.6 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_new_wave_press_nbc_gold.e-s.6
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_new_wave_press_nbc_gold.e-s.6 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_hs_wave_press_nbc_gold.e-s.6
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_hs_wave_press_nbc_gold.e-s.6 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_hs_wave_press_nbc_gold.e-s.6
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_hs_wave_press_nbc_gold.e-s.6 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation_gold.e-s.71
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation_gold.e-s.71 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation_tempus_noerosion_implicit_gold.e-s.71.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation_tempus_noerosion_implicit_gold.e-s.71.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation_tempus_noerosion_implicit_gold.e-s.71.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation_tempus_noerosion_implicit_gold.e-s.71.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation_tempus_noerosion_implicit_gold.e-s.71.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation_tempus_noerosion_implicit_gold.e-s.71.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation_tempus_noerosion_implicit_gold.e-s.71.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation_tempus_noerosion_implicit_gold.e-s.71.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation_tempus_noerosion_explicit_gold.e-s.71.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation_tempus_noerosion_explicit_gold.e-s.71.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation_tempus_noerosion_explicit_gold.e-s.71.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation_tempus_noerosion_explicit_gold.e-s.71.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation_tempus_noerosion_explicit_gold.e-s.71.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation_tempus_noerosion_explicit_gold.e-s.71.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation_tempus_noerosion_explicit_gold.e-s.71.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation_tempus_noerosion_explicit_gold.e-s.71.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation_gold.e-s.71
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation_gold.e-s.71 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation_tempus_noerosion_implicit_gold.e-s.71.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation_tempus_noerosion_implicit_gold.e-s.71.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation_tempus_noerosion_implicit_gold.e-s.71.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation_tempus_noerosion_implicit_gold.e-s.71.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation_tempus_noerosion_implicit_gold.e-s.71.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation_tempus_noerosion_implicit_gold.e-s.71.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation_tempus_noerosion_implicit_gold.e-s.71.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation_tempus_noerosion_implicit_gold.e-s.71.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation_tempus_noerosion_explicit_gold.e-s.71.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation_tempus_noerosion_explicit_gold.e-s.71.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation_tempus_noerosion_explicit_gold.e-s.71.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation_tempus_noerosion_explicit_gold.e-s.71.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation_tempus_noerosion_explicit_gold.e-s.71.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation_tempus_noerosion_explicit_gold.e-s.71.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation_tempus_noerosion_explicit_gold.e-s.71.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation_tempus_noerosion_explicit_gold.e-s.71.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/thermal.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_implicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_implicit.exodiff COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests

#Serial test - new ACE wave pressure NBC
set(OUTFILE1 "thermal_new_wave_press_nbc.e-s.6")
set(REF_FILE1 "thermal_new_wave_press_nbc_gold.e-s.6")
set(OUTFILE2 "mechanical_new_wave_press_nbc.e-s.6")
set(REF_FILE2 "mechanical_new_wave_press_nbc_gold.e-s.6")
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

#Serial test - new ACE hydrostatic pressure NBC
set(OUTFILE1 "thermal_hs_wave_press_nbc.e-s.6")
set(REF_FILE1 "thermal_hs_wave_press_nbc_gold.e-s.6")
set(OUTFILE2 "mechanical_hs_wave_press_nbc.e-s.6")
set(REF_FILE2 "mechanical_hs_wave_press_nbc_gold.e-s.6")
set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake)
add_test(
  NAME ACE_${testName}_HydrostaticPressNBC_Serial
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME1=thermal
    -DTEST_NAME2=mechanical
    -DTEST_ARGS=coupled_hs_wave_press_nbc.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1} 
    -DREF_FILENAME2=${REF_FILE2} -DOUTPUT_FILENAME2=${OUTFILE2} -P
    ${runtest.cmake})
set_tests_properties(ACE_${testName}_HydrostaticPressNBC_Serial PROPERTIES LABELS "LCM;Tpetra;Forward")

#Serial test - denudation with erosion
#Serial test - denudation with erosion
set(OUTFILE1 "thermal_denudation.e-s.71")
set(REF_FILE1 "thermal_denudation_gold.e-s.71")
set(OUTFILE2 "mechanical_denudation.e-s.71")
set(REF_FILE2 "mechanical_denudation_gold.e-s.71")
set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake)
add_test(
NAME ACE_${testName}_Denudation_Serial
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME1=thermal
    -DTEST_NAME2=mechanical
    -DTEST_ARGS=coupled_denudation.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1} 
    -DREF_FILENAME2=${REF_FILE2} -DOUTPUT_FILENAME2=${OUTFILE2} -P
    ${runtest.cmake})
set_tests_properties(ACE_${testName}_Denudation_Serial PROPERTIES LABELS "LCM;Tpetra;Forward")

#Parallel test - denudation with erosion
#set(OUTFILE1 "thermal_denudation.e-s.71")
#set(REF_FILE1 "thermal_denudation_gold.e-s.71")
#set(OUTFILE2 "mechanical_denudation.e-s.71")
#set(REF_FILE2 "mechanical_denudation_gold.e-s.71")
#set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
#add_test(
#NAME ACE_${testName}_Denudation_Parallel
#  COMMAND
#  ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}" -DTEST_NAME1=thermal
#    -DTEST_NAME2=mechanical
#    -DTEST_ARGS=coupled_denudation.yaml -DMPIMNP=4 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
#    -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1} 
#    -DREF_FILENAME2=${REF_FILE2} -DOUTPUT_FILENAME2=${OUTFILE2} -P
#    ${runtest.cmake})
#set_tests_properties(ACE_${testName}_Denudation_Parallel PROPERTIES LABELS "LCM;Tpetra;Forward")
  
#Parallel test - denudation using Tempus w/o erosion w/ explicit thermal time-stepper
set(OUTFILE1 "thermal_denudation_tempus_noerosion.e-s.71")
set(REF_FILE1 "thermal_denudation_tempus_noerosion_explicit_gold.e-s.71")
set(OUTFILE2 "mechanical_denudation_tempus_noerosion.e-s.71")
set(REF_FILE2 "mechanical_denudation_tempus_noerosion_explicit_gold.e-s.71")
set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
add_test(
  NAME ACE_${testName}_Denudation_Tempus_NoErosion_ExplicitThermal_Parallel
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}" -DTEST_NAME1=thermal
    -DTEST_NAME2=mechanical
    -DTEST_ARGS=coupled_denudation_tempus_noerosion.yaml -DMPIMNP=4 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1} 
    -DREF_FILENAME2=${REF_FILE2} -DOUTPUT_FILENAME2=${OUTFILE2} -P
    ${runtest.cmake})
set_tests_properties(ACE_${testName}_Denudation_Tempus_NoErosion_ExplicitThermal_Parallel PROPERTIES LABELS "LCM;Tpetra;Forward")
  
#Parallel test - denudation using Tempus w/o erosion with implicit thermal time-stepper
set(OUTFILE1 "thermal_denudation_tempus_noerosion_implicitThermal.e-s.71")
set(REF_FILE1 "thermal_denudation_tempus_noerosion_implicit_gold.e-s.71")
set(OUTFILE2 "mechanical_denudation_tempus_noerosion_implicitThermal.e-s.71")
set(REF_FILE2 "mechanical_denudation_tempus_noerosion_implicit_gold.e-s.71")
set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
add_test(
  NAME ACE_${testName}_Denudation_Tempus_NoErosion_ImplicitThermal_Parallel
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}" -DTEST_NAME1=thermal_implicit
    -DTEST_NAME2=mechanical
    -DTEST_ARGS=coupled_denudation_tempus_noerosion_all_implicit.yaml -DMPIMNP=4 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1} 
    -DREF_FILENAME2=${REF_FILE2} -DOUTPUT_FILENAME2=${OUTFILE2} -P
    ${runtest.cmake})
set_tests_properties(ACE_${testName}_Denudation_Tempus_NoErosion_ImplicitThermal_Parallel PROPERTIES LABELS "LCM;Tpetra;Forward")
  
