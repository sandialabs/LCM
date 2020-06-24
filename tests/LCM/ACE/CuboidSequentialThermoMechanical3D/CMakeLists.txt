if(NOT ALBANY_PARALLEL_ONLY)
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/coupled.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_thermalOnly.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/coupled_thermalOnly.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_vardt.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/coupled_vardt.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/thermal.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_standalone.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/thermal_standalone.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_standalone_restart.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/thermal_standalone_restart.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_vardt.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/thermal_vardt.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/mechanical.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_vardt.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/mechanical_vardt.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/bluff_salinity.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/bluff_salinity.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clay_frac.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/clay_frac.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/depth.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/depth.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/grid.g
                 ${CMAKE_CURRENT_BINARY_DIR}/grid.g COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/grid.g.4.0
                 ${CMAKE_CURRENT_BINARY_DIR}/grid.g.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/grid.g.4.1
                 ${CMAKE_CURRENT_BINARY_DIR}/grid.g.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/grid.g.4.2
                 ${CMAKE_CURRENT_BINARY_DIR}/grid.g.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/grid.g.4.3
                 ${CMAKE_CURRENT_BINARY_DIR}/grid.g.4.3 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_thermal.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/materials_thermal.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_mechanical.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/materials_mechanical.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/peat_frac.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/peat_frac.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/porosity.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/porosity.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/sand_frac.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/sand_frac.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/silt_frac.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/silt_frac.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/time_fake.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/time_fake.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/waterH_fake.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/waterH_fake.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
                 ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal.exodiff
                 ${CMAKE_CURRENT_BINARY_DIR}/thermal.exodiff COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal-vardt.exodiff
                 ${CMAKE_CURRENT_BINARY_DIR}/thermal-vardt.exodiff COPYONLY)
  set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)

  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  # Add the tests
  set(OUTFILE "thermal119.e")
  set(REF_FILE "${CMAKE_CURRENT_BINARY_DIR}/../CuboidThermoMechanical3D/grid.e")
  add_test(
	  NAME ACE_${testName}
    COMMAND
      ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}" -DTEST_NAME=thermal
      -DTEST_ARGS=coupled.yaml -DMPIMNP=4 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
      -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE} -P
      ${runtest.cmake})
  set_tests_properties(ACE_${testName} PROPERTIES LABELS "LCM;Tpetra;Forward")
  set_tests_properties(ACE_${testName}
                       PROPERTIES DEPENDS CuboidThermoMechanical3D_ThermalOnly)
  
  set(OUTFILE "thermal-vardt40.e")
  set(REF_FILE "thermal119.e")
  add_test(
    NAME ACE_${testName}_vardt
    COMMAND
      ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}" -DTEST_NAME=thermal-vardt
      -DTEST_ARGS=coupled_vardt.yaml -DMPIMNP=4 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
      -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE} -P
      ${runtest.cmake})
  set_tests_properties(ACE_${testName}_vardt PROPERTIES LABELS "Demo;Tpetra;Forward")
  set_tests_properties(ACE_${testName}_vardt
	  PROPERTIES DEPENDS ${testName})
endif()
