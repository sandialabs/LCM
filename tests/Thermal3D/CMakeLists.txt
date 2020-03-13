if(NOT ALBANY_PARALLEL_ONLY)
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_ACE.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_ACE.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g
                 ${CMAKE_CURRENT_BINARY_DIR}/cube.g COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g.4.0
                 ${CMAKE_CURRENT_BINARY_DIR}/cube.g.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g.4.1
                 ${CMAKE_CURRENT_BINARY_DIR}/cube.g.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g.4.2
                 ${CMAKE_CURRENT_BINARY_DIR}/cube.g.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g.4.3
                 ${CMAKE_CURRENT_BINARY_DIR}/cube.g.4.3 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/bc_time.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/bc_time.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/disp.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/disp.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/level.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/level.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ocean_salinity.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/ocean_salinity.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/salinity.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/salinity.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/temp.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/temp.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/time.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/time.txt COPYONLY)


  # Create the test with this name and standard executable
  add_test(${testName} ${Albany.exe} input.yaml)
  add_test(${testName}_ACE ${Albany.exe} input_ACE.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS
                                              "Demo;Tpetra;Forward")
endif()
