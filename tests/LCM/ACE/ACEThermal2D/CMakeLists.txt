if(NOT ALBANY_PARALLEL_ONLY)
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_block_dept.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_block_dept.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square.e
                 ${CMAKE_CURRENT_BINARY_DIR}/square.e COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square.e.4.0
                 ${CMAKE_CURRENT_BINARY_DIR}/square.e.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square.e.4.1
                 ${CMAKE_CURRENT_BINARY_DIR}/square.e.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square.e.4.2
                 ${CMAKE_CURRENT_BINARY_DIR}/square.e.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square.e.4.3
                 ${CMAKE_CURRENT_BINARY_DIR}/square.e.4.3 COPYONLY)

  # Create the test with this name and standard executable
  add_test(${testName} ${Albany.exe} input.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Demo;Tpetra;Forward")
endif()
