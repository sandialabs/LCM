if(NOT ALBANY_PARALLEL_ONLY)
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_SERIAL NAME)

  # Copy mesh file from source dir to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.exo ${CMAKE_CURRENT_BINARY_DIR}/cube.exo COPYONLY)

  # Create the test with this name and standard executable
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  add_test(${testName} ${SerialAlbany.exe} input.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Basic;Tpetra;Forward")
endif()
