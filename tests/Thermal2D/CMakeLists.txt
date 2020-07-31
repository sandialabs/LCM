if(NOT ALBANY_PARALLEL_ONLY)
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_implicit.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_implicit.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicit.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_explicit.yaml COPYONLY)

  # Create the test with this name and standard executable
  add_test(${testName}_Implicit ${Albany.exe} input_implicit.yaml)
  set_tests_properties(${testName}_Implicit PROPERTIES LABELS "Demo;Tpetra;Forward")
endif()
