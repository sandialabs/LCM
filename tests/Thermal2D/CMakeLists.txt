if(NOT ALBANY_PARALLEL_ONLY)
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_backward_euler.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_backward_euler.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_forward_euler.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_forward_euler.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_gerk.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_gerk.yaml COPYONLY)

  # Create the test with this name and standard executable
  add_test(${testName}_BackwardEuler ${Albany.exe} input_backward_euler.yaml)
  set_tests_properties(${testName}_BackwardEuler PROPERTIES LABELS "Demo;Tpetra;Forward")

  add_test(${testName}_ForwardEuler ${Albany.exe} input_forward_euler.yaml)
  set_tests_properties(${testName}_ForwardEuler PROPERTIES LABELS "Demo;Tpetra;Forward")
endif()
