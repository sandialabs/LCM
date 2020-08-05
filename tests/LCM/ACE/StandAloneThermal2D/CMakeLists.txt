if(NOT ALBANY_PARALLEL_ONLY)
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy Input file from source to binary dir
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/input_block_dept_2blks.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/input_block_dept_2blks.yaml COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/input_block_dept_2blks_explicit.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/input_block_dept_2blks_explicit.yaml COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/input_block_dept_4blks.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/input_block_dept_4blks.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_2blks.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/materials_2blks.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_4blks.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/materials_4blks.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square_2blocks.e
                 ${CMAKE_CURRENT_BINARY_DIR}/square_2blocks.e COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square_2blocks.e.4.0
                 ${CMAKE_CURRENT_BINARY_DIR}/square_2blocks.e.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square_2blocks.e.4.1
                 ${CMAKE_CURRENT_BINARY_DIR}/square_2blocks.e.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square_2blocks.e.4.2
                 ${CMAKE_CURRENT_BINARY_DIR}/square_2blocks.e.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square_2blocks.e.4.3
                 ${CMAKE_CURRENT_BINARY_DIR}/square_2blocks.e.4.3 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square_4blocks.e
                 ${CMAKE_CURRENT_BINARY_DIR}/square_4blocks.e COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square_4blocks.e.4.0
                 ${CMAKE_CURRENT_BINARY_DIR}/square_4blocks.e.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square_4blocks.e.4.1
                 ${CMAKE_CURRENT_BINARY_DIR}/square_4blocks.e.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square_4blocks.e.4.2
                 ${CMAKE_CURRENT_BINARY_DIR}/square_4blocks.e.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square_4blocks.e.4.3
                 ${CMAKE_CURRENT_BINARY_DIR}/square_4blocks.e.4.3 COPYONLY)

  # Create the test with this name and standard executable
  add_test(ACE_${testName}_2Blocks ${Albany.exe} input_block_dept_2blks.yaml)
  set_tests_properties(ACE_${testName}_2Blocks PROPERTIES LABELS
                                                      "Demo;Tpetra;Forward")
  add_test(ACE_${testName}_2Blocks_Explicit ${Albany.exe} input_block_dept_2blks_explicit.yaml)
  set_tests_properties(ACE_${testName}_2Blocks_Explicit PROPERTIES LABELS
                                                      "Demo;Tpetra;Forward")

  add_test(ACE_${testName}_4Blocks ${Albany.exe} input_block_dept_4blks.yaml)
  set_tests_properties(ACE_${testName}_4Blocks PROPERTIES LABELS
                                                      "Demo;Tpetra;Forward")
endif()
