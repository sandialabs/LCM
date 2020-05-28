if(ALBANY_IOPX)
  # Name the test with the directory name
  get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy mesh file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/domain.exo ${CMAKE_CURRENT_BINARY_DIR}/domain.exo COPYONLY)

  set(testName ${testNameRoot}_Rythmos)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_rythmos.yaml ${CMAKE_CURRENT_BINARY_DIR}/input_rythmos.yaml COPYONLY)
  add_test(${testName} ${Albany.exe} input_rythmos.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Demo;Tpetra;Forward")

  set(testName ${testNameRoot}_Tempus)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_tempus.yaml ${CMAKE_CURRENT_BINARY_DIR}/input_tempus.yaml COPYONLY)

  add_test(${testName} ${Albany.exe} input_tempus.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Demo;Tpetra;Forward")
endif(ALBANY_IOPX)
