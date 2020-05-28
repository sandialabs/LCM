if(NOT ALBANY_PARALLEL_ONLY)
  # Name the test with the directory name
  get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR}_SERIAL NAME)

  # ####################################################################################################################
  # Tests             ###
  # ####################################################################################################################

  set(testName ${testNameRoot}_no_prec)

  # if (NOT (ALBANY_CHECK_FPE OR ALBANY_STRONG_FPE_CHECK))
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  # Fails with FPE if FPE checking is enabled
  add_test(${testName} ${SerialAlbany.exe} input.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Demo;Tpetra;Forward;Serial")
  # endif ()

  set(testName ${testNameRoot}_Ifpack2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTP.yaml ${CMAKE_CURRENT_BINARY_DIR}/inputTP.yaml COPYONLY)

  add_test(${testName} ${SerialAlbany.exe} inputTP.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Demo;Tpetra;Forward;Serial")
endif()
