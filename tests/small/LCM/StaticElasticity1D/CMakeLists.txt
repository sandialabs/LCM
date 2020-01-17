# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputP.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputP.yaml COPYONLY)
# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
if(ALBANY_IFPACK2)
  if(NOT (ALBANY_CHECK_FPE OR ALBANY_STRONG_FPE_CHECK))
    # Dont need IFPACK2 for this first test - this is unpreconditioned matrix-
    # free. Fails with FPE.
    add_test(${testName}_MF ${Albany.exe} input.yaml)
    set_tests_properties(${testName}_MF PROPERTIES LABELS "LCM;Tpetra;Forward")
  endif()
  add_test(${testName}_Ifpack2 ${Albany.exe} inputP.yaml)
  set_tests_properties(${testName}_Ifpack2 PROPERTIES LABELS
                                                      "LCM;Tpetra;Forward")
endif()
