# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTraction.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTraction.yaml COPYONLY)

# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 1. Create the test with this name and standard executable

if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND (ALBANY_CHECK_FPE
                                                    OR ALBANY_STRONG_FPE_CHECK)
       ))
  # Fails with FPE strong check with Clang
  if(ALBANY_MUELU_EXAMPLES)
    add_test(${testName} ${Albany.exe} input.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "LCM;Tpetra;Forward")
    add_test(${testName}_Traction ${Albany.exe} inputTraction.yaml)
    set_tests_properties(${testName}_Traction PROPERTIES LABELS
                                                         "LCM;Tpetra;Forward")
  endif()
endif()
