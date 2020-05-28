# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSplitSolution.yaml ${CMAKE_CURRENT_BINARY_DIR}/inputSplitSolution.yaml
               COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSplitSolution.yaml ${CMAKE_CURRENT_BINARY_DIR}/inputSplitSolution.yaml
               COPYONLY)

# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_test(${testName} ${Albany.exe} input.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "LCM;Tpetra;Forward")
add_test(${testName}_SplitOutput ${Albany.exe} inputSplitSolution.yaml)
set_tests_properties(${testName}_SplitOutput PROPERTIES LABELS "LCM;Tpetra;Forward")
