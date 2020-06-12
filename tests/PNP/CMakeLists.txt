# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_test2D NAME)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_test2D.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_test2D.yaml COPYONLY)

add_test(${testName} ${Albany.exe} input_test2D.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "Demo;Tpetra;Forward")
