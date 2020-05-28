# Name the test with the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(testName ${testNameRoot}_2DUnsteadyInvPressPulse)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2Dunsteady.yaml ${CMAKE_CURRENT_BINARY_DIR}/input2Dunsteady.yaml
               COPYONLY)

add_test(${testName} ${Albany.exe} input2Dunsteady.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "Demo;Tpetra;Forward")
