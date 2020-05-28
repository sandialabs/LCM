# Copy materials file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Executable to run before the tests
add_executable(HeatProfile HeatProfile.cpp)

# Create the test (actually the data for the test) with this name and standard executable

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)

add_test(
  NAME ${testName}
  COMMAND
    ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}" "-DTEST_ARGS=input.yaml" "-DAUX_ENV=${ALBANY_PRETEST_EXEC}"
    "-DAUX_PROG=${CMAKE_CURRENT_BINARY_DIR}/HeatProfile" -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(${testName} PROPERTIES LABELS "Basic;Tpetra;Forward")
