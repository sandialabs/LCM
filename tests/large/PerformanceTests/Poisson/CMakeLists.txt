# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data.perf
               ${CMAKE_CURRENT_BINARY_DIR}/data.perf COPYONLY)

# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 1. Create the test with this name and standard executable
add_test(${testName}_perf ${performanceTestScript})
add_test(${testName}_perf_2 ${performanceTestScript_2})

# Disable test if there isn't an entry for the current machine in data.perf

# Ignore empty tokens in "listification" of strings
cmake_policy(SET CMP0007 OLD)

file(READ data.perf contents)

string(REGEX REPLACE ";" "\\\\;" contents "${contents}")
string(REGEX REPLACE "\n" ";" contents "${contents}")

foreach(text_line ${contents})
  # message("${text_line}")
  string(REGEX REPLACE "   " ";" text_line "${text_line}")
  string(REGEX REPLACE "  " ";" text_line "${text_line}")
  string(REGEX REPLACE " " ";" text_line "${text_line}")
  list(GET text_line 0 output)
  if("${machineName}" STREQUAL output)
    return()
  endif()
endforeach()
set_tests_properties(
  ${testName}_perf
  PROPERTIES REQUIRED_FILES
             "machine ${machineName} not found in data.perf file")
set_tests_properties(
  ${testName}_perf_2
  PROPERTIES REQUIRED_FILES
             "machine ${machineName}_2 not found in data.perf file")
