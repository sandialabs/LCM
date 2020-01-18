# //*****************************************************************
# //    Albany 3.0: Copyright 2016 National Technology & Engineering Solutions
# of Sandia, LLC (NTESS). // This Software is released under the BSD license
# detailed     // //    in the file license.txt in the top-level Albany
# directory
# //*****************************************************************

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputLD.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputLD.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputPlasLD.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputPlasLD.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputPlasLDMueLuT64.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputPlasLDMueLuT64.yaml COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputElastSPRAdaptMmodelMueLuT64.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/inputElastSPRAdaptMmodelMueLuT64.yaml COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputElastSPRAdaptMmodel.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/inputElastSPRAdaptMmodel.yaml COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputElastSPRStressAdaptMmodel1M.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/inputElastSPRStressAdaptMmodel1M.yaml COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputElastSPRStressAdaptParasolid.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/inputElastSPRStressAdaptParasolid.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole_4_0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole_4_0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole_4_1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole_4_1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole_4_2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole_4_2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole_4_3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole_4_3.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole.xmt_txt
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole.xmt_txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole_mmodel.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole_mmodel.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data.perf
               ${CMAKE_CURRENT_BINARY_DIR}/data.perf COPYONLY)

# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 1. Create the test with this name and standard executable
add_test(${testName}_perf ${performanceTestScript})

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
