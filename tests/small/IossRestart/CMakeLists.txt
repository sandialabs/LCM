# Name the test with the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quadOut.exo
               ${CMAKE_CURRENT_BINARY_DIR}/quadOut.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quadOut.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/quadOut.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quadOut.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/quadOut.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quadOut.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/quadOut.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quadOut.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/quadOut.exo.4.3 COPYONLY)

# ##############################################################################
# Main tests          ###
# ##############################################################################

set(testName ${testNameRoot})

if(ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  add_test(${testName} ${Albany.exe} input.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Basic;Tpetra;Forward")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputEnrich.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputEnrich.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputConvert.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputConvert.yaml COPYONLY)

# ##############################################################################
# Refine tests          ###
# ##############################################################################

if(ALBANY_STK_PERCEPT)
  set(testName ${testNameRoot}_Refine)
endif()
