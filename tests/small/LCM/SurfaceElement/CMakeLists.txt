# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/3materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/3materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)

# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 1. Create the test with this name and standard executable
if(NOT ALBANY_PARALLEL_ONLY)
  if(ALBANY_IFPACK2)
    add_test(${testName}_SERIAL ${SerialAlbany.exe} input.yaml)
    set_tests_properties(${testName}_SERIAL PROPERTIES LABELS
                                                       "LCM;Tpetra;Forward")
  endif()
endif()
