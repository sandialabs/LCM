# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SimpleMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SimpleMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2hex.g
               ${CMAKE_CURRENT_BINARY_DIR}/2hex.g COPYONLY)

# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if(NOT ALBANY_PARALLEL_ONLY)
  add_test(${testName}_SERIAL ${SerialAlbany.exe} input.yaml)
  set_tests_properties(${testName}_SERIAL PROPERTIES LABELS
                                                     "LCM;Tpetra;Forward")
endif()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputParallel.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputParallel.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HMaterials_fast_path.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/HMaterials_fast_path.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)
# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}Parallel NAME)
# 1. Create the test with this name and standard executable
if(NOT ALBANY_PARALLEL_ONLY)
  add_test(${testName}_SERIAL ${SerialAlbany.exe} inputParallel.yaml)
  set_tests_properties(${testName}_SERIAL PROPERTIES LABELS
                                                     "LCM;Tpetra;Forward")
endif()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputOrthogonal.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputOrthogonal.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HMaterials_fast_path.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/HMaterials_fast_path.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)
# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}Orthogonal NAME)
# 1. Create the test with this name and standard executable
if(NOT ALBANY_PARALLEL_ONLY)
  add_test(${testName}_SERIAL ${SerialAlbany.exe} inputOrthogonal.yaml)
  set_tests_properties(${testName}_SERIAL PROPERTIES LABELS
                                                     "LCM;Tpetra;Forward")
endif()
