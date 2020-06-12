# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanics.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/mechanics.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastAdapt.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastAdapt.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechNodeSizeField.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/mechNodeSizeField.yaml COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/smooth_tension_coarse_tet4.exo
  ${CMAKE_CURRENT_BINARY_DIR}/smooth_tension_coarse_tet4.exo COPYONLY)
# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 1. Create the test with this name and standard executable

if(SEACAS_EPU AND SEACAS_EXODIFF)
  add_test(${testName}_ElasticML ${Albany.exe} inputElastAdapt.yaml)
  add_test(${testName} ${Albany.exe} mechanics.yaml)

  add_test(
    NAME ${testName}_NodeSizeField
    COMMAND
      ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
      -DFILE_NAME=tet4_node_size_field_test_output_1
      -DTEST_ARGS=mechNodeSizeField.yaml -DMPIMNP=${MPIMNP}
      -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
      -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P
      ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
  set_tests_properties(${testName}_NodeSizeField
                       PROPERTIES LABELS "LCM;Tpetra;Forward")

endif()
