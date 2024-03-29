# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EXODIFF}
                        ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${runtest.cmake}
                        ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

# Copy Input files from source to binary dir

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Implicit.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Implicit.yaml COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Material_Implicit.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Material_Implicit.yaml COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Implicit.exodiff
  ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Implicit.exodiff COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Implicit.gold.e
  ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Implicit.gold.e COPYONLY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Explicit.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Explicit.yaml COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Material_Explicit.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Material_Explicit.yaml COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Explicit.exodiff
  ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Explicit.exodiff COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Explicit.gold.e
  ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Explicit.gold.e COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests test 1 - Explicit
set(OUTFILE "SingleSlipHard_Explicit.e")
set(REF_FILE "SingleSlipHard_Explicit.gold.e")
add_test(
  NAME CrystalPlasticity_${testName}_Explicit
  COMMAND
    ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}"
    -DTEST_NAME=SingleSlipHard_Explicit
    -DTEST_ARGS=SingleSlipHard_Explicit.yaml -DMPIMNP=1
    -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -DREF_FILENAME=${REF_FILE}
    -DOUTPUT_FILENAME=${OUTFILE} -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P
    ${runtest.cmake})
set_tests_properties(CrystalPlasticity_${testName}_Explicit
                     PROPERTIES LABELS "LCM;Tpetra;Forward")
# test 2 - Implicit
set(OUTFILE "SingleSlipHard_Implicit.e")
set(REF_FILE "SingleSlipHard_Implicit.gold.e")
add_test(
  NAME CrystalPlasticity_${testName}_Implicit
  COMMAND
    ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}"
    -DTEST_NAME=SingleSlipHard_Implicit
    -DTEST_ARGS=SingleSlipHard_Implicit.yaml -DMPIMNP=1
    -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -DREF_FILENAME=${REF_FILE}
    -DOUTPUT_FILENAME=${OUTFILE} -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P
    ${runtest.cmake})
set_tests_properties(CrystalPlasticity_${testName}_Explicit
                     PROPERTIES LABELS "LCM;Tpetra;Forward")
