# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EXODIFF}
                        ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${runtest.cmake}
                        ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityTwoFamilies.g
  ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityTwoFamilies.g COPYONLY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityTwoFamilies.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityTwoFamilies.yaml COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityTwoFamilies_Material.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityTwoFamilies_Material.yaml
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityTwoFamilies.exodiff
  ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityTwoFamilies.exodiff COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityTwoFamilies.gold.e
  ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityTwoFamilies.gold.e COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
set(OUTFILE "DislocationDensityTwoFamilies.e")
set(REF_FILE "DislocationDensityTwoFamilies.gold.e")
add_test(
  NAME CrystalPlasticity_${testName}
  COMMAND
    ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}"
    -DTEST_NAME=DislocationDensityTwoFamilies
    -DTEST_ARGS=DislocationDensityTwoFamilies.yaml -DMPIMNP=1
    -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -DREF_FILENAME=${REF_FILE}
    -DOUTPUT_FILENAME=${OUTFILE} -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P
    ${runtest.cmake})
set_tests_properties(CrystalPlasticity_${testName}
                     PROPERTIES LABELS "LCM;Tpetra;Forward")
