#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

# Regression test for the nodal projection in dynamic runs (GitHub issue #11):
# implicit dynamics on a Tet<10> cube with "Project IP to Nodal Field" written
# every step. Guards that the observer-driven projection writes a correct
# (non-zero) proj_nodal_* field in the first computed frame, via the built-in
# proj_nodal_linear field checked against a gold.

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/project_tet10_dyn.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/project_tet10_dyn.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single-tet10.g
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single-tet10.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/project_tet10_dyn.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/project_tet10_dyn.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/project_tet10_dyn_gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/project_tet10_dyn_gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_serial.cmake COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(OUTFILE "project_tet10_dyn.e")
set(REF_FILE "project_tet10_dyn_gold.e")
add_test(
  NAME ${testName}_Serial
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME=project_tet10_dyn
    -DTEST_ARGS=project_tet10_dyn.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE} -P
    ${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake)
set_tests_properties(${testName}_Serial PROPERTIES LABELS "LCM;Tpetra;Forward")
