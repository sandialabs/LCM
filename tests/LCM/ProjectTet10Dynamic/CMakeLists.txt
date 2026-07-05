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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/project_tet10_dyn_lumped.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/project_tet10_dyn_lumped.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/project_tet10_dyn_lumped.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/project_tet10_dyn_lumped.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/project_tet10_dyn_lumped_gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/project_tet10_dyn_lumped_gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_serial.cmake COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Consistent (Full) mass matrix: proj_nodal_linear is recovered exactly.
add_test(
  NAME ${testName}_Serial
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME=project_tet10_dyn
    -DTEST_ARGS=project_tet10_dyn.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME=project_tet10_dyn_gold.e -DOUTPUT_FILENAME=project_tet10_dyn.e -P
    ${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake)
set_tests_properties(${testName}_Serial PROPERTIES LABELS "LCM;Tpetra;Forward")

# Lumped mass matrix: gold-based (a lumped projection does not reproduce a
# linear field exactly), still guards the GitHub #11 first-frame regression.
add_test(
  NAME ${testName}_Lumped_Serial
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME=project_tet10_dyn_lumped
    -DTEST_ARGS=project_tet10_dyn_lumped.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME=project_tet10_dyn_lumped_gold.e -DOUTPUT_FILENAME=project_tet10_dyn_lumped.e -P
    ${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake)
set_tests_properties(${testName}_Lumped_Serial PROPERTIES LABELS "LCM;Tpetra;Forward")
