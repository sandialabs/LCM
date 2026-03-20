# ThermoMechanicalCuboid: Runs both I/O-based and in-memory ACE solvers
# on a permafrost cuboid with Neumann BCs (geothermal heat flux),
# Dirichlet BCs, and TrapezoidRule mechanical solver. No erosion.

# Copy input files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_io.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/coupled_io.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_im.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/coupled_im.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/thermal.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_thermal.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_thermal.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_mechanical.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_mechanical.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.g
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/depth.txt
               ${CMAKE_CURRENT_BINARY_DIR}/depth.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_compare.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_compare.cmake COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Serial test: run both I/O and IM solvers, verify both complete.
set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest_compare.cmake)
add_test(
  NAME ACE_${testName}_IO_vs_IM
  COMMAND
    ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}"
    -P ${runtest.cmake})
set_tests_properties(ACE_${testName}_IO_vs_IM PROPERTIES LABELS "LCM;Tpetra;Forward")
