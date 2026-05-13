# ThermoMechanicalCuboid: in-memory ACE coupling smoke test on a
# permafrost cuboid with Neumann BCs (geothermal heat flux), Dirichlet
# BCs, and TrapezoidRule mechanical solver. No erosion.

# Copy input files
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

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_test(ACE_${testName}_IM ${SerialAlbany.exe} coupled_im.yaml)
set_tests_properties(ACE_${testName}_IM PROPERTIES LABELS "LCM;Tpetra;Forward")
