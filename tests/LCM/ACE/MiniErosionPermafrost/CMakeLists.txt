#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#
# MiniErosion with the Permafrost (cap plasticity) mechanical material in
# place of J2Erosion. The mesh, thermal physics, and run schedule are
# shared with the MiniErosion test (files configured from its source
# directory); only the mechanical decks differ.

set(MINI_EROSION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../MiniErosion)

# Shared with MiniErosion: thermal decks, mesh, depth profile.
configure_file(${MINI_EROSION_DIR}/thermal_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation.yaml COPYONLY)
configure_file(${MINI_EROSION_DIR}/materials_thermal_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_thermal_denudation.yaml COPYONLY)
configure_file(${MINI_EROSION_DIR}/grid/cuboid_denudation.g
               ${CMAKE_CURRENT_BINARY_DIR}/grid/cuboid_denudation.g COPYONLY)
configure_file(${MINI_EROSION_DIR}/external_files/depth.txt
               ${CMAKE_CURRENT_BINARY_DIR}/external_files/depth.txt COPYONLY)
configure_file(${MINI_EROSION_DIR}/runtest_serial.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_serial.cmake COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_permafrost.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/coupled_permafrost.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_permafrost.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_permafrost.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_mechanical_permafrost.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_mechanical_permafrost.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/permafrost.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/permafrost.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/permafrost_gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/permafrost_gold.e COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Serial test: denudation with element death through the cap-based
# failure criteria (20 steps). The ACE solver writes a single coupled
# Exodus output (permafrost.e); one exodiff against the gold file.
set(OUTFILE "permafrost.e")
set(REF_FILE "permafrost_gold.e")
add_test(
  NAME ACE_${testName}_Denudation_Serial
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME=permafrost
    -DTEST_ARGS=coupled_permafrost.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE} -P
    ${CMAKE_CURRENT_BINARY_DIR}/runtest_serial.cmake)
set_tests_properties(ACE_${testName}_Denudation_Serial PROPERTIES LABELS "LCM;Tpetra;Forward")
