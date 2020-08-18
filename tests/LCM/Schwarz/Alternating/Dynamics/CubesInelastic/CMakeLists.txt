#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.g
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.g
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.gold.e.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.gold.e.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.gold.e.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.gold.e.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.gold.e.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.gold.e.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.gold.e.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.gold.e.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.gold.e.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.gold.e.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.gold.e.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.gold.e.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.gold.e.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.gold.e.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.gold.e.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.gold.e.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboids.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cuboids.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_00.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_00.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_01.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_01.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CubesInelastic.exodiff
	       ${CMAKE_CURRENT_BINARY_DIR}/CubesInelastic.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CubesInelasticParallel.exodiff
	       ${CMAKE_CURRENT_BINARY_DIR}/CubesInelasticParallel.exodiff COPYONLY)
if(ALBANY_DTK)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${runtest_parallel_exodiff.cmake}
                          ${CMAKE_CURRENT_BINARY_DIR}/runtest_parallel_exodiff.cmake)
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${AlbanyPath}
                        ${CMAKE_CURRENT_BINARY_DIR}/Albany)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${runtest_serial_exodiff.cmake}
                        ${CMAKE_CURRENT_BINARY_DIR}/runtest_serial_exodiff.cmake)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Add the tests
set(OUTFILE0 "cuboid_00.e")
set(REF_FILE0 "cuboid_00.gold.e")
set(OUTFILE1 "cuboid_01.e")
set(REF_FILE1 "cuboid_01.gold.e")
add_test(NAME Schwarz_Alternating_Dynamics_${testName}
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME=CubesInelastic
    -DTEST_ARGS=cuboids.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME0=${REF_FILE0} -DOUTPUT_FILENAME0=${OUTFILE0} 
    -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1} 
    -P ${runtest_serial_exodiff.cmake})
set_tests_properties(Schwarz_Alternating_Dynamics_${testName}
                   PROPERTIES LABELS "LCM;Tpetra;Forward")

if(ALBANY_DTK)
  add_test(NAME Schwarz_Alternating_Dynamics_${testName}_Parallel
    COMMAND
    ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}" -DTEST_NAME=CubesInelasticParallel
      -DTEST_ARGS=cuboids.yaml -DMPIMNP=4 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
      -DREF_FILENAME0=${REF_FILE0} -DOUTPUT_FILENAME0=${OUTFILE0} 
      -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1} 
      -P ${runtest_parallel_exodiff.cmake})
  set_tests_properties(Schwarz_Alternating_Dynamics_${testName}_Parallel
                       PROPERTIES LABELS "LCM;Tpetra;Forward")
endif()
