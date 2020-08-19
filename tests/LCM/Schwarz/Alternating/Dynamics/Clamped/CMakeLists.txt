#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_00.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_00.yaml COPYONLY)
       configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Clamped.exodiff
	       ${CMAKE_CURRENT_BINARY_DIR}/Clamped.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_01.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_01.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/clamped.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material_00.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material_00.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material_01.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material_01.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_00.gold.e.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_00.gold.e.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_00.gold.e.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_00.gold.e.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_00.gold.e.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_00.gold.e.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_00.gold.e.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_00.gold.e.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_01.gold.e.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_01.gold.e.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_01.gold.e.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_01.gold.e.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_01.gold.e.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_01.gold.e.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_01.gold.e.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_01.gold.e.4.3 COPYONLY)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${AlbanyPath}
                        ${CMAKE_CURRENT_BINARY_DIR}/Albany)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${runtest_parallel_exodiff.cmake}
                        ${CMAKE_CURRENT_BINARY_DIR}/runtest_parallel_exodiff.cmake)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
if(ALBANY_DTK)
  set(OUTFILE0 "clamped_00.e")
  set(REF_FILE0 "clamped_00.gold.e")
  set(OUTFILE1 "clamped_01.e")
  set(REF_FILE1 "clamped_01.gold.e")
  add_test(NAME Schwarz_Alternating_Dynamics_${testName}_Parallel
    COMMAND
    ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}" -DTEST_NAME=Clamped
      -DTEST_ARGS=clamped.yaml -DMPIMNP=4 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
      -DREF_FILENAME0=${REF_FILE0} -DOUTPUT_FILENAME0=${OUTFILE0} 
      -DREF_FILENAME1=${REF_FILE1} -DOUTPUT_FILENAME1=${OUTFILE1} 
      -P ${runtest_parallel_exodiff.cmake})
  set_tests_properties(Schwarz_Alternating_Dynamics_${testName}_Parallel
                       PROPERTIES LABELS "LCM;Tpetra;Forward")

endif()
