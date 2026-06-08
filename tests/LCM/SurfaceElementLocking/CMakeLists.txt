#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

if(SEACAS_EXODIFF)
  # Create a symlink to exodiff
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EXODIFF}
                          ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${AlbanyPath}
                          ${CMAKE_CURRENT_BINARY_DIR}/Albany)

  # input files
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputNoVolJ.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputNoVolJ.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputVolJ.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputVolJ.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAveragePressure.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAveragePressure.yaml COPYONLY)

  # material files
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/1materialNoVolJ.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/1materialNoVolJ.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/1materialVolJ.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/1materialVolJ.yaml COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/1materialAveragePressure.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/1materialAveragePressure.yaml COPYONLY)

  # mesh file
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface1ElemMeshRev.g
                 ${CMAKE_CURRENT_BINARY_DIR}/surface1ElemMeshRev.g COPYONLY)

  # exodiff files
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NoVolJ.exodiff
                 ${CMAKE_CURRENT_BINARY_DIR}/NoVolJ.exodiff COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/VolJ.exodiff
                 ${CMAKE_CURRENT_BINARY_DIR}/VolJ.exodiff COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AveragePressure.exodiff
                 ${CMAKE_CURRENT_BINARY_DIR}/AveragePressure.exodiff COPYONLY)

  # gold files
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/outNoVolJ.gold.e
                 ${CMAKE_CURRENT_BINARY_DIR}/outNoVolJ.gold.e COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/outVolJ.gold.e
                 ${CMAKE_CURRENT_BINARY_DIR}/outVolJ.gold.e COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/outAveragePressure.gold.e
    ${CMAKE_CURRENT_BINARY_DIR}/outAveragePressure.gold.e COPYONLY)

  # cmake runtest file
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
                 ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)

  # 1. Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # 1. Create the test with this name and standard executable
  if(NOT ALBANY_PARALLEL_ONLY)
    # test 1 - NoVolJ
    set(OUTFILE "outNoVolJ.e")
    set(REF_FILE "outNoVolJ.gold.e")
    add_test(
      NAME ${testName}_NoVolJ
      COMMAND
        ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME=NoVolJ
        -DTEST_ARGS=inputNoVolJ.yaml -DMPIMNP=1
        -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
        -DOUTPUT_FILENAME=${OUTFILE}
        -DREF_FILENAME=${REF_FILE}
        -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
    set_tests_properties(${testName}_NoVolJ PROPERTIES LABELS
                                                       "LCM;Tpetra;Forward")
    # test 2 - VolJ
    set(OUTFILE "outVolJ.e")
    set(REF_FILE "outVolJ.gold.e")
    add_test(
      NAME ${testName}_VolJ
      COMMAND
        ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME=VolJ
        -DTEST_ARGS=inputVolJ.yaml -DMPIMNP=1
        -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
        -DOUTPUT_FILENAME=${OUTFILE}
        -DREF_FILENAME=${REF_FILE}
        -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
    set_tests_properties(${testName}_VolJ PROPERTIES LABELS
                                                     "LCM;Tpetra;Forward")
    # test 3 - AveragePressure
    set(OUTFILE "outAveragePressure.e")
    set(REF_FILE "outAveragePressure.gold.e")
    add_test(
      NAME ${testName}_AveragePressure
      COMMAND
        ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}"
        -DTEST_NAME=AveragePressure -DTEST_ARGS=inputAveragePressure.yaml
        -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
        -DOUTPUT_FILENAME=${OUTFILE}
        -DREF_FILENAME=${REF_FILE}
        -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P
        ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
    set_tests_properties(${testName}_AveragePressure
                         PROPERTIES LABELS "LCM;Tpetra;Forward")

  endif()
endif()
