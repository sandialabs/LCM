#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file "license.txt" in the top-level Albany directory
#

if(SEACAS_EXODIFF AND SEACAS_ALGEBRA)
  # Create a symlink to exodiff
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EXODIFF}
                          ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_ALGEBRA}
                          ${CMAKE_CURRENT_BINARY_DIR}/algebra)
  if(ALBANY_IFPACK2)
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${AlbanyPath}
                            ${CMAKE_CURRENT_BINARY_DIR}/Albany)
  endif()

  if(ALBANY_IFPACK2)
    # input files
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputNoVolJ.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputNoVolJ.yaml COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputVolJ.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputVolJ.yaml COPYONLY)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/inputAveragePressure.yaml
      ${CMAKE_CURRENT_BINARY_DIR}/inputAveragePressure.yaml COPYONLY)
  endif()

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

  # algebra file
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/alg.i
                 ${CMAKE_CURRENT_BINARY_DIR}/alg.i COPYONLY)

  # exodiff files
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NoVolJ.exodiff
                 ${CMAKE_CURRENT_BINARY_DIR}/NoVolJ.exodiff COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/VolJ.exodiff
                 ${CMAKE_CURRENT_BINARY_DIR}/VolJ.exodiff COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AveragePressure.exodiff
                 ${CMAKE_CURRENT_BINARY_DIR}/AveragePressure.exodiff COPYONLY)

  # gold files
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/outNoVolJ.alg.gold.e
                 ${CMAKE_CURRENT_BINARY_DIR}/outNoVolJ.alg.gold.e COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/outVolJ.alg.gold.e
                 ${CMAKE_CURRENT_BINARY_DIR}/outVolJ.alg.gold.e COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/outAveragePressure.alg.gold.e
    ${CMAKE_CURRENT_BINARY_DIR}/outAveragePressure.alg.gold.e COPYONLY)

  if(ALBANY_IFPACK2)
    # cmake runtest file
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
                   ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)
  endif()
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_algebra_NoVolJ.sh
                 ${CMAKE_CURRENT_BINARY_DIR}/run_algebra_NoVolJ.sh COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_algebra_VolJ.sh
                 ${CMAKE_CURRENT_BINARY_DIR}/run_algebra_VolJ.sh COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/run_algebra_AveragePressure.sh
    ${CMAKE_CURRENT_BINARY_DIR}/run_algebra_AveragePressure.sh COPYONLY)

  # 1. Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # 1. Create the test with this name and standard executable
  if(NOT ALBANY_PARALLEL_ONLY)
    if(ALBANY_IFPACK2)
      # test 1 - NoVolJ
      set(OUTFILE "NoVolJ.e")
      set(REF_FILE "outNoVolJ.alg.gold.e")
      set(ALG_OUTFILE "outNoVolJ.alg.e")
      set(ALG_FILE "run_algebra_NoVolJ.sh")
      add_test(
        NAME ${testName}_NoVolJ
        COMMAND
          ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME=NoVolJ
          -DTEST_ARGS=inputNoVolJ.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -DSEACAS_ALGEBRA=${SEACAS_ALGEBRA}
          -DALGEBRA_FILE=${ALG_FILE} -DREF_FILENAME=${REF_FILE}
          -DOUTPUT_FILENAME=${ALG_OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P
          ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
      set_tests_properties(${testName}_NoVolJ PROPERTIES LABELS
                                                         "LCM;Tpetra;Forward")
      # test 2 - VolJ
      set(OUTFILE "VolJ.e")
      set(REF_FILE "outVolJ.alg.gold.e")
      set(ALG_OUTFILE "outVolJ.alg.e")
      set(ALG_FILE "run_algebra_VolJ.sh")
      add_test(
        NAME ${testName}_VolJ
        COMMAND
          ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME=VolJ
          -DTEST_ARGS=inputVolJ.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -DSEACAS_ALGEBRA=${SEACAS_ALGEBRA}
          -DALGEBRA_FILE=${ALG_FILE} -DREF_FILENAME=${REF_FILE}
          -DOUTPUT_FILENAME=${ALG_OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P
          ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
      set_tests_properties(${testName}_VolJ PROPERTIES LABELS
                                                       "LCM;Tpetra;Forward")
      # test 3 - AveragePressure
      set(OUTFILE "AveragePressure.e")
      set(REF_FILE "outAveragePressure.alg.gold.e")
      set(ALG_OUTFILE "outAveragePressure.alg.e")
      set(ALG_FILE "run_algebra_AveragePressure.sh")
      add_test(
        NAME ${testName}_AveragePressure
        COMMAND
          ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}"
          -DTEST_NAME=AveragePressure -DTEST_ARGS=inputAveragePressure.yaml
          -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DSEACAS_ALGEBRA=${SEACAS_ALGEBRA} -DALGEBRA_FILE=${ALG_FILE}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${ALG_OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P
          ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
      set_tests_properties(${testName}_AveragePressure
                           PROPERTIES LABELS "LCM;Tpetra;Forward")

    endif()
  endif()
endif(SEACAS_EXODIFF AND SEACAS_ALGEBRA)
