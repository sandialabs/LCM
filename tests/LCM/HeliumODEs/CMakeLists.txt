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
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HeBubbles.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/HeBubbles.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HeBubblesDecay.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/HeBubblesDecay.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hexOneElement.g
                 ${CMAKE_CURRENT_BINARY_DIR}/hexOneElement.g COPYONLY)

  # material files
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/materialsScaledPlasticity.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/materialsScaledPlasticity.yaml COPYONLY)

  # exodiff files
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HeBubbles.exodiff
                 ${CMAKE_CURRENT_BINARY_DIR}/HeBubbles.exodiff COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HeBubblesDecay.exodiff
                 ${CMAKE_CURRENT_BINARY_DIR}/HeBubblesDecay.exodiff COPYONLY)

  # gold files
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HeBubbles.gold.e
                 ${CMAKE_CURRENT_BINARY_DIR}/HeBubbles.gold.e COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HeBubblesDecay.gold.e
                 ${CMAKE_CURRENT_BINARY_DIR}/HeBubblesDecay.gold.e COPYONLY)

  # cmake runtest file
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
                 ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)

  # 1. Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # 1. Create the test with this name and standard executable
  if(NOT ALBANY_PARALLEL_ONLY)
    # test 1: HeBubbles
    set(OUTFILE "HeBubbles.e")
    set(REF_FILE "HeBubbles.gold.e")
    add_test(
      NAME ${testName}_HeBubbles
      COMMAND
        ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}"
        -DTEST_NAME=HeBubbles -DTEST_ARGS=HeBubbles.yaml -DMPIMNP=1
        -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -DREF_FILENAME=${REF_FILE}
        -DOUTPUT_FILENAME=${OUTFILE} -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P
        ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
    set_tests_properties(${testName}_HeBubbles PROPERTIES LABELS
                                                          "LCM;Tpetra;Forward")
    if(NOT DISABLE_LCM_EXODIFF_SENSITIVE_TESTS)
      # test 2: HeBubblesDecay
      set(OUTFILE "HeBubblesDecay.e")
      set(REF_FILE "HeBubblesDecay.gold.e")
      add_test(
        NAME ${testName}_HeBubblesDecay
        COMMAND
          ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}"
          -DTEST_NAME=HeBubblesDecay -DTEST_ARGS=HeBubblesDecay.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -DREF_FILENAME=${REF_FILE}
          -DOUTPUT_FILENAME=${OUTFILE} -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
      set_tests_properties(${testName}_HeBubblesDecay
                           PROPERTIES LABELS "LCM;Tpetra;Forward")
    endif()
  endif()
endif()
