# 1. Run the program (TEST_PROG already wraps mpirun for parallel runs)

message("Running the command:")
message("${TEST_PROG} " " ${TEST_ARGS}")

EXECUTE_PROCESS(COMMAND ${TEST_PROG} ${TEST_ARGS}
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Albany didn't run: test failed")
endif()

# 2. Merge per-rank Exodus output with epu

if (NOT SEACAS_EPU)
  message(FATAL_ERROR "Cannot find epu")
endif()

SET(EPU_COMMAND ${SEACAS_EPU} -auto ${OUTPUT_FILENAME}.${MPIMNP}.0)

message("Running the command:")
message("${EPU_COMMAND}")

EXECUTE_PROCESS(COMMAND ${EPU_COMMAND}
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "epu failed")
endif()

# 3. Find and run exodiff against the (serial) gold

if (NOT SEACAS_EXODIFF)
  message(FATAL_ERROR "Cannot find exodiff")
endif()

SET(EXODIFF_TEST ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME}.exodiff -steps -1 ${OUTPUT_FILENAME} ${REF_FILENAME})

message("Running the command:")
message("${EXODIFF_TEST}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST}
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()
