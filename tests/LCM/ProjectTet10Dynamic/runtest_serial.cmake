# 1. Run the program and generate the exodus output

message("Running the command:")
message("${TEST_PROG} " " ${TEST_ARGS}")

EXECUTE_PROCESS(COMMAND ${TEST_PROG} ${TEST_ARGS}
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Albany didn't run: test failed")
endif()


# 2. Find and run exodiff
#
# M4: the ACE thermo-mechanical solver writes a single coupled Exodus
# output file (both physics' fields on the shared mesh), so the test
# does one exodiff instead of the former separate thermal/mechanical
# comparisons.

if (NOT SEACAS_EXODIFF)
  message(FATAL_ERROR "Cannot find exodiff")
endif()

if(DEFINED MPIMNP AND ${MPIMNP} GREATER 1)
  SET(EXODIFF_TEST ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME}.exodiff -steps -1 ${OUTPUT_FILENAME} ${REF_FILENAME})
ELSE()
  SET(EXODIFF_TEST ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff -steps -1 ${OUTPUT_FILENAME} ${REF_FILENAME})
ENDIF()

message("Running the command:")
message("${EXODIFF_TEST}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST}
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()
