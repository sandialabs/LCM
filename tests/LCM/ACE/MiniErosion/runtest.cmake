# 1. Run the program and generate the exodus output

message("Running the command:")
message("${TEST_PROG} " " ${TEST_ARGS}")

EXECUTE_PROCESS(COMMAND ${TEST_PROG} ${TEST_ARGS}
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Albany didn't run: test failed")
endif()


# 2. Find and run exodiff

if (NOT SEACAS_EXODIFF)
  message(FATAL_ERROR "Cannot find exodiff")
endif()

if(DEFINED MPIMNP AND ${MPIMNP} GREATER 1)
  SET(EXODIFF_TEST0 ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME}.exodiff -steps -1 ${OUTPUT_FILENAME}.4.0 ${REF_FILENAME}.4.0)
  SET(EXODIFF_TEST1 ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME}.exodiff -steps -1 ${OUTPUT_FILENAME}.4.1 ${REF_FILENAME}.4.1)
  SET(EXODIFF_TEST2 ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME}.exodiff -steps -1 ${OUTPUT_FILENAME}.4.2 ${REF_FILENAME}.4.2)
  SET(EXODIFF_TEST3 ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME}.exodiff -steps -1 ${OUTPUT_FILENAME}.4.3 ${REF_FILENAME}.4.3)
ELSE()
  SET(EXODIFF_TEST0 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff -steps -1 ${OUTPUT_FILENAME}.4.0 ${REF_FILENAME}.4.0)
  SET(EXODIFF_TEST1 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff -steps -1 ${OUTPUT_FILENAME}.4.1 ${REF_FILENAME}.4.1)
  SET(EXODIFF_TEST2 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff -steps -1 ${OUTPUT_FILENAME}.4.2 ${REF_FILENAME}.4.2)
  SET(EXODIFF_TEST3 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff -steps -1 ${OUTPUT_FILENAME}.4.3 ${REF_FILENAME}.4.3)
ENDIF()
#if(DEFINED MPIMNP AND ${MPIMNP} GREATER 1)
#  SET(EXODIFF_TEST ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME}.exodiff ${OUTPUT_FILENAME} ${REF_FILENAME})
#ELSE()
#  SET(EXODIFF_TEST ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff ${OUTPUT_FILENAME} ${REF_FILENAME})
#ENDIF()

message("Running the command:")
message("${EXODIFF_TEST0}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST0}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST1}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST1}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST2}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST2}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST3}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST3}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()
