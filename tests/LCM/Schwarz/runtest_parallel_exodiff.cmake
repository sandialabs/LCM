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
  SET(EXODIFF_TEST0_0 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff -m ${OUTPUT_FILENAME0}.4.0 ${REF_FILENAME0}.4.0)
  SET(EXODIFF_TEST0_1 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff -m ${OUTPUT_FILENAME0}.4.1 ${REF_FILENAME0}.4.1)
  SET(EXODIFF_TEST0_2 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff -m ${OUTPUT_FILENAME0}.4.2 ${REF_FILENAME0}.4.2)
  SET(EXODIFF_TEST0_3 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff -m ${OUTPUT_FILENAME0}.4.3 ${REF_FILENAME0}.4.3)
  SET(EXODIFF_TEST1_0 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff -m ${OUTPUT_FILENAME1}.4.0 ${REF_FILENAME1}.4.0)
  SET(EXODIFF_TEST1_1 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff -m ${OUTPUT_FILENAME1}.4.1 ${REF_FILENAME1}.4.1)
  SET(EXODIFF_TEST1_2 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff -m ${OUTPUT_FILENAME1}.4.2 ${REF_FILENAME1}.4.2)
  SET(EXODIFF_TEST1_3 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff -m ${OUTPUT_FILENAME1}.4.3 ${REF_FILENAME1}.4.3)
ELSE()
  SET(EXODIFF_TEST0_0 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff ${OUTPUT_FILENAME0}.4.0 ${REF_FILENAME0}.4.0)
  SET(EXODIFF_TEST0_1 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff ${OUTPUT_FILENAME0}.4.1 ${REF_FILENAME0}.4.1)
  SET(EXODIFF_TEST0_2 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff ${OUTPUT_FILENAME0}.4.2 ${REF_FILENAME0}.4.2)
  SET(EXODIFF_TEST0_3 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff ${OUTPUT_FILENAME0}.4.3 ${REF_FILENAME0}.4.3)
  SET(EXODIFF_TEST1_0 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff ${OUTPUT_FILENAME1}.4.0 ${REF_FILENAME1}.4.0)
  SET(EXODIFF_TEST1_1 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff ${OUTPUT_FILENAME1}.4.1 ${REF_FILENAME1}.4.1)
  SET(EXODIFF_TEST1_2 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff ${OUTPUT_FILENAME1}.4.2 ${REF_FILENAME1}.4.2)
  SET(EXODIFF_TEST1_3 ${SEACAS_EXODIFF} -i -f ${TEST_NAME}.exodiff ${OUTPUT_FILENAME1}.4.3 ${REF_FILENAME1}.4.3)
ENDIF()

message("Running the command:")
message("${EXODIFF_TEST0_0}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST0_0}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST0_1}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST0_1}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST0_2}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST0_2}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST0_3}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST0_3}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST1_0}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST1_0}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST1_1}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST1_1}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST1_2}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST1_2}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST1_3}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST1_3}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()
