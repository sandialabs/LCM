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
  SET(EXODIFF_TEST00 ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME1}.exodiff -steps -1 ${OUTPUT_FILENAME1}.4.0 ${REF_FILENAME1}.4.0)
  SET(EXODIFF_TEST01 ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME1}.exodiff -steps -1 ${OUTPUT_FILENAME1}.4.1 ${REF_FILENAME1}.4.1)
  SET(EXODIFF_TEST02 ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME1}.exodiff -steps -1 ${OUTPUT_FILENAME1}.4.2 ${REF_FILENAME1}.4.2)
  SET(EXODIFF_TEST03 ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME1}.exodiff -steps -1 ${OUTPUT_FILENAME1}.4.3 ${REF_FILENAME1}.4.3)
  SET(EXODIFF_TEST10 ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME2}.exodiff -steps -1 ${OUTPUT_FILENAME2}.4.0 ${REF_FILENAME2}.4.0)
  SET(EXODIFF_TEST11 ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME2}.exodiff -steps -1 ${OUTPUT_FILENAME2}.4.1 ${REF_FILENAME2}.4.1)
  SET(EXODIFF_TEST12 ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME2}.exodiff -steps -1 ${OUTPUT_FILENAME2}.4.2 ${REF_FILENAME2}.4.2)
  SET(EXODIFF_TEST13 ${SEACAS_EXODIFF} -i -m -f ${TEST_NAME2}.exodiff -steps -1 ${OUTPUT_FILENAME2}.4.3 ${REF_FILENAME2}.4.3)
ELSE()
  SET(EXODIFF_TEST00 ${SEACAS_EXODIFF} -i -f ${TEST_NAME1}.exodiff -steps -1 ${OUTPUT_FILENAME1}.4.0 ${REF_FILENAME1}.4.0)
  SET(EXODIFF_TEST01 ${SEACAS_EXODIFF} -i -f ${TEST_NAME1}.exodiff -steps -1 ${OUTPUT_FILENAME1}.4.1 ${REF_FILENAME1}.4.1)
  SET(EXODIFF_TEST02 ${SEACAS_EXODIFF} -i -f ${TEST_NAME1}.exodiff -steps -1 ${OUTPUT_FILENAME1}.4.2 ${REF_FILENAME1}.4.2)
  SET(EXODIFF_TEST03 ${SEACAS_EXODIFF} -i -f ${TEST_NAME1}.exodiff -steps -1 ${OUTPUT_FILENAME1}.4.3 ${REF_FILENAME1}.4.3)
  SET(EXODIFF_TEST10 ${SEACAS_EXODIFF} -i -f ${TEST_NAME2}.exodiff -steps -1 ${OUTPUT_FILENAME2}.4.0 ${REF_FILENAME2}.4.0)
  SET(EXODIFF_TEST11 ${SEACAS_EXODIFF} -i -f ${TEST_NAME2}.exodiff -steps -1 ${OUTPUT_FILENAME2}.4.1 ${REF_FILENAME2}.4.1)
  SET(EXODIFF_TEST12 ${SEACAS_EXODIFF} -i -f ${TEST_NAME2}.exodiff -steps -1 ${OUTPUT_FILENAME2}.4.2 ${REF_FILENAME2}.4.2)
  SET(EXODIFF_TEST13 ${SEACAS_EXODIFF} -i -f ${TEST_NAME2}.exodiff -steps -1 ${OUTPUT_FILENAME2}.4.3 ${REF_FILENAME2}.4.3)
ENDIF()

message("Running the command:")
message("${EXODIFF_TEST00}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST00}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST01}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST01}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST02}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST02}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST03}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST03}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST10}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST10}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST11}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST11}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST12}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST12}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

message("Running the command:")
message("${EXODIFF_TEST13}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST13}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

