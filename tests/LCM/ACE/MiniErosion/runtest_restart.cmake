# Restart consistency: a run continued from the middle of another run must
# reach the same final state as the run it continues.
#
# Part 1 is the reference run, writing a restartable output file. Part 2
# restarts from its tenth frame and integrates to the same final time. The two
# final states are then compared.
#
# This is a self-comparison -- no gold file. What it guards is that the restart
# reloads everything the models carry from step to step: the solution and its
# time derivatives, and the element state (stress, plastic strain, failure
# history, thermal state). Dropping any of them lets the continued run keep
# going and look plausible while following a different trajectory, which is
# exactly what this test is here to catch.

message("Running the command:")
message("${TEST_PROG} " " ${BASE_ARGS}")

EXECUTE_PROCESS(COMMAND ${TEST_PROG} ${BASE_ARGS}
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Albany didn't run the reference case: test failed")
endif()

message("Running the command:")
message("${TEST_PROG} " " ${RESTART_ARGS}")

EXECUTE_PROCESS(COMMAND ${TEST_PROG} ${RESTART_ARGS}
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Albany didn't run the restarted case: test failed")
endif()

if (NOT SEACAS_EXODIFF)
  message(FATAL_ERROR "Cannot find exodiff")
endif()

# On several ranks each run leaves one Exodus file per rank; merge both sets
# before comparing. Same decomposition for both runs, so epu numbers them the
# same way, but compare with -m anyway as the other parallel ACE test does.
set(EXODIFF_MAP_FLAG "")
if(MPIMNP AND MPIMNP GREATER 1)
  set(EXODIFF_MAP_FLAG "-m")
  if (NOT SEACAS_EPU)
    message(FATAL_ERROR "Cannot find epu")
  endif()
  foreach(MERGE_FILE ${BASE_FILENAME} ${RESTART_FILENAME})
    EXECUTE_PROCESS(COMMAND ${SEACAS_EPU} -auto ${MERGE_FILE}.${MPIMNP}.0
                    RESULT_VARIABLE HAD_ERROR)
    if(HAD_ERROR)
      message(FATAL_ERROR "epu failed on ${MERGE_FILE}: test failed")
    endif()
  endforeach()
endif()

# Compare the final frames only. The restarted run's FIRST frame is written
# before it has solved anything, so the thermal quantities that the thermal fill
# derives from temperature (density, heat capacity, thermal inertia, ice
# saturation) still sit at their initialization values there; they are correct
# from the first solved frame on. Every later frame agrees, and the final frame
# is the one that carries the accumulated difference if the restart lost state.
SET(EXODIFF_TEST ${SEACAS_EXODIFF} -i ${EXODIFF_MAP_FLAG} -f ${TEST_NAME}.exodiff
    -explicit last:last ${BASE_FILENAME} ${RESTART_FILENAME})

message("Running the command:")
message("${EXODIFF_TEST}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST}
    RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()
