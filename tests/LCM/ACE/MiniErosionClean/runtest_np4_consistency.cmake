# ACE element-death PARALLEL-CONSISTENCY driver.
#
# Runs the coupled thermo-mechanical erosion case on 4 ranks, merges the
# per-rank Exodus output with epu, then asserts (via compare_death_positions.py)
# that the FINAL eroded cell set is identical to the serial reference gold --
# compared BY ELEMENT CENTROID POSITION, not by frame index or element id.
#
# Why not a plain exodiff against the serial gold?  The np4 run takes a
# partition-dependent solver cutback, so it has a different NUMBER of output
# frames and a slightly different end time than serial, even though the death
# SET is identical. A frame-by-frame exodiff would spuriously fail. The Python
# check looks only at the last time step and matches cells by position, which is
# invariant to both the frame-count difference and epu's element renumbering.
#
# Expected variables:
#   TEST_PROG        - mpirun-wrapped Albany (parallel launcher)
#   TEST_ARGS        - input yaml (coupled_denudation.yaml)
#   MPIMNP           - number of ranks (4)
#   SEACAS_EPU       - path to epu
#   OUTPUT_FILENAME  - per-rank / merged exodus base name (denudation.e)
#   REF_FILENAME     - serial reference gold (denudation_gold.e)
#   PYTHON_EXE       - python3 interpreter
#   COMPARE_SCRIPT   - compare_death_positions.py

# 1. Run the parallel program (TEST_PROG already wraps mpirun).

message("Running the command:")
message("${TEST_PROG} ${TEST_ARGS}")

execute_process(COMMAND ${TEST_PROG} ${TEST_ARGS}
                RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
  message(FATAL_ERROR "Albany didn't run: test failed")
endif()

# 2. Merge the per-rank Exodus output with epu -> ${OUTPUT_FILENAME}.

if(NOT SEACAS_EPU)
  message(FATAL_ERROR "Cannot find epu")
endif()

set(EPU_COMMAND ${SEACAS_EPU} -auto ${OUTPUT_FILENAME}.${MPIMNP}.0)
message("Running the command:")
message("${EPU_COMMAND}")

execute_process(COMMAND ${EPU_COMMAND}
                RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
  message(FATAL_ERROR "epu failed")
endif()

# 3. Position-based dead-cell-set comparison against the serial gold.

set(COMPARE_COMMAND ${PYTHON_EXE} ${COMPARE_SCRIPT} ${REF_FILENAME} ${OUTPUT_FILENAME})
message("Running the command:")
message("${COMPARE_COMMAND}")

execute_process(COMMAND ${COMPARE_COMMAND}
                RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
  message(FATAL_ERROR "Parallel-consistency test failed: np4 erosion set differs from serial")
endif()
