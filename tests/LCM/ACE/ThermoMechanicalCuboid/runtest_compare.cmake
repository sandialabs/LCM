# Run both I/O and in-memory ACE solvers.
# Verifies that both complete successfully on a thermo-mechanical problem
# with Neumann BCs and the TrapezoidRule mechanical solver.

# --- Step 1: Run I/O solver ---
message("=== Running I/O solver ===")
message("${TEST_PROG} coupled_io.yaml")

EXECUTE_PROCESS(COMMAND ${TEST_PROG} coupled_io.yaml
                RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
  message(FATAL_ERROR "I/O solver failed to run")
endif()

# --- Step 2: Run IM solver ---
message("=== Running IM solver ===")
message("${TEST_PROG} coupled_im.yaml")

EXECUTE_PROCESS(COMMAND ${TEST_PROG} coupled_im.yaml
                RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
  message(FATAL_ERROR "IM solver failed to run")
endif()

message("=== Both I/O and IM solvers completed successfully ===")
