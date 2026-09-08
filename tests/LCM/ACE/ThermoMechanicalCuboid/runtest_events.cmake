# Regression test for the event intervals of the ACE thermo-mechanical
# solver. The deck runs for 7200 s with a 900 s step and two event
# intervals, [1800, 2700] at 300 s and [4500, 5400] at 450 s. The solver
# must land exactly on every interval boundary, use the interval's step
# inside it, and resume the outside step on leaving. All times are exact
# in binary floating point. The solver prints them in scientific notation
# with a precision that varies along the run, so trailing zeros of the
# mantissa are stripped before the comparison.
#
# Before the fix this deck could not run at all: the final times and the
# time steps were read from the initial-times file, and the size check
# was mis-parenthesized so that it failed for more than one interval.

message("Running the command:")
message("${TEST_PROG} ${TEST_ARGS}")

execute_process(COMMAND ${TEST_PROG} ${TEST_ARGS}
                OUTPUT_FILE events.log
                RESULT_VARIABLE HAD_ERROR)
if(HAD_ERROR)
  message(FATAL_ERROR "Albany didn't run: test failed")
endif()

file(READ events.log LOG)
string(REGEX MATCHALL "Time               :[^\n]*" TIME_LINES "${LOG}")
set(TIMES "")
foreach(line IN LISTS TIME_LINES)
  string(REGEX REPLACE "Time               :" "" t "${line}")
  string(STRIP "${t}" t)
  string(REGEX REPLACE "0+e" "e" t "${t}")
  list(APPEND TIMES "${t}")
endforeach()

set(EXPECTED 0.e+00 9.e+02 1.8e+03 2.1e+03 2.4e+03 2.7e+03 3.6e+03 4.5e+03 4.95e+03 5.4e+03 6.3e+03)

if(NOT "${TIMES}" STREQUAL "${EXPECTED}")
  message("Expected step times: ${EXPECTED}")
  message("Actual step times:   ${TIMES}")
  message(FATAL_ERROR "Event intervals produced the wrong step sequence: test failed")
endif()
message("Step sequence matches the two event intervals.")
