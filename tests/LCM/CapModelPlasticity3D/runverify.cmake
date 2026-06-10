# Run an Albany cap-model verification case and compare the response
# against the independent reference implementation (cap_reference.py).
#
# Required defs: TEST_PROG (Albany), INPUT (yaml), EXO (output exodus),
#                PATHMODE (hydrostatic|confined|triaxial), PYTHON

execute_process(COMMAND ${TEST_PROG} ${INPUT}
                RESULT_VARIABLE ALBANY_RES)
if (NOT ALBANY_RES EQUAL 0)
  message(FATAL_ERROR "Albany failed: ${ALBANY_RES}")
endif()

execute_process(COMMAND ${PYTHON} cap_verify.py ${EXO} ${PATHMODE}
                RESULT_VARIABLE VERIFY_RES)
if (NOT VERIFY_RES EQUAL 0)
  message(FATAL_ERROR "Verification vs reference implementation failed")
endif()
