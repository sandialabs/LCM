#maximum 32 bytes uploaded to CDash for each passed test
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE "32")

# CTest's built-in error patterns do not match linker failures, so a failed
# link was reported as "0 Compiler errors" and snl_build() returned success
# with the executable missing. That is how a broken CEE build went unnoticed
# while ctest then ran a short suite. Count linker failures as errors so
# snl_build()'s NERRS check trips.
set(CTEST_CUSTOM_ERROR_MATCH
    ${CTEST_CUSTOM_ERROR_MATCH}
    "collect2: error:"
    "undefined reference to"
    "ld returned [0-9]+ exit status"
    "No rule to make target"
    "cannot find -l")
