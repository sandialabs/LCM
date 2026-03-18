# LCM Build Script
# Called via: ctest -S lcm_build.cmake
#   -DPACKAGE:STRING=trilinos|lcm
#   -DBUILD_THREADS:STRING=N
#   -DCLEAN_BUILD:BOOL=ON|OFF
#   -DDO_CONFIG:BOOL=ON|OFF
#   -DDO_BUILD:BOOL=ON|OFF
#   -DDO_TEST:BOOL=ON|OFF
#   -DCTEST_DO_SUBMIT:BOOL=OFF

include("${CMAKE_CURRENT_LIST_DIR}/lcm_do_package.cmake")

set(CTEST_TEST_TYPE Nightly)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_PROJECT_NAME "Albany-LCM")
set(CTEST_COMMAND "ctest -D ${CTEST_TEST_TYPE}")

# Default CDash submission to OFF; can be overridden via -DCTEST_DO_SUBMIT:BOOL=ON
if (NOT DEFINED CTEST_DO_SUBMIT)
  set(CTEST_DO_SUBMIT OFF)
endif()

message("PACKAGE ${PACKAGE}")
message("BUILD_THREADS ${BUILD_THREADS}")
message("CLEAN_BUILD ${CLEAN_BUILD}")
message("DO_CONFIG ${DO_CONFIG}")
message("DO_BUILD ${DO_BUILD}")
message("DO_TEST ${DO_TEST}")

set(BUILD_ID_STRING "$ENV{ARCH}-$ENV{TOOL_CHAIN}-$ENV{BUILD_TYPE}")
message("BUILD_ID_STRING ${BUILD_ID_STRING}")

# Build pass-through arguments from explicit boolean flags
set(PASS_ARGS "RESULT_VARIABLE" "PACKAGE_ERR")
if (CLEAN_BUILD)
  set(PASS_ARGS ${PASS_ARGS} "CLEAN_BUILD" "CLEAN_INSTALL")
endif()
if (DO_CONFIG)
  set(PASS_ARGS ${PASS_ARGS} "DO_CONFIG")
endif()
if (DO_BUILD)
  set(PASS_ARGS ${PASS_ARGS} "DO_BUILD")
endif()
if (DO_TEST)
  set(PASS_ARGS ${PASS_ARGS} "DO_TEST")
endif()
set(PASS_ARGS ${PASS_ARGS} "PACKAGE" "${PACKAGE}")
set(PASS_ARGS ${PASS_ARGS} "BUILD_THREADS" "${BUILD_THREADS}")
set(PASS_ARGS ${PASS_ARGS} "BUILD_ID_STRING" "${BUILD_ID_STRING}")

cmake_host_system_information(RESULT LCM_HOSTNAME QUERY HOSTNAME)

set(CTEST_BUILD_NAME "${LCM_HOSTNAME}-${PACKAGE}-$ENV{TOOL_CHAIN}-$ENV{BUILD_TYPE}")
set(CTEST_SITE "${LCM_HOSTNAME}")
set(CTEST_SOURCE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")
set(CTEST_BINARY_DIRECTORY "$ENV{LCM_DIR}/${PACKAGE}-build-${BUILD_ID_STRING}")
message("CTEST_BINARY_DIRECTORY ${CTEST_BINARY_DIRECTORY}")
snl_mkdir("${CTEST_BINARY_DIRECTORY}")

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake"
  "${CTEST_BINARY_DIRECTORY}/CTestCustom.cmake"
  COPYONLY)

# Over-write default limit for output
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE 5000000)
set(CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE 5000000)

ctest_start(${CTEST_TEST_TYPE})

lcm_do_package(${PASS_ARGS})

if (PACKAGE_ERR)
  message(FATAL_ERROR "lcm_do_package returned \"${PACKAGE_ERR}\"")
else()
  message("lcm_do_package returned \"${PACKAGE_ERR}\"")
endif()
