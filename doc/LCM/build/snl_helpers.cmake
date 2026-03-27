if(SNL_HELPERS_CMAKE)
  return()
endif()
set(SNL_HELPERS_CMAKE true)

include(CMakeParseArguments)

function(snl_rmdir DIR)
  message("snl_rmdir ${DIR}")
  if (EXISTS "${DIR}")
    message("rm -rf ${DIR}")
    file(REMOVE_RECURSE "${DIR}")
  endif()
endfunction(snl_rmdir)

function(snl_mkdir DIR)
  if (NOT EXISTS "${DIR}")
    file(MAKE_DIRECTORY "${DIR}")
  endif()
endfunction(snl_mkdir)

function(snl_update REPO_URL BRANCH SOURCE_DIR ERR)
  if (NOT BRANCH)
    set(BRANCH "master")
  endif()
  if (NOT EXISTS "${SOURCE_DIR}")
    execute_process(COMMAND "${CTEST_GIT_COMMAND}"
      clone -b "${BRANCH}" "${REPO_URL}" "${SOURCE_DIR}"
      RESULT_VARIABLE CLONE_ERR)
    if (CLONE_ERR)
      message(WARNING "Cannot clone ${REPO_URL} branch ${BRANCH}")
      set(${ERR} ${CLONE_ERR} PARENT_SCOPE)
      return()
    endif()
  endif()
  ctest_update(SOURCE "${SOURCE_DIR}" RETURN_VALUE FILES_CHANGED)
  if (FILES_CHANGED LESS 0)
    message(WARNING "Cannot update ${REPO_URL} branch ${BRANCH}")
  endif()
  set(${ERR} 0 PARENT_SCOPE)
endfunction(snl_update)

function(snl_submit PART)
  if (CTEST_DO_SUBMIT)
    ctest_submit(PARTS ${PART} RETRY_COUNT 3 RETRY_DELAY 10)
  endif()
endfunction(snl_submit)

function(snl_config SOURCE_DIR BUILD_DIR CONFIG_OPTS ERR)
  snl_mkdir("${BUILD_DIR}")
  ctest_configure(
    BUILD "${BUILD_DIR}"
    SOURCE "${SOURCE_DIR}"
    APPEND
    OPTIONS "${CONFIG_OPTS}"
    RETURN_VALUE CONFIG_ERR
  )
  snl_submit("Configure")
  if (CONFIG_ERR)
    message(WARNING "Cannot configure!")
    set(${ERR} ${CONFIG_ERR} PARENT_SCOPE)
    return()
  endif()
  set(${ERR} 0 PARENT_SCOPE)
endfunction(snl_config)

function(snl_build BUILD_DIR NUM_THREADS TARGET ERR)
  set(CTEST_USE_LAUNCHERS 1)
  ctest_build(
    BUILD "${BUILD_DIR}"
    APPEND
    FLAGS "-j ${NUM_THREADS}"
    TARGET "${TARGET}"
    NUMBER_ERRORS NERRS
    NUMBER_WARNINGS NWARNS
    RETURN_VALUE BUILD_ERR
  )
  if ((NOT BUILD_ERR) AND (NERRS GREATER "0"))
    message("BUILD_ERR was ${BUILD_ERR} despite NERRS being ${NERRS}")
    set(BUILD_ERR "-${NERRS}")
  endif()
  snl_submit("Build")
  if (BUILD_ERR)
    message(WARNING "Cannot make ${TARGET}!")
    set(${ERR} ${BUILD_ERR} PARENT_SCOPE)
    return()
  endif()
  set(${ERR} 0 PARENT_SCOPE)
endfunction(snl_build)

function(snl_test BUILD_DIR)
  message("ctest_read_custom_files(${CTEST_BINARY_DIRECTORY})")
  ctest_read_custom_files("${CTEST_BINARY_DIRECTORY}")
  ctest_test(
    BUILD "${BUILD_DIR}"
    APPEND
  )
  snl_submit("Test")
endfunction(snl_test)

function(snl_do_subproject)
  set(BOOL_OPTS
      "CLEAN_SOURCE"
      "CLEAN_BUILD"
      "CLEAN_INSTALL"
      "DO_PROJECT"
      "DO_UPDATE"
      "DO_CONFIG"
      "DO_BUILD"
      "DO_INSTALL"
      "DO_TEST")
  set(ONE_VALUE_OPTS
      "PROJECT"
      "REPO_URL"
      "BRANCH"
      "SOURCE_DIR"
      "BUILD_DIR"
      "INSTALL_DIR"
      "BUILD_THREADS"
      "RESULT_VARIABLE"
    )
  set(MULTI_VALUE_OPTS
      "CONFIG_OPTS"
    )
  cmake_parse_arguments(SNL "${BOOL_OPTS}" "${ONE_VALUE_OPTS}" "${MULTI_VALUE_OPTS}" ${ARGN})
  if (SNL_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "snl_do_subproject called with unrecognized arguments \"${SNL_UNPARSED_ARGUMENTS}\", all arguments were \"${ARGN}\"")
  endif()
  if (SNL_CLEAN_INSTALL)
    snl_rmdir("${SNL_INSTALL_DIR}")
  endif()
  if (SNL_CLEAN_BUILD)
    snl_rmdir("${SNL_BUILD_DIR}")
  endif()
  if (SNL_CLEAN_SOURCE)
    snl_rmdir("${SNL_SOURCE_DIR}")
  endif()
  if (SNL_DO_UPDATE)
    snl_update("${SNL_REPO_URL}" "${SNL_BRANCH}"
        "${SNL_SOURCE_DIR}" UPDATE_ERR)
    if (UPDATE_ERR)
      if (SNL_RESULT_VARIABLE)
        set(${SNL_RESULT_VARIABLE} ${UPDATE_ERR} PARENT_SCOPE)
      endif()
      return()
    endif()
  endif()
  if (SNL_DO_CONFIG)
    snl_config("${SNL_SOURCE_DIR}" "${SNL_BUILD_DIR}" "${SNL_CONFIG_OPTS}" CONFIG_ERR)
    if (CONFIG_ERR)
      if (SNL_RESULT_VARIABLE)
        set(${SNL_RESULT_VARIABLE} ${CONFIG_ERR} PARENT_SCOPE)
      endif()
      return()
    endif()
  endif()
  if (SNL_DO_BUILD)
    snl_build("${SNL_BUILD_DIR}" "${SNL_BUILD_THREADS}" "all" BUILD_ERR)
    message("snl_build()  returned ${BUILD_ERR}")
    if (BUILD_ERR)
      if (SNL_RESULT_VARIABLE)
        set(${SNL_RESULT_VARIABLE} ${BUILD_ERR} PARENT_SCOPE)
      endif()
      return()
    endif()
  endif()
  if (SNL_DO_TEST)
    snl_test("${SNL_BUILD_DIR}")
  else()
    # Run ctest with a pattern that matches nothing to generate an empty
    # test result set, then submit it.  CDash requires a Test submission
    # to consider the build complete.
    ctest_test(BUILD "${SNL_BUILD_DIR}" INCLUDE "^$" APPEND)
    snl_submit("Test")
  endif()
  if (SNL_DO_INSTALL)
    snl_mkdir("${SNL_INSTALL_DIR}")
    snl_build("${SNL_BUILD_DIR}" "${SNL_BUILD_THREADS}" "install" INSTALL_ERR)
    if (INSTALL_ERR)
      if (SNL_RESULT_VARIABLE)
        set(${SNL_RESULT_VARIABLE} ${INSTALL_ERR} PARENT_SCOPE)
      endif()
      return()
    endif()
  endif()
  if (SNL_RESULT_VARIABLE)
    set(${SNL_RESULT_VARIABLE} 0 PARENT_SCOPE)
  endif()
endfunction(snl_do_subproject)
