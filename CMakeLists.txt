#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

# CMAKE File for Albany building against an installed Trilinos

cmake_minimum_required(VERSION 2.8.12)
include(CMakeDependentOption)

option(USE_NEW_POLICY_CMP0060 "Flag to turn on new behavior for policy CMP0060"
       OFF)
if(USE_NEW_POLICY_CMP0060)
  cmake_policy(SET CMP0060 NEW)
endif()

if(NOT DEFINED SET_COMPILERS_AUTOMATICALLY)
  set(SET_COMPILERS_AUTOMATICALLY TRUE)
endif()

if(NOT SET_COMPILERS_AUTOMATICALLY)
  message("Experimental configuration.")
  project(Albany CXX C Fortran)
endif()

# Advertise that we support shared libs
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

# Get Trilinos as one entity
if(DEFINED ALBANY_TRILINOS_DIR)
  set(CMAKE_PREFIX_PATH ${ALBANY_TRILINOS_DIR} ${CMAKE_PREFIX_PATH})
endif()
find_package(Trilinos REQUIRED)

option(
  ALBANY_SUPPRESS_TRILINOS_WARNINGS
  "Whether or not Trilinos headers should be treated as 'system' headers (hence, without issuing warnings)"
  OFF)

# Trilinos_BIN_DIRS probably should be defined in the Trilinos config. Until it
# is, set it here. This is needed to find SEACAS tools used during testing (epu,
# etc).
set(Trilinos_BIN_DIRS "${Trilinos_DIR}/../../../bin")
set(Trilinos_LIB_DIRS "${Trilinos_DIR}/../../../lib")

# Set Trilinos commit id
if(EXISTS "${Trilinos_DIR}/../../../TrilinosRepoVersion.txt")
  file(STRINGS "${Trilinos_DIR}/../../../TrilinosRepoVersion.txt"
       TRILINOS_VERSION_FILE)
  set(TRILINOS_GIT_COMMIT_ID_LINE_LOC 1)
  list(GET TRILINOS_VERSION_FILE ${TRILINOS_GIT_COMMIT_ID_LINE_LOC}
           TRILINOS_GIT_COMMIT_ID_LINE)
  string(REGEX MATCH "^[^ ]+" Trilinos_GIT_COMMIT_ID
               ${TRILINOS_GIT_COMMIT_ID_LINE})
  set(ALBANY_TRILINOS_GIT_COMMIT_ID ${Trilinos_GIT_COMMIT_ID})
endif()

message("\nFound Trilinos!  Here are the details: ")
message("-- Trilinos_DIR = ${Trilinos_DIR}")
message("-- Trilinos_VERSION = ${Trilinos_VERSION}")
message("-- Trilinos_GIT_COMMIT_ID = ${Trilinos_GIT_COMMIT_ID}")
message("-- Trilinos_BIN_DIRS = ${Trilinos_BIN_DIRS}")
message("-- Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
message("-- Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
message("-- Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
message("-- Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
message("End of Trilinos details\n")

# Compress Trilinos lists that are used
list(REVERSE Trilinos_INCLUDE_DIRS)
list(REMOVE_DUPLICATES Trilinos_INCLUDE_DIRS)
list(REVERSE Trilinos_INCLUDE_DIRS)

list(REVERSE Trilinos_TPL_INCLUDE_DIRS)
list(REMOVE_DUPLICATES Trilinos_TPL_INCLUDE_DIRS)
list(REVERSE Trilinos_TPL_INCLUDE_DIRS)

list(REVERSE Trilinos_LIBRARIES)
list(REMOVE_DUPLICATES Trilinos_LIBRARIES)
list(REVERSE Trilinos_LIBRARIES)

list(REVERSE Trilinos_TPL_LIBRARIES)
list(REMOVE_DUPLICATES Trilinos_TPL_LIBRARIES)
list(REVERSE Trilinos_TPL_LIBRARIES)

list(FIND Trilinos_PACKAGE_LIST STKClassic STKClassic_List_ID)
if(STKClassic_List_ID GREATER -1)
  message(
    FATAL_ERROR
      "STKClassic is no longer used and conflicts with STKIo and STKMesh.")
endif()

# Set Albany branch and commit id
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE ALBANY_GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE ALBANY_GIT_COMMIT_ID
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message("Albany version details:")
message("-- ALBANY_GIT_BRANCH = ${ALBANY_GIT_BRANCH}")
message("-- ALBANY_GIT_COMMIT_ID = ${ALBANY_GIT_COMMIT_ID}")
message("End of Albany version details\n")

# Optional Installation helpers Note that some of this functionality depends on
# CMAKE > 2.8.8
set(INSTALL_ALBANY
    FALSE
    CACHE BOOL "Whether Albany should be installed")
if(INSTALL_ALBANY)
  message("")
  message("Installation files for Albany will be created")
  message("-- install location = ${CMAKE_INSTALL_PREFIX}")
  message("")
  set(BINARY_INSTALL_DIR bin)
  set(INCLUDE_INSTALL_DIR include)
  set(LIB_INSTALL_DIR lib)
  include(CMakePackageConfigHelpers)
endif()

option(ENABLE_CROSS_COMPILE "Flag to turn on cross-compiling tools" OFF)
if(ENABLE_CROSS_COMPILE)
  set(Albany_CROSS_COMPILE ON)
  message("Building with cross-compiling tools.")
else()
  set(Albany_CROSS_COMPILE OFF)
endif()

if(SET_COMPILERS_AUTOMATICALLY)
  message("Setting and checking of compilers:")
  if(Albany_CROSS_COMPILE)
    # Force the compilers to be the same as Trilinos (GAH experimental) This
    # syntax is needed when cross compiling or the compilers get checked again
    # by the Albany configure, and will probably fail, because the options won't
    # be right and the exes won't run on the host.
    include(CMakeForceCompiler)
    set(CMAKE_SYSTEM_NAME Generic)
    cmake_force_cxx_compiler(${Trilinos_CXX_COMPILER} Generic)
    cmake_force_c_compiler(${Trilinos_C_COMPILER} Generic)
    cmake_force_fortran_compiler(${Trilinos_Fortran_COMPILER} Generic)
    # SET(CMAKE_SYSTEM_NAME ${Trilinos_SYSTEM_NAME})
    # CMAKE_FORCE_CXX_COMPILER(${Trilinos_CXX_COMPILER}
    # ${Trilinos_CXX_COMPILER_ID}) CMAKE_FORCE_C_COMPILER(${Trilinos_C_COMPILER}
    # ${Trilinos_C_COMPILER_ID})
    # CMAKE_FORCE_Fortran_COMPILER(${Trilinos_Fortran_COMPILER}
    # ${Trilinos_Fortran_COMPILER_ID}) SET(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES
    # ${Trilinos_Fortran_IMPLICIT_LINK_LIBRARIES})
  else()
    set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
    set(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
    set(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER})
  endif()
else()
  # Make sure the compilers match.
  message("Checking compilers:")
  if(NOT ${Trilinos_CXX_COMPILER} STREQUAL ${CMAKE_CXX_COMPILER})
    message(
      FATAL_ERROR
        "C++ compilers don't match (Trilinos: ${Trilinos_CXX_COMPILER}, ${PROJECT_NAME}: ${CMAKE_CXX_COMPILER})."
    )
  endif()
  if(NOT ${Trilinos_C_COMPILER} STREQUAL ${CMAKE_C_COMPILER})
    message(
      FATAL_ERROR
        "C compilers don't match (Trilinos: ${Trilinos_C_COMPILER}, ${PROJECT_NAME}: ${CMAKE_C_COMPILER})."
    )
  endif()
  if(NOT ${Trilinos_Fortran_COMPILER} STREQUAL ${CMAKE_Fortran_COMPILER})
    message(
      FATAL_ERROR
        "Fortran compilers don't match (Trilinos: ${Trilinos_Fortran_COMPILER}, ${PROJECT_NAME}: ${CMAKE_Fortran_COMPILER})."
    )
  endif()
  set(Trilinos_CXX_COMPILER_ID ${CMAKE_CXX_COMPILER_ID})
  set(Trilinos_C_COMPILER_ID ${CMAKE_C_COMPILER_ID})
  set(Trilinos_Fortran_COMPILER_ID ${CMAKE_Fortran_COMPILER_ID})
endif()

# Build Albany as shared libraries if Trilinos was compiled that way
if(Trilinos_BUILD_SHARED_LIBS)
  set(Albany_LIBRARY_TYPE SHARED)
  set(BUILD_SHARED_LIBS ON)
  message("-- Building Albany with shared libraries")
else(Trilinos_BUILD_SHARED_LIBS)
  set(Albany_LIBRARY_TYPE STATIC)
endif(Trilinos_BUILD_SHARED_LIBS)

option(Albany_BUILD_STATIC_EXE
       "Flag to turn on building of static executables in Albany" OFF)
if((NOT BUILD_SHARED_LIBS) AND (Albany_BUILD_STATIC_EXE))
  message("-- Building Albany with static executables")
endif((NOT BUILD_SHARED_LIBS) AND (Albany_BUILD_STATIC_EXE))

# Option to turn on and off deprecation warnings
option(Albany_DEPRECATED_DECLARATIONS
       "Flag to turn on warnings for deprecated code" ON)
if(Albany_DEPRECATED_DECLARATIONS)
  set(ALBANY_ENABLE_DEPRECATED ON)
  message("-- Deprecated code warnings Enabled")
else()
  message("-- Deprecated code warnings are NOT Enabled.")
endif()

# End of setup and error checking NOTE: PROJECT command checks for compilers, so
# this statement is moved AFTER setting CMAKE_CXX_COMPILER from Trilinos

if(SET_COMPILERS_AUTOMATICALLY)
  project(Albany)
endif()

# AGS: Adding these lines so Ctest can be run to submit to cdash dashboard Uses
# CTestConfig.cmake file for Cdash info.
enable_testing()
include(CTest)

set(ALBANY_ENABLE_FORTRAN
    ON
    CACHE BOOL "enable fortran")

if(CMAKE_Fortran_COMPILER AND ALBANY_ENABLE_FORTRAN) # Enable Fortran if it is
                                                     # enabled in Trilinos.
  enable_language(Fortran)
endif()

# Make Albany inherit Trilinos compiler flags. The most important are the
# presence or absence of -NDEBUG (which can affect linking for complicated
# reasons) and -std=c++11.
set(CMAKE_CXX_FLAGS "${Trilinos_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
# GAH - temporarily set Albany flags for clang debugging SET(CMAKE_CXX_FLAGS
# "-ggdb -std=c++11 -O2 -march=native -DNDEBUG")

if(SET_COMPILERS_AUTOMATICALLY)
  set(CMAKE_AR ${Trilinos_AR})
  set(CMAKE_LINKER ${Trilinos_LINKER})
endif()

# No longer needed but keep for now as this shows how to silence specific
# warnings on Intel compiler IF(${Trilinos_CXX_COMPILER_ID} MATCHES "Intel")
# SET(USING_INTEL TRUE) On Intel, turn of interprocedural optimization at
# linking to keep linker from hanging This could possibly be removed with
# further study (GAH) SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-
# ip") SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd2536,2196,858,488,161,47")
# ENDIF()

# Compiler details
set(ALBANY_CXX_COMPILER_ID ${CMAKE_CXX_COMPILER_ID})
message("-- ALBANY_CXX_COMPILER_ID = ${ALBANY_CXX_COMPILER_ID}")
set(ALBANY_CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
message("-- ALBANY_CXX_COMPILER_VERSION = ${ALBANY_CXX_COMPILER_VERSION}")
if(EXISTS "${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h")
  file(READ "${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h"
       KOKKOS_CORE_CONFIG_FILE)
  string(REGEX MATCH "KOKKOS_ENABLE_OPENMP" ALBANY_ENABLE_OPENMP
               ${KOKKOS_CORE_CONFIG_FILE})
  if(ALBANY_ENABLE_OPENMP)
    message("-- Kokkos is configured to use OpenMP, Albany will also.")
  endif()
  string(REGEX MATCH "KOKKOS_ENABLE_CUDA" ALBANY_ENABLE_CUDA
               ${KOKKOS_CORE_CONFIG_FILE})
  if(ALBANY_ENABLE_CUDA)
    message("-- Kokkos is configured to use CUDA, Albany will also.")
    string(REGEX MATCH "KOKKOS_COMPILER_CUDA_VERSION ([0-9]*)" _
                 ${KOKKOS_CORE_CONFIG_FILE})
    set(ALBANY_CUDA_COMPILER_VERSION "${CMAKE_MATCH_1}")
    message("-- ALBANY_CUDA_COMPILER_VERSION = ${ALBANY_CUDA_COMPILER_VERSION}")
  endif()
else()
  message(
    FATAL_ERROR
      "\nError: ${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h not found!")
endif()
message("End of compiler details\n")

# Check for some required Trilinos package dependencies (Piro, Phalanx) ##
message("Checking Trilinos build for required and optional packages")

# ##############################################################################
# Check REQUIRED packages ###
# ##############################################################################

message("\n  **********************************")
message("  *** Required Trilinos Packages ***")
message("  **********************************\n")
# Piro
list(FIND Trilinos_PACKAGE_LIST Piro Piro_List_ID)
if(Piro_List_ID GREATER -1)
  message(
    "-- Looking for Piro (required) in Trilinos ...                Found!")
else()
  message(
    "-- Looking for Piro (required) in Trilinos ...                NOT Found!")
  message(
    FATAL_ERROR
      "Albany REQUIRES Piro. See file trilinos-cmake in doc directory."
  )
endif()

# Phalanx
list(FIND Trilinos_PACKAGE_LIST Phalanx Phalanx_List_ID)
if(Piro_List_ID GREATER -1)
  message(
    "-- Looking for Phalanx (required) in Trilinos ...             Found!")
else()
  message(
    "-- Looking for Phalanx (required) in Trilinos ...             NOT Found!")
  message(
    FATAL_ERROR
      "Albany REQUIRES Phalanx. See file trilinos-cmake in doc directory."
  )
endif()

# NOX
list(FIND Trilinos_PACKAGE_LIST NOX NOX_List_ID)
if(Piro_List_ID GREATER -1)
  message(
    "-- Looking for NOX (required) in Trilinos ...                 Found!")
else()
  message(
    "-- Looking for NOX (required) in Trilinos ...                 NOT Found!")
  message(
    FATAL_ERROR
      "Albany REQUIRES NOX. See file trilinos-cmake in doc directory."
  )
endif()

# ThyraTpetraAdapters
list(FIND Trilinos_PACKAGE_LIST ThyraTpetraAdapters ThyraTpetraAdapters_List_ID)
if(Piro_List_ID GREATER -1)
  message(
    "-- Looking for ThyraTpetraAdapters (required) in Trilinos ... Found!")
else()
  message(
    "-- Looking for ThyraTpetraAdapters (required) in Trilinos ... NOT Found!")
  message(
    FATAL_ERROR
      "Albany REQUIRES ThyraTpetraAdapters. See file trilinos-cmake in doc directory."
  )
endif()

# Intrepid2
list(FIND Trilinos_PACKAGE_LIST Intrepid2 Intrepid2_List_ID)
if(Piro_List_ID GREATER -1)
  message(
    "-- Looking for Intrepid2 (required) in Trilinos ...           Found!")
else()
  message(
    "-- Looking for Intrepid2 (required) in Trilinos ...           NOT Found!")
  message(
    FATAL_ERROR
      "Albany REQUIRES Intrepid2. See file trilinos-cmake in doc directory."
  )
endif()

# MiniTensor
list(FIND Trilinos_PACKAGE_LIST MiniTensor MiniTensor_List_ID)
if(Piro_List_ID GREATER -1)
  message(
    "-- Looking for MiniTensor (required) in Trilinos ...          Found!")
else()
  message(
    "-- Looking for MiniTensor (required) in Trilinos ...          NOT Found!")
  message(
    FATAL_ERROR
      "Albany REQUIRES MiniTensor. See file trilinos-cmake in doc directory."
  )
endif()

# ##############################################################################
# Check OPTIONAL packages ###
# ##############################################################################

message("\n  **********************************")
message("  *** Optional Trilinos Packages ***")
message("  **********************************\n")
# Check for some OPTIONAL trilinos packages, which set DEFINES in Alany build
# ###

# MPI check -- defaults to TRUE MESSAGE("-- Checking if MPI is enabled in
# Trilinos:")

list(FIND Trilinos_TPL_LIST MPI MPI_List_ID)
if(MPI_List_ID GREATER -1)
  message("-- Looking for MPI in Trilinos:      Found")
  set(ALBANY_MPI TRUE)
else()
  message("-- Looking for MPI in Trilinos:      NOT found, MPI NOT ENABLED!")
  set(ALBANY_MPI FALSE)
endif()

# set optional dependency on stk
list(FIND Trilinos_PACKAGE_LIST STK have_stk_List_ID)
if(have_stk_List_ID GREATER -1)
  message("-- Looking for STK:                  Found")
  set(ALBANY_STK TRUE)

  # set optional dependency on SEACAS, for I/O via stk_io MESSAGE("-- Looking
  # for SEACAS:")
  list(FIND Trilinos_PACKAGE_LIST SEACASIoss SEACASIoss_List_ID)
  if(SEACASIoss_List_ID GREATER -1)
    message("  -- Looking for SEACASIoss:         Found")
    set(ALBANY_SEACAS TRUE)
    list(FIND SEACASIoss_LIBRARIES Iopx Iopx_List_ID)
    if(Iopx_List_ID GREATER -1)
      message("  -- Looking for Iopx:               Found")
      set(ALBANY_IOPX TRUE)
    else()
      message("  -- Looking for Iopx:               NOT found.")
      set(ALBANY_IOPX FALSE)
    endif()
  else()
    message("  -- Looking for SEACASIoss:         NOT found.")
    set(ALBANY_SEACAS FALSE)
  endif()

  # set optional dependency on stk_percept, for uniform adaptivity
  list(FIND Trilinos_LIBRARIES stk_percept stk_percept_List_ID)
  if(stk_percept_List_ID GREATER -1)
    message("  -- Looking for stk_percept:        Found")
    set(ALBANY_STK_PERCEPT TRUE)
  else()
    message("  -- Looking for stk_percept:        NOT found.")
    set(ALBANY_STK_PERCEPT FALSE)
  endif()

else()
  message("-- Looking for STK:                  NOT found.")
  set(ALBANY_STK FALSE)
endif()

# set optional dependency on Panzer's ExprEval subpackage
list(FIND Trilinos_PACKAGE_LIST PanzerExprEval PanzerExprEval_List_ID)
if(PanzerExprEval_List_ID GREATER -1)
  message("-- Looking for PanzerExprEval:       Found")
  set(ALBANY_PANZER_EXPR_EVAL TRUE)
else()
  message("-- Looking for PanzerExprEval:       NOT found.")
  set(ALBANY_PANZER_EXPR_EVAL FALSE)
endif()

# set optional dependency on Zoltan, for partitioning MESSAGE("-- Looking for
# Zoltan:")
list(FIND Trilinos_PACKAGE_LIST Zoltan Zoltan_List_ID)
if(Zoltan_List_ID GREATER -1)
  set(ALBANY_ZOLTAN TRUE)
  if(ALBANY_STK)
    set(ALBANY_STK_REBALANCE TRUE)
    message("-- Looking for Zoltan:               Found.")
  else()
    set(ALBANY_STK_REBALANCE FALSE)
    message("-- Looking for Zoltan:               Found.")
  endif()
else()
  message("-- Looking for Zoltan:               NOT found.")
  set(ALBANY_ZOLTAN FALSE)
  set(ALBANY_STK_REBALANCE FALSE)
endif()

# set optional dependency on Amesos2, for solving and preconditioning
list(FIND Trilinos_PACKAGE_LIST Amesos2 Amesos2_List_ID)
if(Amesos2_List_ID GREATER -1)
  message("-- Looking for Amesos2:              Found.")
  set(ALBANY_AMESOS2 TRUE)
  # We can't rely on Amesos2_TPL_LIST any longer; KLU2 and Basker are not TPLs.
  # These will soon be enabled by default in Amesos2. So we'll just assume we
  # have a coarse direct solver.
  set(ALBANY_AMESOS2_OPERATIONAL_SOLVER TRUE)
else()
  message("-- Looking for Amesos2:              NOT found.")
  set(ALBANY_AMESOS2 FALSE)
  set(ALBANY_AMESOS2_OPERATIONAL_SOLVER FALSE)
endif()

# set optional dependency on Ifpack2, for preconditioning
list(FIND Trilinos_PACKAGE_LIST Ifpack2 Ifpack2_List_ID)
if(Ifpack2_List_ID GREATER -1)
  message("-- Looking for Ifpack2:              Found.")
  set(ALBANY_IFPACK2 TRUE)
else()
  message("-- Looking for Ifpack2:              NOT found.")
  set(ALBANY_IFPACK2 FALSE)
endif()

# set optional dependency on Teko, for preconditioning
list(FIND Trilinos_PACKAGE_LIST Teko Teko_List_ID)
if(Teko_List_ID GREATER -1)
  message("-- Looking for Teko:                 Found.")
  set(ALBANY_TEKO TRUE)
else()
  message("-- Looking for Teko:                 NOT found.")
  set(ALBANY_TEKO FALSE)
endif()

# set optional dependency on MueLu, for preconditioning
list(FIND Trilinos_PACKAGE_LIST MueLu MueLu_List_ID)
if(MueLu_List_ID GREATER -1)
  message("-- Looking for MueLu:                Found.")
  set(ALBANY_MUELU TRUE)
  list(FIND MueLu_PACKAGE_LIST Ifpack2 MueLu_Ifpack2_List_ID)
  list(FIND MueLu_PACKAGE_LIST Amesos2 MueLu_Amesos2_List_ID)
  if(MueLu_Ifpack2_List_ID GREATER -1
     AND MueLu_Amesos2_List_ID GREATER -1
     AND ALBANY_AMESOS2_OPERATIONAL_SOLVER)
    message("  -- MueLu tests enabled")
    set(ALBANY_MUELU_EXAMPLES TRUE)
  else()
    message(
      "  -- MueLu tests NOT enabled. Ifpack2 and Amesos2 compiled with an operational solver are required."
    )
    set(ALBANY_MUELU_EXAMPLES FALSE)
  endif()
else()
  message("-- Looking for MueLu:                NOT found.")
  set(ALBANY_MUELU FALSE)
  set(ALBANY_MUELU_EXAMPLES FALSE)
endif()

# set optional dependency on FROSch, for preconditioning
list(FIND Trilinos_PACKAGE_LIST ShyLU_DDFROSch FROSch_List_ID)
if(FROSch_List_ID GREATER -1)
  message("-- Looking for FROSch:               Found.")
  set(ALBANY_FROSCH TRUE)
else()
  message("-- Looking for FROSch:               NOT found.")
  set(ALBANY_FROSCH FALSE)
endif()

list(FIND Trilinos_PACKAGE_LIST Rythmos Rythmos_List_ID)
if(Rythmos_List_ID GREATER -1)
  message("-- Looking for Rythmos:              Found.")
  set(ALBANY_RYTHMOS TRUE)
else()
  message("-- Looking for Rythmos:              NOT found.")
  set(ALBANY_RYTHMOS FALSE)
endif()

# Check if Trilinos has ROL
list(FIND Trilinos_PACKAGE_LIST ROL ROL_List_ID)
if(ROL_List_ID GREATER -1)
  message("-- Looking for ROL in Trilinos:      Found")
  set(ALBANY_ROL TRUE)
else()
  message("-- Looking for ROL in Trilinos:    NOT found.")
  set(ALBANY_ROL FALSE)
endif()

# Check if Trilinos has DTK
list(FIND Trilinos_PACKAGE_LIST DataTransferKit DataTransferKit_List_ID)
if(DataTransferKit_List_ID GREATER -1)
  message("-- Looking for DTK in Trilinos:      Found")
  set(ALBANY_DTK TRUE)
else()
  message("-- Looking for DTK in Trilinos:      NOT found.")
  set(ALBANY_DTK FALSE)
endif()

# Check if Trilinos has Tempus
list(FIND Trilinos_PACKAGE_LIST Tempus Tempus_List_ID)
if(Tempus_List_ID GREATER -1)
  message("-- Looking for Tempus in Trilinos:   Found")
  set(ALBANY_TEMPUS TRUE)
else()
  message("-- Looking for Tempus in Trilinos:   NOT found.")
  if(ENABLE_LCM)
    message(
      FATAL_ERROR
        "LCM enabled but Tempus not installed from Trilinos (LCM needs Tempus)!  Please recompile Trilinos with Tempus enabled."
    )
  endif()
  set(ALBANY_TEMPUS FALSE)
endif()

set(ALBANY_EPETRA FALSE)
set(ALBANY_DAKOTA FALSE)

# ##############################################################################
# Check for optional physics sets in Albany ###
# ##############################################################################

message("\nChecking which Albany physics sets are enabled:")

# set optional dependency on demoPDEs, defaults to Enabled
option(ENABLE_DEMO_PDES "Flag to turn on demonstration PDEs problems" ON)
if(ENABLE_DEMO_PDES)
  message("-- DEMO_PDES                       is Enabled")
  set(ALBANY_DEMO_PDES TRUE)
else()
  message("-- DEMO_PDES                       is NOT Enabled.")
  set(ALBANY_DEMO_PDES FALSE)
endif()

# set optional dependency on ATO, defaults to Disabled
option(ENABLE_LCM "Flag to turn on LCM Mechanics source code" OFF)
option(ENABLE_ATO "Flag to turn on ATO source code" OFF)
if(ENABLE_ATO)
  message("-- ATO                             is Enabled")
  set(ALBANY_ATO TRUE)
  if(ENABLE_ISOLIB)
    set(ATO_USES_ISOLIB TRUE)
  endif()
  if(ENABLE_DOTK)
    set(ATO_USES_DOTK TRUE)
  endif()
  if(ENABLE_NLopt)
    set(ATO_USES_NLOPT TRUE)
  endif()
  if(ENABLE_Cogent)
    set(ATO_USES_COGENT TRUE)
  endif()
else()
  message("-- ATO                             is NOT Enabled.")
  set(ALBANY_ATO FALSE)
endif()

# set optional dependency on LCM, defaults to Disabled
option(ENABLE_LCM_TEST_EXES "Flag to turn on LCM Mechanics test executables" ON)
option(DISABLE_LCM_EXODIFF_SENSITIVE_TESTS
       "Flag to turn off LCM exodiff sensitive tests" OFF)
set(LCM_TEST_EXES FALSE)
if(ENABLE_LCM OR ENABLE_ATO)
  if(NOT ENABLE_LCM)
    message(
      "-- LCM                             is Enabled because ATO requires LCM.")
  else()
    message("-- LCM                             is Enabled.")
  endif()
  set(ALBANY_LCM TRUE)
  if(ENABLE_LCM AND NOT ALBANY_ROL)
    message(
      FATAL_ERROR
        "LCM enabled but ROL not installed from Trilinos (LCM needs ROL)")
  endif()
  if(ENABLE_LCM_TEST_EXES AND ALBANY_STK) # Need STK for the EXES
    set(LCM_TEST_EXES TRUE)
  endif()
  if(DISABLE_LCM_EXODIFF_SENSITIVE_TESTS)
    message("  -- LCM exodiff sensitive tests are DISABLED!")
    set(DISABLE_LCM_EXODIFF_SENSITIVE_TESTS TRUE)
  endif()
else()
  message("-- LCM                             is NOT Enabled.")
  set(ALBANY_LCM FALSE)
endif()

# set optional dependency on MPMD, defaults to Disabled
option(ENABLE_MPMD "Flag to turn on MPMD source code" OFF)
if(ENABLE_MPMD)
  message("-- MPMP executable                 is Enabled.")
  set(ALBANY_MPMD TRUE)
else()
  message("-- MPMP executable                 is NOT Enabled.")
  set(ALBANY_MPMD FALSE)
endif()

# Set optional build of LandIce (Land Ice SciDAC project), defaults to Disabled
option(ENABLE_LANDICE "Flag to turn on LandIce Source code" OFF)
option(ENABLE_CISM_INTERFACE
       "Flag to turn on LandIce Interface to the CISM code" OFF)
option(ENABLE_MPAS_INTERFACE "Flag to turn on LandIce Source code" OFF)
if(ENABLE_LANDICE)
  message("-- LandIce                         is Enabled")
  set(ALBANY_LANDICE TRUE)

  # set optional dependency of mesh on parameters, defaults to Disabled
  option(ENABLE_USE_CISM_FLOW_PARAMETERS
         "Flag to use the parameters of Glen's law used by Cism" ON)
  if(ENABLE_USE_CISM_FLOW_PARAMETERS)
    message("  -- USE_CISM_FLOW_PARAMETERS      is Enabled.")
    set(USE_CISM_FLOW_PARAMETERS TRUE)
  else()
    set(USE_CISM_FLOW_PARAMETERS FALSE)
    message("  -- USE_CISM_FLOW_PARAMETERS      is NOT Enabled.")
  endif()

  # Set optional build of MPAS Interface for LandIce (Land Ice SciDAC project),
  # defaults to Disabled
  if(ENABLE_MPAS_INTERFACE)
    message("-- MPAS_INTERFACE                  is Enabled")
    set(MPAS_HAS_LANDICE TRUE)
  else()
    message("  -- MPAS_INTERFACE                is NOT Enabled.")
    set(MPAS_HAS_LANDICE FALSE)
  endif()

  # Set optional build of CISM Interface for LandIce (Land Ice SciDAC project),
  # defaults to Disabled
  option(ENABLE_CISM_REDUCED_COMM
         "Flag to turn on reduced communicator for CISM-Albany runs" ON)
  option(ENABLE_CISM_CHECK_COMPARISONS
         "Flag to turn on comparisons checking for CISM-Albany" OFF)

  if(ENABLE_CISM_INTERFACE)
    message("  -- CISM_INTERFACE                is Enabled")
    set(CISM_HAS_LANDICE TRUE)
    if(ENABLE_CISM_CHECK_COMPARISONS)
      set(CISM_CHECK_COMPARISONS TRUE)
      message(
        "  -- CISM_CHECK_COMPARISONS      is Enabled. Error will be thrown if comparisons fail for CISM-Albany run."
      )
    else()
      message(
        "  -- CISM_CHECK_COMPARISONS      is Disabled.  Error will NOT be thrown if comparisons fail for CISM-Albany run.  Please be cautious when running ctest from CISM!"
      )
    endif()
    if(ENABLE_CISM_REDUCED_COMM)
      set(REDUCED_COMM TRUE)
      message(
        "  -- CISM_REDUCED_COMM           is Enabled.  CISM-Albany will use reduced communicator."
      )
    endif()
    set(CISM_HAS_LANDICE TRUE)
    if(NOT CISM_INCLUDE_DIR)
      message("\nError:  CISM_INCLUDE_DIR not defined\n")
    else()
      message("  -- CISM_INCLUDE_DIR ${CISM_INCLUDE_DIR}")
    endif()
    if(NOT CISM_EXE_DIR)
      message("\n CISM_EXE_DIR not defined.  CismAlbany test will not run.\n")
    else()
      message("  -- CISM_EXE_DIR ${CISM_EXE_DIR}")
    endif()
  else()
    message("  -- CISM_INTERFACE                is NOT Enabled.")
    set(CISM_HAS_LANDICE FALSE)
  endif()

else()
  message("-- LandIce                         is NOT Enabled.")
  set(ALBANY_LANDICE FALSE)
  if(ENABLE_MPAS_INTERFACE)
    message(FATAL_ERROR "\nError:  ENABLE_MPAS_INTERFACE requires ENABLE_LANDICE
  to be ON\n")
  elseif(ENABLE_CISM_INTERFACE)
    message(FATAL_ERROR "\nError:  ENABLE_CISM_INTERFACE requires ENABLE_LANDICE
  to be ON\n")
  endif()
endif()

option(ENABLE_TSUNAMI "Flag to turn on TSUNAMI Source code" OFF)
if(ENABLE_TSUNAMI)
  message("-- TSUNAMI                         is Enabled")
  set(ALBANY_TSUNAMI TRUE)
else()
  message("-- TSUNAMI                         is NOT Enabled.")
  set(ALBANY_TSUNAMI FALSE)
endif()

# set optional dependency of mesh on parameters, defaults to Disabled
option(
  ENABLE_MESH_DEPENDS_ON_PARAMETERS
  "Flag to turn on dependency of mesh on parameters, e.g for shape optimization"
  OFF)
if(ENABLE_MESH_DEPENDS_ON_PARAMETERS)
  message("-- MESH_DEPENDS_ON_PARAMETERS      is Enabled")
  set(ALBANY_MESH_DEPENDS_ON_PARAMETERS TRUE)
else()
  message("-- MESH_DEPENDS_ON_PARAMETERS      is NOT Enabled.")
  set(ALBANY_MESH_DEPENDS_ON_PARAMETERS FALSE)
endif()

# set optional dependency of mesh on solution, defaults to Disabled
option(ENABLE_MESH_DEPENDS_ON_SOLUTION
       "Flag to turn on dependency of mesh on solution" OFF)
if(ENABLE_MESH_DEPENDS_ON_SOLUTION)
  message("-- MESH_DEPENDS_ON_SOLUTION        is Enabled")
  set(ALBANY_MESH_DEPENDS_ON_SOLUTION TRUE)
else()
  message("-- MESH_DEPENDS_ON_SOLUTION        is NOT Enabled.")
  set(ALBANY_MESH_DEPENDS_ON_SOLUTION FALSE)
endif()

# set optional dependency of dist parameters on solution, defaults to Disabled
option(ENABLE_PARAMETERS_DEPEND_ON_SOLUTION
       "Flag to turn on dependency of mesh on solution" OFF)
if(ENABLE_PARAMETERS_DEPEND_ON_SOLUTION)
  message("-- PARAMETERS_DEPEND_ON_SOLUTION   is Enabled")
  set(ALBANY_PARAMETERS_DEPEND_ON_SOLUTION TRUE)
elseif(ENABLE_MESH_DEPENDS_ON_SOLUTION)
  message(
    "-- PARAMETERS_DEPEND_ON_SOLUTION   is Enabled because MESH_DEPENDS_ON_SOLUTION is enabled"
  )
  set(ALBANY_PARAMETERS_DEPEND_ON_SOLUTION TRUE)
else()
  message("-- PARAMETERS_DEPEND_ON_SOLUTION   is NOT Enabled.")
  set(ALBANY_PARAMETERS_DEPEND_ON_SOLUTION FALSE)
endif()

# Set optional build of Aeras (Atmosphere Dynamics LDRD), defaults to Disabled
option(ENABLE_AERAS "Flag to turn on Aeras Source code" OFF)
option(
  ENABLE_AERAS_IMPLICIT_HS
  "Flag to turn on implicit time-int scheme for Aeras hydrostatic Source code"
  OFF)
if(ENABLE_AERAS)
  message("-- Aeras                           is Enabled")
  set(ALBANY_AERAS TRUE)
  if(ENABLE_AERAS_IMPLICIT_HS)
    message("  -- Aeras implicit hydrostatic    is Enabled")
    set(AERAS_IMPLICIT_HS TRUE)
  else()
    message("  -- Aeras implicit hydrostatic    is NOT Enabled")
    set(AERAS_IMPLICIT_HS FALSE)
  endif()
else()
  message("-- Aeras                           is NOT Enabled.")
  set(ALBANY_AERAS FALSE)
endif()

# optional dependency on LAME, defaults to FALSE
option(ENABLE_LAME "Flag to turn on LAME material library source code" OFF)
if(ENABLE_LAME)
  message("-- LAME                            is enabled")
  set(ALBANY_LAME TRUE)
  # The Sierra math toolkit is needed for LAME
  if(NOT MATH_TOOLKIT_INCLUDE_DIR)
    message("\nError:  MATH_TOOLKIT_INCLUDE_DIR not defined\n")
  else()
    message("  -- MATH_TOOLKIT_INCLUDE_DIR ${MATH_TOOLKIT_INCLUDE_DIR}")
  endif()
  if(NOT MATH_TOOLKIT_LIBRARY_DIR)
    message("\nError:  MATH_TOOLKIT_LIBRARY_DIR not defined\n")
  else()
    message("  -- MATH_TOOLKIT_LIBRARY_DIR ${MATH_TOOLKIT_LIBRARY_DIR}")
  endif()
  # LAME itself
  if(NOT LAME_INCLUDE_DIR)
    message("\nError:  LAME_INCLUDE_DIR not defined\n")
  else()
    message("  -- LAME_INCLUDE_DIR ${LAME_INCLUDE_DIR}")
  endif()
  if(NOT LAME_LIBRARY_DIR)
    message("\nError:  LAME_LIBRARY_DIR not defined\n")
  else()
    message("  -- LAME_LIBRARY_DIR ${LAME_LIBRARY_DIR}")
  endif()
else()
  message("-- LAME                            is NOT Enabled.")
  set(ALBANY_LAME FALSE)
endif()

# optional dependency on LAMENT, defaults to FALSE
option(ENABLE_LAMENT "Flag to turn on LAMENT lmaterial library source code" OFF)
if(ENABLE_LAMENT)
  message("-- LAMENT                          is enabled")
  set(ALBANY_LAMENT TRUE)
  if(NOT LAMENT_INCLUDE_DIR)
    message("\nError:  LAMENT_INCLUDE_DIR not defined\n")
  else()
    message("  -- LAMENT_INCLUDE_DIR ${LAMENT_INCLUDE_DIR}")
  endif()
  if(NOT LAMENT_LIBRARY_DIR)
    message("\nError:  LAMENT_LIBRARY_DIR not defined\n")
  else()
    message("  -- LAMENT_LIBRARY_DIR ${LAMENT_LIBRARY_DIR}")
  endif()
else()
  message("-- LAMENT                          is NOT Enabled.")
  set(ALBANY_LAMENT FALSE)
endif()

# Add option for ASCR work, defaults to disabled
option(ENABLE_ASCR
       "Flag to turn on ASCR-funded coupled UQ research source code" OFF)
if(ENABLE_ASCR)
  message("-- ASCR                            is Enabled")
  set(ALBANY_ASCR TRUE)
else()
  message("-- ASCR                            is NOT Enabled.")
  set(ALBANY_ASCR FALSE)
endif()

# Set optional dependency on AMP, defaults to Disabled
option(ENABLE_AMP
       "Flag to turn on AMP (additive manufacturing processes) source code" OFF)
if(ENABLE_AMP)
  set(ALBANY_APF TRUE)
  message("-- AMP                             is Enabled")
  set(ALBANY_AMP TRUE)
else()
  message("-- AMP                             is NOT Enabled.")
  set(ALBANY_AMP FALSE)
endif()

# Set optional dependency on SCOREC PUMI,  optional adaptivity
option(ENABLE_SCOREC "Flag to turn on SCOREC dependency" OFF)
set(ALBANY_SCOREC_DIR
    ""
    CACHE PATH "Path to standalone SCOREC installation")
if(ENABLE_SCOREC)
  # Check whether SCOREC was built as part of Trilinos
  list(FIND Trilinos_PACKAGE_LIST SCOREC SCOREC_List_ID)
  if(SCOREC_List_ID GREATER -1)
    # It was built in Trilinos
    list(FIND Trilinos_TPL_LIST SimModel SimModel_List_ID)
    # Check whether Simmetrix is also used
    if(SimModel_List_ID GREATER -1)
      message(STATUS "Simmetrix SimModel is a Trilinos TPL")
      set(ALBANY_SCOREC_SIMMODEL ON)
    endif()
    set(ALBANY_SCOREC_STANDALONE FALSE)
  else()
    # Not built in Trilinos, try to find standalone build in SCOREC_PREFIX
    find_package(SCOREC 2.0.0 PATHS "${ALBANY_SCOREC_DIR}")
    if(NOT SCOREC_FOUND)
      message(FATAL_ERROR "\nError: can't find either SCOREC package\n")
    endif()
    message(STATUS "SCOREC Found at ${SCOREC_CONFIG}")
    set(ALBANY_SCOREC_STANDALONE TRUE)
  endif()
  set(ALBANY_APF TRUE)
  message("-- SCOREC                          is enabled")
  set(ALBANY_SCOREC TRUE)
else()
  message("-- SCOREC                          is NOT Enabled.")
  set(ALBANY_SCOREC FALSE)
endif()

# RPI APF mesh interface
if(ALBANY_APF)
  message("-- APF                             is Enabled")
else()
  message("-- APF                             is NOT Enabled.")
endif()

option(ENABLE_OMEGA_H "Enable Omega_h dependency for adaptation" OFF)
set(ALBANY_OMEGA_H_DIR
    ""
    CACHE PATH "Path to Omega_h installation")
if(ENABLE_OMEGA_H)
  find_package(Omega_h 9.0.0 REQUIRED PATHS "${ALBANY_OMEGA_H_DIR}")
  message(STATUS "Omega_h found at ${Omega_h_CONFIG}")
  set(ALBANY_OMEGA_H TRUE)
endif()

# ParaView's Catalyst CoProcessing package
option(ENABLE_CATALYST
       "Flag to turn on support for the Catalyst CoProcessing library" OFF)
if(ENABLE_CATALYST)
  message("-- Catalyst                        is Enabled")
  set(ALBANY_CATALYST TRUE)
else()
  message("-- Catalyst                        is NOT Enabled.")
  set(ALBANY_CATALYST FALSE)
endif()

# Add option for CONTACT work, defaults to disabled
option(ENABLE_CONTACT "Flag to turn on CONTACT code in LCM" OFF)
if(ENABLE_CONTACT)
  message("-- CONTACT                         is Enabled")
  set(ALBANY_CONTACT TRUE)
else()
  message("-- CONTACT                         is NOT Enabled.")
  set(ALBANY_CONTACT FALSE)
endif()

# Other Albany Template Options here:
message("\nAlbany Templated Data Type Options:")

# Throw error if using ENABLE_FAST_FELIX
if(ENABLE_FAST_FELIX)
  message(
    FATAL_ERROR
      "\nError: ENABLE_FAST_FELIX has been removed! Use ENABLE_FAD_TYPE instead.
  Example for a tetrahedral mesh in LandIce:
    -D ENABLE_FAD_TYPE:STRING=SFad
    -D ALBANY_SFAD_SIZE=8
  For a hexahedral mesh, use ALBANY_SFAD_SIZE=16")
endif()

# Set FAD data type
set(ENABLE_FAD_TYPE
    "DFad"
    CACHE STRING "Sacado forward mode automatic differentiation data type")
if(ENABLE_FAD_TYPE STREQUAL "SFad")
  set(ALBANY_FAD_TYPE_SFAD TRUE)
  set(ALBANY_SFAD_SIZE
      32
      CACHE INT "Number of derivative components chosen at compile-time for AD")
  message("-- FAD_TYPE  is SFad, ALBANY_SFAD_SIZE=${ALBANY_SFAD_SIZE}")
  message(
    "---> WARNING: problems with elemental DOFs > ${ALBANY_SFAD_SIZE} will fail"
  )
elseif(ENABLE_FAD_TYPE STREQUAL "SLFad")
  set(ALBANY_FAD_TYPE_SLFAD TRUE)
  set(ALBANY_SLFAD_SIZE
      32
      CACHE
        INT
        "Maximum number of derivative components chosen at compile-time for AD")
  message("-- FAD_TYPE  is SLFad, ALBANY_SLFAD_SIZE=${ALBANY_SLFAD_SIZE}")
  message(
    "---> WARNING: problems with elemental DOFs > ${ALBANY_SLFAD_SIZE} will fail"
  )
elseif(ENABLE_FAD_TYPE STREQUAL "DFad")
  message("-- FAD_TYPE  is DFad (default)")
else()
  message(
    FATAL_ERROR
      "\nError: ENABLE_FAD_TYPE = ${ENABLE_FAD_TYPE} is not recognized!
  Options: SFad, SLFad, DFad (default)")
endif()

# Set FAD data type for Tangent
set(ENABLE_TAN_FAD_TYPE
    "DFad"
    CACHE STRING
          "Sacado forward mode automatic differentiation data type for Tangent")
if(ENABLE_TAN_FAD_TYPE STREQUAL "SFad")
  set(ALBANY_TAN_FAD_TYPE_SFAD TRUE)
  set(ALBANY_TAN_SFAD_SIZE
      32
      CACHE
        INT
        "Number of derivative components chosen at compile-time for Tangent AD")
  message(
    "-- TAN_FAD_TYPE is SFad, ALBANY_TAN_SFAD_SIZE=${ALBANY_TAN_SFAD_SIZE}")
  message(
    "---> WARNING: problems with Tangent elemental DOFs > ${ALBANY_TAN_SFAD_SIZE} will fail"
  )
elseif(ENABLE_TAN_FAD_TYPE STREQUAL "SLFad")
  set(ALBANY_TAN_FAD_TYPE_SLFAD TRUE)
  set(ALBANY_TAN_SLFAD_SIZE
      32
      CACHE
        INT
        "Maximum number of derivative components chosen at compile-time for Tangent AD"
  )
  message(
    "-- TAN_FAD_TYPE is SLFad, ALBANY_TAN_SLFAD_SIZE=${ALBANY_TAN_SLFAD_SIZE}")
  message(
    "---> WARNING: problems with Tangent elemental DOFs > ${ALBANY_TAN_SLFAD_SIZE} will fail"
  )
elseif(ENABLE_TAN_FAD_TYPE STREQUAL "DFad")
  message("-- TAN_FAD_TYPE is DFad (default)")
else()
  message(
    FATAL_ERROR
      "\nError: ENABLE_TAN_FAD_TYPE = ${ENABLE_TAN_FAD_TYPE} is not recognized!
  Options: SFad, SLFad, DFad")
endif()

# Check if FAD data type is the same
if(ENABLE_FAD_TYPE STREQUAL ENABLE_TAN_FAD_TYPE)
  if(ALBANY_FAD_TYPE_SFAD AND NOT ALBANY_SFAD_SIZE EQUAL ALBANY_TAN_SFAD_SIZE)
    set(ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE TRUE)
    message("-- SFAD_SIZE is not TAN_SFAD_SIZE")
  elseif(ALBANY_FAD_TYPE_SLFAD AND NOT ALBANY_SLFAD_SIZE EQUAL
                                   ALBANY_TAN_SLFAD_SIZE)
    set(ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE TRUE)
    message("-- SLFAD_SIZE is not TAN_SLFAD_SIZE")
  endif()
else()
  set(ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE TRUE)
  message("-- FAD_TYPE is not TAN_FAD_TYPE")
endif()

# optionally disable the use of the Trilinos stokhos package
option(ENABLE_STOKHOS "Flag to enable / disable the use of Stokhos in Albany"
       OFF)
if(ENABLE_STOKHOS)
  message("-- Stokhos   is Enabled")
  set(ALBANY_STOKHOS TRUE)
  list(FIND Trilinos_PACKAGE_LIST Stokhos STOKHOS_List_ID)
  if(NOT STOKHOS_List_ID GREATER -1)
    message(FATAL_ERROR "\nError: STOKHOS option requires Stokhos\n")
  endif()
else()
  set(ALBANY_STOKHOS FALSE)
endif()

# Disable the RTC capability if Trilinos is not built with Pamgen
list(FIND Trilinos_PACKAGE_LIST Pamgen PAMGEN_List_ID)
if(NOT PAMGEN_List_ID GREATER -1)
  set(ALBANY_PAMGEN FALSE)
  message("-- Pamgen package not found, disabling run time compiler support.")
else()
  set(ALBANY_PAMGEN TRUE)
  message("-- Pamgen package found, enabling run time compiler support.")
endif()

# Other Miscellaneous Albany Options here:
message("\nMiscellaneous Albany Options:")

# Optional OpenMP settings
if(ALBANY_ENABLE_OPENMP)
  if(ENABLE_LARGE_TESTS AND ENABLE_LCM)
    message(
      "---> LCM WARNING: turning off OrientationOnMesh tests since KokkosNode = OpenMP (see issue #339)!"
    )
  endif(ENABLE_LARGE_TESTS AND ENABLE_LCM)
  if(ENABLE_LCM)
    message(
      "---> LCM WARNING: turning off ACE tests since KokkosNode = OpenMP (see issue #501)!"
    )
  endif(ENABLE_LCM)
endif()

# Optional CUDA settings
if(ALBANY_ENABLE_CUDA)
  if(ENABLE_LCM)
    message(
      FATAL_ERROR
        "\nError: LCM does not work with a CUDA KokkosNode!  Recompile with LCM off."
    )
  endif()
  set(NUM_GPUS_PER_NODE
      4
      CACHE INT "set Number of GPUs per Node")
  message("-- NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}")
  set(KOKKOS_NDEVICES "--kokkos-ndevices=${NUM_GPUS_PER_NODE}")
  if(ENABLE_CUDA_ERROR_CHECK)
    set(ALBANY_CUDA_ERROR_CHECK ON)
    message("-- CUDA Error Check        is Enabled")
  else()
    message("-- CUDA Error Check        is NOT Enabled.")
  endif()
  if(ENABLE_CUDA_NVTX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lnvToolsExt")
    set(ALBANY_CUDA_NVTX ON)
    message("-- CUDA NVTX               is Enabled")
  else()
    message("-- CUDA NVTX               is NOT Enabled.")
  endif()
endif()

# set optional dependency on the BGL, defaults to Enabled This option is added
# due to issued with compiling BGL with the intel compilers see Trilinos
# bugzilla bug #6343
#
option(ENABLE_BGL
       "Flag to turn on the Phalanx graphviz capabilities (dependent on BGL)"
       ON)
if(ENABLE_BGL)
  message("-- BGL                       is Enabled")
  set(ALBANY_BGL TRUE)
else()
  message(
    "-- BGL                       is NOT Enabled. Graph visualization in Phalanx will not be available!"
  )
  set(ALBANY_BGL FALSE)
endif()

# Add option to check for hardware floating point exceptions using SSE, defaults
# to disabled
option(ENABLE_CHECK_FPE
       "Flag to turn on floating point exception (NAN) checking" OFF)

# Add option to check for hardware floating point exceptions using SSE, defaults
# to disabled
option(ENABLE_CHECK_FPE
       "Flag to turn on floating point exception (NAN) checking" OFF)
if(ENABLE_CHECK_FPE)
  message("-- FPE checking              is Enabled")
  set(ALBANY_CHECK_FPE TRUE)
else()
  message("-- FPE checking              is NOT Enabled.")
  set(ALBANY_CHECK_FPE FALSE)
endif()

option(ENABLE_STRONG_FPE_CHECK
       "Flag to turn on strong checking for floating point exceptions" OFF)
if(ENABLE_STRONG_FPE_CHECK)
  message("-- Strong FPE checking       is Enabled")
  set(ALBANY_STRONG_FPE_CHECK TRUE)
else()
  message("-- Strong FPE checking       is NOT Enabled.")
  set(ALBANY_STRONG_FPE_CHECK FALSE)
endif()

# Add option to set FTZ and DAZ
option(ENABLE_FLUSH_DENORMALS "Flag to turn on FTZ and DAZ" OFF)
if(ENABLE_FLUSH_DENORMALS)
  message("-- FTZ/DAZ flags             is Enabled")
  set(ALBANY_FLUSH_DENORMALS TRUE)
else()
  message("-- FTZ/DAZ flags             is NOT Enabled.")
  set(ALBANY_FLUSH_DENORMALS FALSE)
endif()

# amb Default to on until the merge.
option(ENABLE_KOKKOS_UNDER_DEVELOPMENT
       "Flag to turn on Kokkos experimental code in Albany" ON)
if(ENABLE_KOKKOS_UNDER_DEVELOPMENT)
  message("-- Kokkos development        is Enabled")
  set(ALBANY_KOKKOS_UNDER_DEVELOPMENT TRUE)
else()
  message("-- Kokkos development        is NOT Enabled.")
  set(ALBANY_KOKKOS_UNDER_DEVELOPMENT FALSE)
endif()

option(ENABLE_ALBANY_VERBOSE "Flag to turn on verbose output" OFF)
if(NOT ENABLE_ALBANY_VERBOSE)
  message("-- Verbose printing          is NOT Enabled")
  set(ALBANY_VERBOSE FALSE)
else()
  message("-- Verbose printing          is Enabled")
  set(ALBANY_VERBOSE TRUE)
endif()

# Option to enable "speculative" features that are not ready for general
# visibility
option(ENABLE_LCM_SPECULATIVE "Flag to turn on speculative LCM features/code"
       OFF)
if(NOT ENABLE_LCM_SPECULATIVE)
  message("-- Speculative LCM features  are NOT Enabled")
  set(LCM_SPECULATIVE FALSE)
else()
  message("-- Speculative LCM features  are Enabled")
  set(LCM_SPECULATIVE TRUE)
endif()

# Enhanced debugging option
option(ENABLE_DEBUGGING "Flag to turn on enhanced debugging info" OFF)
if(ENABLE_DEBUGGING)
  message("-- Enhanced debugging        is Enabled")
  set(ALBANY_DEBUG TRUE)
else()
  message("-- Enhanced debugging        is NOT Enabled.")
  set(ALBANY_DEBUG FALSE)
endif()

# Add option to build libraries but not link any executables (for LandIce uner
# MPAS)
option(ALBANY_LIBRARIES_ONLY
       "Flag to only build libraries and not link executables" OFF)
if(ALBANY_LIBRARIES_ONLY)
  message("-- ALBANY_LIBRARIES_ONLY is ON")
  if(LCM_TEST_EXES)
    set(LCM_TEST_EXES FALSE)
    message("--->  Disabling LCM_TEST_EXES")
  endif()
endif()

# Add option to include Dakota restart tests in the test suite
if(NOT APPLE)
  cmake_dependent_option(
    ENABLE_DAKOTA_RESTART_EXAMPLES "Flag to include Dakota restart tests" ON
    "ALBANY_DAKOTA" OFF)
else(NOT APPLE)
  # Portable Dakota restart files is broken on MacOS, deactivate test by default
  cmake_dependent_option(
    ENABLE_DAKOTA_RESTART_EXAMPLES "Flag to include Dakota restart tests" OFF
    "ALBANY_DAKOTA" OFF)
endif(NOT APPLE)
if(ENABLE_DAKOTA_RESTART_EXAMPLES)
  message("-- Dakota restart testing    is Enabled")
  set(ALBANY_DAKOTA_RESTART_EXAMPLES TRUE)
else(ENABLE_DAKOTA_RESTART_EXAMPLES)
  if(ALBANY_DAKOTA)
    message("-- Dakota restart testing    is NOT Enabled")
  endif(ALBANY_DAKOTA)
  set(ALBANY_DAKOTA_RESTART_EXAMPLES FALSE)
endif(ENABLE_DAKOTA_RESTART_EXAMPLES)

# Set which memory analysis system routines are available.
option(ENABLE_MALLINFO "Use mallinfo() in Albany::printMemoryAnalysis()" off)
option(ENABLE_GETRUSAGE "Use getrusage() in Albany::printMemoryAnalysis()" off)
option(ENABLE_KERNELGETMEMORYSIZE
       "Use Kernel_GetMemorySize() in Albany::printMemoryAnalysis()" off)
if(ENABLE_MALLINFO)
  set(ALBANY_HAVE_MALLINFO TRUE)
  message("-- Memory: mallinfo()        is Enabled.")
endif()
if(ENABLE_GETRUSAGE)
  set(ALBANY_HAVE_GETRUSAGE TRUE)
  message("-- Memory: getrusage()       is Enabled.")
endif()
if(ENABLE_KERNELGETMEMORYSIZE)
  set(ALBANY_HAVE_KERNELGETMEMORYSIZE ON)
  message("-- Memory: Kernel_GetMemorySize() is Enabled.")
endif()

# Mesh database tools.
option(ENABLE_MESHDB_TOOLS "Flag to turn on mesh database tools" OFF)
if(ENABLE_MESHDB_TOOLS)
  message("-- Mesh database tools       are Enabled.")
  set(ALBANY_MESHDB_TOOLS TRUE)
endif()

# add a target to generate API documentation with Doxygen
if(NOT DISABLE_DOXYGEN)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/Doxyfile.in
                   ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/albanyMainPage.doc
                   ${CMAKE_CURRENT_BINARY_DIR}/albanyMainPage.doc)
    add_custom_target(
      doc
      ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM)
  endif(DOXYGEN_FOUND)
endif()

# End of CMake options to Albany

add_subdirectory(src)

message("\n ***********************")
message(" *** Testing Options ***")
message(" ***********************\n")

# Add option to enable performance tests. Results are machine-specific
option(ENABLE_PERFORMANCE_TESTS "Flag to turn on Performance Tests" OFF)
if(ENABLE_PERFORMANCE_TESTS)
  message(
    "-- Performance Tests         are Enabled, adding tests/*/PerformanceTests")
  set(ALBANY_PERFORMANCE_TESTS TRUE)
else()
  message("-- Performance Tests         are NOT Enabled.")
  set(ALBANY_PERFORMANCE_TESTS FALSE)
endif()

# Add option to enable performance tests. Results are machine-specific
option(ENABLE_LARGE_TESTS "Run tests from the Albany-large-tests repository" ON)
if(ENABLE_LARGE_TESTS)
  message("-- Large Tests               are Enabled, adding tests/large/")
  set(ALBANY_LARGE_TESTS TRUE)
else()
  message("-- Large Tests               are NOT Enabled.")
  set(ALBANY_LARGE_TESTS FALSE)
endif()

# Set option to disable compiling some code for nightly tests Some code sections
# take so long to compile with NVCC that nightly tests are timing out. This
# option allows disabling those sections.
option(NIGHTLY_TESTING "Flag to turn off selected code during nightly testing"
       OFF)
if(NIGHTLY_TESTING)
  message("-- NIGHTLY_TESTING           is Enabled")
  set(ALBANY_NIGHTLY_TESTING TRUE)
else()
  message("-- NIGHTLY_TESTING           is NOT Enabled.")
  set(ALBANY_NIGHTLY_TESTING FALSE)
endif()

# Timeout real long running tests
if(ALBANY_CTEST_TIMEOUT)
  message(
    "\n-- Setting maximum test run time to be       ${ALBANY_CTEST_TIMEOUT} seconds"
  )
else()
  message("\n-- No limits are set on maximum test run time")
endif()

# Make a clean build target for convenience

add_custom_target(
  cmake_clean
  COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt
  COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/Makefile
  COMMAND ${CMAKE_COMMAND} -E remove_directory
          ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles
  COMMAND ${CMAKE_COMMAND} -E remove
          ${CMAKE_CURRENT_BINARY_DIR}/cmake_install.cmake
  COMMAND ${CMAKE_COMMAND} -E remove
          ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake)

option(ALBANY_PARALLEL_ONLY "Only run the parallel tests" OFF)

if(ALBANY_MPI)
  if(ALBANY_MPI_EXEC)
    set(MPIEX ${ALBANY_MPI_EXEC})
  else()
    set(MPIEX ${Trilinos_MPI_EXEC})
  endif()
  if(ALBANY_MPI_EXEC_NUMPROCS_FLAG)
    set(MPINPF ${ALBANY_MPI_EXEC_NUMPROCS_FLAG})
  else()
    set(MPINPF ${Trilinos_MPI_EXEC_NUMPROCS_FLAG})
  endif()
  if(ALBANY_MPI_EXEC_MAX_NUMPROCS)
    set(MPIMNP ${ALBANY_MPI_EXEC_MAX_NUMPROCS})
  else()
    set(MPIMNP ${Trilinos_MPI_EXEC_MAX_NUMPROCS})
  endif()
  if(ALBANY_MPI_EXEC_LEADING_OPTIONS)
    set(MPIPRE ${ALBANY_MPI_EXEC_LEADING_OPTIONS})
    separate_arguments(MPIPRE)
  endif()
  if(ALBANY_MPI_EXEC_TRAILING_OPTIONS)
    set(MPIPOST ${ALBANY_MPI_EXEC_TRAILING_OPTIONS})
    separate_arguments(MPIPOST)
  endif()
  set(SERIAL_CALL ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${MPIPOST})
  set(PARALLEL_CALL ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST})
  string(REPLACE ";" " " SERIAL_CALL_TEXT "${SERIAL_CALL}")
  string(REPLACE ";" " " PARALLEL_CALL_TEXT "${PARALLEL_CALL}")
  message("-- CTest serial call: ${SERIAL_CALL_TEXT}")
  message("-- CTest parallel call: ${PARALLEL_CALL_TEXT}")
endif(ALBANY_MPI)

# Add option to run any pretest executables in a specific environment
# (mpiexec.hydra, srun, or other)
if(DEFINED ALBANY_PRETEST_EXEC)
  message("-- ALBANY_PRETEST_EXEC is ${ALBANY_PRETEST_EXEC}")
else()
  # Not set in configure - set it to be empty
  set(ALBANY_PRETEST_EXEC
      ""
      CACHE STRING "Command to use to execute pretests")
endif()

# Add option to specify path to desired SEACAS utils
if(DEFINED ALBANY_SEACAS_PATH)
  message("-- ALBANY_SEACAS_PATH is ${ALBANY_SEACAS_PATH}")
endif()

# Add Albany test -- in the "tests" directory
if(ALBANY_LIBRARIES_ONLY)
  message("\nALBANY_LIBRARIES_ONLY has been set: turning off all tests.")
else()
  add_subdirectory(tests)
endif()

message("\nAnd CMake says...:")
