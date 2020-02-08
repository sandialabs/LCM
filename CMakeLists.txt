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

if(SET_COMPILERS_AUTOMATICALLY)
  message("Setting and checking of compilers:")
  set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
  set(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
  set(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER})
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

# Compiler details
set(ALBANY_CXX_COMPILER_ID ${CMAKE_CXX_COMPILER_ID})
message("-- ALBANY_CXX_COMPILER_ID = ${ALBANY_CXX_COMPILER_ID}")
set(ALBANY_CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
message("-- ALBANY_CXX_COMPILER_VERSION = ${ALBANY_CXX_COMPILER_VERSION}")
if(EXISTS "${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h")
  file(READ "${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h"
       KOKKOS_CORE_CONFIG_FILE)
  set(ALBANY_ENABLE_OPENMP OFF)
  if(ALBANY_ENABLE_OPENMP)
    message("-- Kokkos is configured to use OpenMP, Albany will also.")
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
      "Albany REQUIRES Piro. See file trilinos-cmake in doc directory.")
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
      "Albany REQUIRES Phalanx. See file trilinos-cmake in doc directory.")
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
    FATAL_ERROR "Albany REQUIRES NOX. See file trilinos-cmake in doc directory."
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
      "Albany REQUIRES Intrepid2. See file trilinos-cmake in doc directory.")
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
      "Albany REQUIRES MiniTensor. See file trilinos-cmake in doc directory.")
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

# set dependency on stk
list(FIND Trilinos_PACKAGE_LIST STK have_stk_List_ID)
if(have_stk_List_ID GREATER -1)
  message("-- Looking for STK:                  Found")

  # set dependency on SEACAS, for I/O via stk_io MESSAGE("-- Looking for
  # SEACAS:")
  list(FIND Trilinos_PACKAGE_LIST SEACASIoss SEACASIoss_List_ID)
  if(SEACASIoss_List_ID GREATER -1)
    message("  -- Looking for SEACASIoss:         Found")
    list(FIND SEACASIoss_LIBRARIES Iopx Iopx_List_ID)
    if(Iopx_List_ID GREATER -1)
      message("  -- Looking for Iopx:               Found")
      set(ALBANY_IOPX TRUE)
    else()
      message("  -- Looking for Iopx:               NOT found.")
      set(ALBANY_IOPX FALSE)
    endif()
  else()
    message(FATAL_ERROR "SEACAS is required.")
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
  message(FATAL_ERROR "STK is required.")
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
  set(ALBANY_STK_REBALANCE TRUE)
  message("-- Looking for Zoltan:               Found.")
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
else()
  message(FATAL_ERROR "Tempus is required.")
endif()

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

option(ENABLE_LCM_TEST_EXES "Flag to turn on LCM Mechanics test executables" ON)
option(DISABLE_LCM_EXODIFF_SENSITIVE_TESTS
       "Flag to turn off LCM exodiff sensitive tests" OFF)
set(LCM_TEST_EXES TRUE)
if(NOT ALBANY_ROL)
  message(FATAL_ERROR "ROL not installed from Trilinos (LCM needs ROL)")
endif()
if(DISABLE_LCM_EXODIFF_SENSITIVE_TESTS)
  message("  -- LCM exodiff sensitive tests are DISABLED!")
  set(DISABLE_LCM_EXODIFF_SENSITIVE_TESTS TRUE)
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

option(ENABLE_OMEGA_H "Enable Omega_h dependency for adaptation" OFF)
set(ALBANY_OMEGA_H_DIR
    ""
    CACHE PATH "Path to Omega_h installation")
if(ENABLE_OMEGA_H)
  find_package(Omega_h 9.0.0 REQUIRED PATHS "${ALBANY_OMEGA_H_DIR}")
  message(STATUS "Omega_h found at ${Omega_h_CONFIG}")
  set(ALBANY_OMEGA_H TRUE)
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
