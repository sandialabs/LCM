#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

# Adaptation

set(SOURCES
    AAdapt_AbstractAdapter.cpp
    AAdapt_AdaptiveSolutionManager.cpp
    AAdapt_InitialCondition.cpp
    AAdapt_AnalyticFunction.cpp
    AAdapt_SolutionObserver.cpp
    AAdapt_RC_DataTypes.hpp
    AAdapt_RC_Manager.cpp
    AAdapt_RC_Field.cpp
    AAdapt_RC_Reader.cpp
    AAdapt_RC_Writer.cpp
    AAdapt_RC_Projector_impl.cpp)

set(HEADERS
    AAdapt_AbstractAdapter.hpp
    AAdapt_AdaptiveSolutionManager.hpp
    AAdapt_InitialCondition.hpp
    AAdapt_AnalyticFunction.hpp
    AAdapt_SolutionObserver.hpp
    AAdapt_RC_Manager.hpp
    AAdapt_RC_Field.hpp
    AAdapt_RC_Reader.hpp
    AAdapt_RC_Writer.hpp
    AAdapt_RC_Projector_impl.hpp)

if(ALBANY_STK)
  set(SOURCES ${SOURCES} AAdapt_CopyRemesh.cpp)
  set(HEADERS ${HEADERS} AAdapt_CopyRemesh.hpp)
endif()

# Random Criterion & Stress Fracture depend on LCM Speculative features and
# code, currently
if(ALBANY_LCM AND ALBANY_STK)
  if(LCM_SPECULATIVE)
    set(SOURCES ${SOURCES} AAdapt_RandomCriterion.cpp AAdapt_RandomFracture.cpp
                AAdapt_StressFracture.cpp)

    set(HEADERS ${HEADERS} AAdapt_RandomCriterion.hpp AAdapt_RandomFracture.hpp
                AAdapt_StressFracture.hpp)
  endif()

  # Topology Modification depends on LCM and STK
  if(ALBANY_BGL)
    set(SOURCES ${SOURCES} AAdapt_Erosion.cpp AAdapt_TopologyModification.cpp)

    set(HEADERS ${HEADERS} AAdapt_Erosion.hpp AAdapt_TopologyModification.hpp)
  endif()

  if(ALBANY_STK_PERCEPT)
    set(SOURCES ${SOURCES} AAdapt_STKAdapt.cpp AAdapt_STKConstantSizeField.cpp)

    set(HEADERS ${HEADERS} AAdapt_STKAdapt.hpp AAdapt_STKAdapt_Def.hpp
                AAdapt_STKConstantSizeField.hpp)

  endif()
endif()

include_directories(
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}
  /
  src
  ${Albany_SOURCE_DIR}
  /
  src
  /
  evaluators
  ${Albany_SOURCE_DIR}
  /
  src
  /
  problems
  ${Albany_SOURCE_DIR}
  /
  src
  /
  responses
  ${Albany_SOURCE_DIR}
  /
  src
  /
  disc
  ${Albany_SOURCE_DIR}
  /
  src
  /
  utility
  ${Albany_SOURCE_DIR}
  /
  src
  /
  disc
  /
  stk
  ${Albany_SOURCE_DIR}
  /
  src
  /
  LCM
  /
  ${Albany_SOURCE_DIR}
  /
  src
  /
  LCM
  /
  evaluators
  ${Albany_SOURCE_DIR}
  /
  src
  /
  LCM
  /
  utils)

add_library(albanyAdapt ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

set_target_properties(albanyAdapt PROPERTIES PUBLIC_HEADER "${HEADERS}")

target_link_libraries(albanyAdapt ${SCOREC_LIB})

if(ALBANY_OMEGA_H)
  target_link_libraries(albanyAdapt Omega_h::omega_h)
endif()

if(INSTALL_ALBANY)
  install(
    TARGETS albanyAdapt
    EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
endif()
