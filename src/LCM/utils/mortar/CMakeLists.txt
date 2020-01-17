#
# B) Define the header and source files (and directories)
#

#
# src
#

set(HEADERS "")
set(SOURCES "")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(HEADERS ${HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/../${PACKAGE_NAME}_config.h)

#
# Core files
#

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

append_set(
  HEADERS
  Moertel_Tolerances.hpp
  Moertel_ExplicitTemplateInstantiation.hpp
  Moertel_FunctionT.hpp
  Moertel_IntegratorT.hpp
  Moertel_IntegratorT_Def.hpp
  Moertel_InterfaceT.hpp
  Moertel_InterfaceT_Complete_Def.hpp
  Moertel_InterfaceT_Project_Def.hpp
  Moertel_InterfaceT_Integrate_Def.hpp
  Moertel_InterfaceT_Integrate3D_Def.hpp
  Moertel_InterfaceT_Tools_Def.hpp
  Moertel_ManagerT.hpp
  Moertel_ManagerT_Def.hpp
  Moertel_UtilsT.hpp
  Moertel_UtilsT_Def.hpp
  Moertel_NodeT.hpp
  Moertel_NodeT_Def.hpp
  Moertel_PnodeT.hpp
  Moertel_PnodeT_Def.hpp
  Moertel_PointT.hpp
  Moertel_PointT_Def.hpp
  Moertel_SegmentT.hpp
  Moertel_SegmentT_Def.hpp
  Moertel_OverlapT.hpp
  Moertel_OverlapT_Def.hpp
  Moertel_OverlapT_Utils_Def.hpp
  Moertel_Convexhull_Def.hpp
  Moertel_ProjectorT.hpp
  Moertel_ProjectorT_Def.hpp
  Moertel_Projector3DT_Def.hpp)

append_set(
  SOURCES
  Moertel_ManagerT.cpp
  Moertel_InterfaceT.cpp
  Moertel_IntegratorT.cpp
  Moertel_NodeT.cpp
  Moertel_PnodeT.cpp
  Moertel_PointT.cpp
  Moertel_SegmentT.cpp
  Moertel_OverlapT.cpp
  Moertel_ProjectorT.cpp
  Moertel_UtilsT.cpp)

#
# C) Define the targets for package's library(s)
#

tribits_add_library(t_moertel HEADERS ${HEADERS} SOURCES ${SOURCES})
