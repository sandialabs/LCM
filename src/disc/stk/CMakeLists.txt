#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of Sandia, LLC (NTESS). This Software is
# released under the BSD license detailed in the file license.txt in the top-level Albany directory.
#

# stk
set(SOURCES
    Albany_AsciiSTKMesh2D.cpp
    Albany_AsciiSTKMeshStruct.cpp
    Albany_GenericSTKFieldContainer.cpp
    Albany_GenericSTKMeshStruct.cpp
    Albany_GmshSTKMeshStruct.cpp
    Albany_IossSTKMeshStruct.cpp
    Albany_MultiSTKFieldContainer.cpp
    Albany_OrdinarySTKFieldContainer.cpp
    Albany_SideSetSTKMeshStruct.cpp
    Albany_STKDiscretization.cpp
    Albany_STKFieldContainerHelper.cpp
    Albany_STKNodeFieldContainer.cpp
    Albany_STKNodeSharing.cpp
    Albany_STK3DPointStruct.cpp
    Albany_TmplSTKMeshStruct.cpp)
set(HEADERS
    Albany_AbstractSTKFieldContainer.hpp
    Albany_AbstractSTKMeshStruct.hpp
    Albany_AsciiSTKMeshStruct.hpp
    Albany_AsciiSTKMesh2D.hpp
    Albany_GenericSTKMeshStruct.hpp
    Albany_GmshSTKMeshStruct.hpp
    Albany_GenericSTKFieldContainer.hpp
    Albany_GenericSTKFieldContainer_Def.hpp
    Albany_IossSTKMeshStruct.hpp
    Albany_MultiSTKFieldContainer.hpp
    Albany_MultiSTKFieldContainer_Def.hpp
    Albany_NodalGraphUtils.hpp
    Albany_OrdinarySTKFieldContainer.hpp
    Albany_OrdinarySTKFieldContainer_Def.hpp
    Albany_SideSetSTKMeshStruct.hpp
    Albany_STKDiscretization.hpp
    Albany_STKFieldContainerHelper.hpp
    Albany_STKNodeFieldContainer.hpp
    Albany_STKNodeFieldContainer_Def.hpp
    Albany_STKNodeSharing.hpp
    Albany_STK3DPointStruct.hpp
    Albany_TmplSTKMeshStruct.hpp
    Albany_TmplSTKMeshStruct_Def.hpp)

include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} ${Albany_SOURCE_DIR}/src)

add_library(albanySTK ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
target_link_libraries(albanySTK ${Trilinos_LIBRARIES})

set_target_properties(albanySTK PROPERTIES PUBLIC_HEADER "${HEADERS}")

if(INSTALL_ALBANY)
  install(
    TARGETS albanySTK
    EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
endif()

# Can remove following once stk_rebalance is in trilinos again
if(ALBANY_STK_REBALANCE)
  add_subdirectory(percept)
  target_link_libraries(albanySTK albanySTKRebalance)
endif()
