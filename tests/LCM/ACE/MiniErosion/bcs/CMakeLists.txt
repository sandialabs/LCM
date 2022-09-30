#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Ablufftemp2.txt
	${CMAKE_CURRENT_BINARY_DIR}/Ablufftemp2.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Lvals.txt
	${CMAKE_CURRENT_BINARY_DIR}/Lvals.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ocean_sal.txt
	${CMAKE_CURRENT_BINARY_DIR}/ocean_sal.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/waterA.txt
	${CMAKE_CURRENT_BINARY_DIR}/waterA.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/waterH.txt
	${CMAKE_CURRENT_BINARY_DIR}/waterH.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/kvals.txt
	${CMAKE_CURRENT_BINARY_DIR}/kvals.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TimeHis_ExSec3600_2018MINI.txt
	${CMAKE_CURRENT_BINARY_DIR}/TimeHis_ExSec3600_2018MINI.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/waterHA.txt
	${CMAKE_CURRENT_BINARY_DIR}/waterHA.txt COPYONLY)
