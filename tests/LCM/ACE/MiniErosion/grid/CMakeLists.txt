#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_wave_press_nbc.g
        	${CMAKE_CURRENT_BINARY_DIR}/cuboid_wave_press_nbc.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.g
        	${CMAKE_CURRENT_BINARY_DIR}/cuboid.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.g.4.0
        	${CMAKE_CURRENT_BINARY_DIR}/cuboid.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.g.4.1
        	${CMAKE_CURRENT_BINARY_DIR}/cuboid.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.g.4.2
        	${CMAKE_CURRENT_BINARY_DIR}/cuboid.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.g.4.3
        	${CMAKE_CURRENT_BINARY_DIR}/cuboid.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_denudation.g
        	${CMAKE_CURRENT_BINARY_DIR}/cuboid_denudation.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_denudation.g.2.0
        	${CMAKE_CURRENT_BINARY_DIR}/cuboid_denudation.g.2.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_denudation.g.2.1
        	${CMAKE_CURRENT_BINARY_DIR}/cuboid_denudation.g.2.1 COPYONLY)
