#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Ablufftemp.txt
        	${CMAKE_CURRENT_BINARY_DIR}/Ablufftemp.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ocean_sal.txt
               ${CMAKE_CURRENT_BINARY_DIR}/ocean_sal.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/time.txt
               ${CMAKE_CURRENT_BINARY_DIR}/time.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/waterH.txt
        	${CMAKE_CURRENT_BINARY_DIR}/waterH.txt COPYONLY)

