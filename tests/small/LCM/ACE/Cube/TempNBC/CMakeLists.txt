#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g
               ${CMAKE_CURRENT_BINARY_DIR}/cube.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/disp.txt
               ${CMAKE_CURRENT_BINARY_DIR}/disp.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/temp.txt
               ${CMAKE_CURRENT_BINARY_DIR}/temp.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/level.txt
               ${CMAKE_CURRENT_BINARY_DIR}/level.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/time.txt
               ${CMAKE_CURRENT_BINARY_DIR}/time.txt COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_test(ACE_Cube_${testName} ${SerialAlbany.exe} cube.yaml)
set_tests_properties(ACE_Cube_${testName} PROPERTIES LABELS "LCM;Tpetra;Forward")

