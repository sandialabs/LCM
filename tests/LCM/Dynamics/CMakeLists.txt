#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

# Copy Input file from source to binary dir
add_subdirectory(ClampedSDBC)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dynamics.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/dynamics.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/elastic.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/elastic.yaml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
add_test(${testName} ${Albany.exe} dynamics.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "LCM;Tpetra;Forward")
