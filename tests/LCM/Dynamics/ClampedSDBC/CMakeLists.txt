#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped-piro-tr-stk-shift.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/clamped-piro-tr-stk-shift.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material-clamped-stk.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material-clamped-stk.yaml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
add_test(Dynamics_${testName} ${Albany.exe} clamped-piro-tr-stk-shift.yaml)
set_tests_properties(Dynamics_${testName} PROPERTIES LABELS "LCM;Tpetra;Forward")
