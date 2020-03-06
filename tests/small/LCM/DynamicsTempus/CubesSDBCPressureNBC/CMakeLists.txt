#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-pnbc.g
               ${CMAKE_CURRENT_BINARY_DIR}/cube-pnbc.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material-pnbc.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material-pnbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-pnbc.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube-pnbc.yaml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_test(Serial_Dynamic_${testName}_NewmarkImplicitDForm_PressureNBC_Tempus
         ${SerialAlbany.exe} cube-pnbc.yaml)
set_tests_properties(Serial_Dynamic_${testName}_NewmarkImplicitDForm_PressureNBC_Tempus
                     PROPERTIES LABELS "LCM;Tpetra;Forward")
