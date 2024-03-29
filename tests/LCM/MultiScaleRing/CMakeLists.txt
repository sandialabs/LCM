# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_nobridge.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_nobridge.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_nobridge.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_nobridge.yaml COPYONLY)

# 1. Copy mesh file from source to binary dir. We only need a single file as
#   there is only one master process used in this test.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ring.exo
               ${CMAKE_CURRENT_BINARY_DIR}/ring.exo COPYONLY)

# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 1. Create the MPALE_dummy executable, for Albany to start up. The dummy
#   executable just echos back the stress values at the QPs.

add_executable(MPALE_dummy MPALE_dummy.cpp)

# 1. Run the Albany test

add_test(${testName}_nobridge_SERIAL ${SerialAlbany.exe} input_nobridge.yaml)
set_tests_properties(${testName}_nobridge_SERIAL
                     PROPERTIES LABELS "LCM;Tpetra;Forward")
