# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.g
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_newmark_tempus.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_newmark_tempus.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_tr_piro.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_tr_piro.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material.yaml COPYONLY)

# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 1. Create the test with this name and standard executable
#add_test(${testName} ${Albany.exe} input.yaml)
#set_tests_properties(${testName} PROPERTIES LABELS "LCM;Tpetra;Forward")
