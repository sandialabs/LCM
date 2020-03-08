# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputRandom.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputRandom.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_s_Random.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_s_Random.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)

# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 1. Create the test with this name and standard executable

# add_test(${testName} ${Albany.exe} inputRandom.yaml)
