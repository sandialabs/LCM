# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Copy input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dummy_field.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/dummy_field.ascii COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_populate_mesh.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_populate_mesh.yaml COPYONLY)

add_test(${testName} ${Albany.exe} input_populate_mesh.yaml)
