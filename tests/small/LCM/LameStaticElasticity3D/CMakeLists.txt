# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMatrixFree.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMatrixFree.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputProbeLame.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputProbeLame.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputProbeLameT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputProbeLameT.yaml COPYONLY)
# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 1. Create the test with this name and standard executable
add_test(${testName}_MatrixFree ${Albany.exe} inputMatrixFree.yaml)
add_test(${testName}_ProbeLame ${Albany.exe} inputProbeLameT.yaml)
