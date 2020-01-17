# The root name for the tests in this directory is the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# ##############################################################################
# Dirichlet Dist Param tests ###
# ##############################################################################

set(testName ${testNameRoot}_Dirichlet_Dist_Param)

# Copy Input files from source to binary dir
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/input_dirichlet_dist_paramT.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/input_dirichlet_dist_paramT.yaml COPYONLY)

# Create the test with this name and standard executable
add_test(${testName}_Tpetra ${Albany.exe} input_dirichlet_dist_paramT.yaml)
set_tests_properties(${testName}_Tpetra PROPERTIES LABELS
                                                   "Basic;Tpetra;Forward")

# ##############################################################################
# Dirichlet Mixed Params tests ###
# ##############################################################################

set(testName ${testNameRoot}_Dirichlet_Mixed_Params)

# Copy Input file from source to binary dir
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/input_dirichlet_mixed_paramsT.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/input_dirichlet_mixed_paramsT.yaml COPYONLY)

# Create the test with this name and standard executable
add_test(${testName}_Tpetra ${Albany.exe} input_dirichlet_mixed_paramsT.yaml)
set_tests_properties(${testName}_Tpetra PROPERTIES LABELS
                                                   "Basic;Tpetra;Forward")

# ##############################################################################
# Dirichlet Mixed Params tests ###
# ##############################################################################

set(testName ${testNameRoot}_Conductivity_Dist_Param)

# Copy Input file from source to binary dir
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/input_conductivity_dist_paramT.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/input_conductivity_dist_paramT.yaml COPYONLY)
# Create the test with this name and standard executable
add_test(${testName}_Tpetra ${Albany.exe} input_conductivity_dist_paramT.yaml)
set_tests_properties(${testName}_Tpetra PROPERTIES LABELS
                                                   "Basic;Tpetra;Forward")

if(ALBANY_IOPX)
  # Copy Input file from source to binary dir
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/input_conductivity_dist_param_restartT.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/input_conductivity_dist_param_restartT.yaml
    COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/steady2d_conduct_opt.exo
                 ${CMAKE_CURRENT_BINARY_DIR}/steady2d_conduct_opt.exo COPYONLY)

  # Create the test with this name and standard executable
  add_test(${testName}_Restart_Tpetra ${Albany.exe}
           input_conductivity_dist_param_restartT.yaml)
  set_tests_properties(${testName}_Restart_Tpetra
                       PROPERTIES LABELS "Basic;Tpetra;Forward")
  set_tests_properties(${testName}_Restart_Tpetra PROPERTIES DEPENDS
                                                             ${testName}_Tpetra)
endif()
