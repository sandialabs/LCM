# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_linear_implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_linear_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_linear_explicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_linear_explicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_nonlinear_implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_nonlinear_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material_linear_implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material_linear_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material_linear_explicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material_linear_explicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material_linear_implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material_linear_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material_nonlinear_implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material_nonlinear_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped.g
               ${CMAKE_CURRENT_BINARY_DIR}/clamped.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_tet.g
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_tet.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/extract_fields_from_exo.m
               ${CMAKE_CURRENT_BINARY_DIR}/extract_fields_from_exo.m COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plot_soln_and_compute_errs_vs_exact.m
               ${CMAKE_CURRENT_BINARY_DIR}/plot_soln_and_compute_errs_vs_exact.m COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plot_soln_dot_and_compute_errs_vs_exact.m
               ${CMAKE_CURRENT_BINARY_DIR}/plot_soln_dot_and_compute_errs_vs_exact.m COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plot_solns_compare_to_matlab_code.m
               ${CMAKE_CURRENT_BINARY_DIR}/plot_solns_compare_to_matlab_code.m COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plot_solns.m
               ${CMAKE_CURRENT_BINARY_DIR}/plot_solns.m COPYONLY)
# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 1. Create the test with this name and standard executable
add_test(${testName}_StVenantKirchhoff_Implicit_Stabilized ${Albany.exe} clamped_linear_implicit.yaml)
set_tests_properties(${testName}_StVenantKirchhoff_Implicit_Stabilized PROPERTIES LABELS "LCM;Tpetra;Forward")
