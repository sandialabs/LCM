# Copy Input files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicit_fd.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicit_fd.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_implicit_fd.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_implicit_fd.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
	       ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)

# 1. Cap Explicit
add_test(CapModelPlasticity3D_Explicit ${SerialAlbany.exe} input_explicit.yaml)
set_tests_properties(CapModelPlasticity3D_Explicit PROPERTIES LABELS "LCM;Tpetra;Forward")

# 2. Cap Implicit
add_test(CapModelPlasticity3D_Implicit ${SerialAlbany.exe} input_implicit.yaml)
set_tests_properties(CapModelPlasticity3D_Implicit PROPERTIES LABELS "LCM;Tpetra;Forward")

# 3. Cap Explicit FD
add_test(CapModelPlasticity3D_ExplicitFD ${SerialAlbany.exe} input_explicit_fd.yaml)
set_tests_properties(CapModelPlasticity3D_ExplicitFD PROPERTIES LABELS "LCM;Tpetra;Forward")

# 4. Cap Implicit FD
add_test(CapModelPlasticity3D_ImplicitFD ${SerialAlbany.exe} input_implicit_fd.yaml)
set_tests_properties(CapModelPlasticity3D_ImplicitFD PROPERTIES LABELS "LCM;Tpetra;Forward")
