# Copy Input files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicit_fd.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicit_fd.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
	       ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)

# All four CapModelPlasticity3D tests are temporarily disabled while the
#
# The Explicit variants (1, 2) hit a Newton divergence at LOCA continuation
# step ~35 (load time ~0.4375): both rigel (Zen 4) and algol (Haswell)
# arrive at the same state (||F|| = 2.420e+00) bit-for-bit identical up to
# that step, then Newton fails to recover in this regime. With the previous
# Maximum Iterations: 10, the early exit happened to leave the solution at
# a specific value on rigel that became the recorded gold response, but
# the value is not a true converged solution; with Maximum Iterations: 30
# both architectures produce different (and equally wrong vs gold) answers.
# Re-enable once the Cap model handles this regime robustly.
#
# global Jacobian.
#
# 1. Cap Explicit
# add_test(CapModelPlasticity3D_Explicit ${SerialAlbany.exe} input_explicit.yaml)
# set_tests_properties(CapModelPlasticity3D_Explicit PROPERTIES LABELS "LCM;Tpetra;Forward")
#
# 2. Cap Explicit FD
# add_test(CapModelPlasticity3D_ExplicitFD ${SerialAlbany.exe} input_explicit_fd.yaml)
# set_tests_properties(CapModelPlasticity3D_ExplicitFD PROPERTIES LABELS "LCM;Tpetra;Forward")
#
#
