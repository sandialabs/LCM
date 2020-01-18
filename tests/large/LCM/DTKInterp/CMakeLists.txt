#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

# This test is not currently set up to run on the BGQ...
if(NOT ALBANY_PARALLEL_ONLY
   AND ALBANY_LCM
   AND LCM_TEST_EXES
   AND ALBANY_DTK)
  set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
  add_subdirectory(Meshes)
  add_subdirectory(Error)
  add_subdirectory(InterpVolToNs)

endif()
