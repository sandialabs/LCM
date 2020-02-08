# Disable these unit tests on the BGQ for now.
if(NOT ALBANY_PARALLEL_ONLY)
  add_test(utLocalNonlinearSolver
           ${Albany_BINARY_DIR}/src/LCM/utLocalNonlinearSolver)
  add_test(utMiniSolvers ${Albany_BINARY_DIR}/src/LCM/utMiniSolvers)
  if(ALBANY_ROL)
    add_test(utMiniSolversROL ${Albany_BINARY_DIR}/src/LCM/utMiniSolversROL)
  endif()
  add_test(utSurfaceElement ${Albany_BINARY_DIR}/src/LCM/utSurfaceElement)
  add_test(utHeliumODEs ${Albany_BINARY_DIR}/src/LCM/utHeliumODEs)
  if(ALBANY_LAME)
    add_test(utLameStress_elastic
             ${Albany_BINARY_DIR}/src/LCM/utLameStress_elastic)
  endif()
  # create a custom target "make utest" that runs only the unit tests
  add_custom_target(utest COMMAND ctest)
endif()
