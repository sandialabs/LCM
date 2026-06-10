# Cap plasticity model verification tests.
#
# The model response (single element, all DOFs prescribed -> true material
# point) is compared time-step by time-step against an independent
# reference implementation (cap_reference.py) transcribed directly from
# the publications, with complex-step derivatives. Three load paths:
#
#   hydrostatic - cap-only response; additionally checks the analytic
#                 identities I1 = X(kappa) and the crush curve.
#   confined    - 1D compression (Sun, Chen & Ostien 2014, Sec. 5.1):
#                 shear + cap interaction, kinematic hardening to its
#                 limit surface.
#   triaxial    - psi = 0.8 (Lode-angle terms active) + fully
#                 non-associative parameters (L, phi, Q distinct).
#
# The comparisons require python3 with numpy and netCDF4; the tests are
# skipped (with a status message) when unavailable.

foreach(f input_hydrostatic.yaml input_confined.yaml input_triaxial.yaml
          materials.yaml materials_triaxial.yaml
          cap_reference.py cap_verify.py runverify.cmake)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${f}
                 ${CMAKE_CURRENT_BINARY_DIR}/${f} COPYONLY)
endforeach()

find_program(PYTHON3_FOR_CAP python3)
set(CAP_PY_OK FALSE)
if (PYTHON3_FOR_CAP)
  execute_process(COMMAND ${PYTHON3_FOR_CAP} -c "import numpy, netCDF4"
                  RESULT_VARIABLE _cap_py_res
                  OUTPUT_QUIET ERROR_QUIET)
  if (_cap_py_res EQUAL 0)
    set(CAP_PY_OK TRUE)
  endif()
endif()

if (CAP_PY_OK)
  add_test(NAME CapModel_Oracle_SelfCheck
           COMMAND ${PYTHON3_FOR_CAP} cap_reference.py selfcheck
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(CapModel_Oracle_SelfCheck PROPERTIES LABELS "LCM;Tpetra;Forward")

  foreach(case hydrostatic confined triaxial)
    add_test(NAME CapModel_Verify_${case}
             COMMAND ${CMAKE_COMMAND}
                     "-DTEST_PROG=${SerialAlbany.exe}"
                     "-DINPUT=input_${case}.yaml"
                     "-DEXO=cap_${case}.exo"
                     "-DPATHMODE=${case}"
                     "-DPYTHON=${PYTHON3_FOR_CAP}"
                     -P runverify.cmake
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(CapModel_Verify_${case} PROPERTIES LABELS "LCM;Tpetra;Forward")
  endforeach()
else()
  message(STATUS "CapModel verification tests skipped: python3 with numpy+netCDF4 not found")
endif()
