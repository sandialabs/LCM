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
#   confined_coarse - confined path with 20 global steps: increments
#                 large enough that the adaptive substepping engages
#                 (~150 substep evaluations across the history).
#   confined_fd, triaxial_fd - the same paths under finite-deformation
#                 (exponential/logarithmic map) kinematics; the
#                 reference mirrors the kinematics exactly.
#
# The comparisons require python3 with numpy and netCDF4; the tests are
# skipped (with a status message) when unavailable.

foreach(f input_hydrostatic.yaml input_confined.yaml input_triaxial.yaml
          input_confined_coarse.yaml input_confined_fd.yaml
          input_triaxial_fd.yaml input_confined_permafrost.yaml
          input_confined_thaw.yaml input_confined_f05.yaml
          input_death_tension.yaml input_death_backstress.yaml
          input_death_crush.yaml input_death_eqps.yaml
          input_death_displacement.yaml input_death_distortion.yaml
          materials.yaml materials_triaxial.yaml materials_fd.yaml
          materials_triaxial_fd.yaml materials_permafrost.yaml
          materials_permafrost_thaw.yaml materials_permafrost_f05.yaml
          materials_permafrost_death_tension.yaml
          materials_permafrost_death_backstress.yaml
          materials_permafrost_death_crush.yaml
          materials_permafrost_death_eqps.yaml
          materials_permafrost_death_displacement.yaml
          materials_permafrost_death_distortion.yaml
          input_porosity.yaml materials_permafrost_porosity.yaml
          depth_test.txt porosity_test.txt
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

  foreach(case hydrostatic confined triaxial confined_coarse confined_fd triaxial_fd)
    if (case STREQUAL "confined_coarse")
      set(_cap_pathmode confined)
    else()
      set(_cap_pathmode ${case})
    endif()
    add_test(NAME CapModel_Verify_${case}
             COMMAND ${CMAKE_COMMAND}
                     "-DTEST_PROG=${SerialAlbany.exe}"
                     "-DINPUT=input_${case}.yaml"
                     "-DEXO=cap_${case}.exo"
                     "-DPATHMODE=${_cap_pathmode}"
                     "-DPYTHON=${PYTHON3_FOR_CAP}"
                     -P runverify.cmake
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(CapModel_Verify_${case} PROPERTIES LABELS "LCM;Tpetra;Forward")
  endforeach()
  # Permafrost Phase-0 scaffold: must reproduce CapModel/reference exactly
  add_test(NAME Permafrost_Equivalence_confined
           COMMAND ${CMAKE_COMMAND}
                   "-DTEST_PROG=${SerialAlbany.exe}"
                   "-DINPUT=input_confined_permafrost.yaml"
                   "-DEXO=permafrost_confined.exo"
                   "-DPATHMODE=confined"
                   "-DPYTHON=${PYTHON3_FOR_CAP}"
                   -P runverify.cmake
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(Permafrost_Equivalence_confined PROPERTIES LABELS "LCM;Tpetra;Forward")

  # Thaw under load: ice saturation falls 1.0 -> 0.3 during confined
  # compression; verifies the saturation-to-parameter map (elasticity
  # split, nu cap, parameter blending) against the reference.
  add_test(NAME Permafrost_Verify_thaw
           COMMAND ${CMAKE_COMMAND}
                   "-DTEST_PROG=${SerialAlbany.exe}"
                   "-DINPUT=input_confined_thaw.yaml"
                   "-DEXO=permafrost_thaw.exo"
                   "-DPATHMODE=permafrost_thaw"
                   "-DPYTHON=${PYTHON3_FOR_CAP}"
                   -P runverify.cmake
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(Permafrost_Verify_thaw PROPERTIES LABELS "LCM;Tpetra;Forward")

  # Constant ice saturation 0.5: the full saturation-to-parameter map
  # (including the nu cap) without time variation.
  add_test(NAME Permafrost_Verify_f05
           COMMAND ${CMAKE_COMMAND}
                   "-DTEST_PROG=${SerialAlbany.exe}"
                   "-DINPUT=input_confined_f05.yaml"
                   "-DEXO=permafrost_f05.exo"
                   "-DPATHMODE=permafrost_f05"
                   "-DPYTHON=${PYTHON3_FOR_CAP}"
                   -P runverify.cmake
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(Permafrost_Verify_f05 PROPERTIES LABELS "LCM;Tpetra;Forward")

  # Porosity-profile override: W = porosity(z) per integration point; the
  # two Gauss-point z-groups are independent material points verified
  # against two reference runs.
  add_test(NAME Permafrost_Verify_porosity
           COMMAND ${CMAKE_COMMAND}
                   "-DTEST_PROG=${SerialAlbany.exe}"
                   "-DINPUT=input_porosity.yaml"
                   "-DEXO=permafrost_porosity.exo"
                   "-DPATHMODE=porosity_profile"
                   "-DPYTHON=${PYTHON3_FOR_CAP}"
                   -P runverify.cmake
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(Permafrost_Verify_porosity PROPERTIES LABELS "LCM;Tpetra;Forward")

  # Per-mode failure/death verification: each deck enables exactly one
  # failure criterion on an already-verified trajectory; the harness
  # checks the indicator series against the reference (machine precision)
  # and the failure_modes bitmask / failure_state / cell_death series
  # against the oracle-predicted trip step (exact).
  foreach(mode tension backstress crush eqps displacement distortion)
    add_test(NAME Permafrost_Death_${mode}
             COMMAND ${CMAKE_COMMAND}
                     "-DTEST_PROG=${SerialAlbany.exe}"
                     "-DINPUT=input_death_${mode}.yaml"
                     "-DEXO=permafrost_death_${mode}.exo"
                     "-DPATHMODE=death_${mode}"
                     "-DPYTHON=${PYTHON3_FOR_CAP}"
                     -P runverify.cmake
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(Permafrost_Death_${mode} PROPERTIES LABELS "LCM;Tpetra;Forward")
  endforeach()
else()
  message(STATUS "CapModel verification tests skipped: python3 with numpy+netCDF4 not found")
endif()
