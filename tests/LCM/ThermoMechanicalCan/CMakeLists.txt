#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

# Mechanics
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Mechanics.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/Mechanics.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsACE.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsACE.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/timevals.txt
               ${CMAKE_CURRENT_BINARY_DIR}/timevals.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hsvals.txt
               ${CMAKE_CURRENT_BINARY_DIR}/hsvals.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Hbvals.txt
               ${CMAKE_CURRENT_BINARY_DIR}/Hbvals.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Lvals.txt
	       ${CMAKE_CURRENT_BINARY_DIR}/Lvals.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/kvals.txt
	       ${CMAKE_CURRENT_BINARY_DIR}/kvals.txt COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperature.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperature.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsMaterials.yaml COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperatureMaterials.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperatureMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/brick_model_2.g
               ${CMAKE_CURRENT_BINARY_DIR}/brick_model_2.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/brick_model_2.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/brick_model_2.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/brick_model_2.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/brick_model_2.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/brick_model_2.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/brick_model_2.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/brick_model_2.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/brick_model_2.g.4.3 COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if(ALBANY_MUELU_EXAMPLES)
  add_test(${testName}_mech ${Albany.exe} Mechanics.yaml)
  set_tests_properties(${testName}_mech PROPERTIES LABELS "LCM;Tpetra;Forward")
  add_test(${testName}_mech_ACE_PressureNBC ${Albany.exe} MechanicsACE.yaml)
  set_tests_properties(${testName}_mech_ACE_PressureNBC PROPERTIES LABELS "LCM;Tpetra;Forward")
  add_test(${testName}_thermomech ${Albany.exe} MechanicsWithTemperature.yaml)
  set_tests_properties(${testName}_thermomech PROPERTIES LABELS
                                                         "LCM;Tpetra;Forward")
endif()
