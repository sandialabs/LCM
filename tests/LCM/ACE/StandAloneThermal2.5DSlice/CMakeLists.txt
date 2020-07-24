if(NOT ALBANY_PARALLEL_ONLY)
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy Input file from source to binary dir
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Ablufftemp6_ExSec3600_Grid0.1_DBC_2011.txt
    ${CMAKE_CURRENT_BINARY_DIR}/Ablufftemp6_ExSec3600_Grid0.1_DBC_2011.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ARESTblufftemp6_ExSec3600_Grid0.1_DBC_2011.txt
    ${CMAKE_CURRENT_BINARY_DIR}/ARESTblufftemp6_ExSec3600_Grid0.1_DBC_2011.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/bluff_salinity.txt
    ${CMAKE_CURRENT_BINARY_DIR}/bluff_salinity.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/clay_frac.txt
    ${CMAKE_CURRENT_BINARY_DIR}/clay_frac.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/depth.txt
    ${CMAKE_CURRENT_BINARY_DIR}/depth.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Oblufftemp1_ExSec3600_Grid0.1_DBC_2011.txt
    ${CMAKE_CURRENT_BINARY_DIR}/Oblufftemp1_ExSec3600_Grid0.1_DBC_2011.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Oblufftemp2_ExSec3600_Grid0.1_DBC_2011.txt
    ${CMAKE_CURRENT_BINARY_DIR}/Oblufftemp2_ExSec3600_Grid0.1_DBC_2011.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Oblufftemp3_ExSec3600_Grid0.1_DBC_2011.txt
    ${CMAKE_CURRENT_BINARY_DIR}/Oblufftemp3_ExSec3600_Grid0.1_DBC_2011.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Oblufftemp4_ExSec3600_Grid0.1_DBC_2011.txt
    ${CMAKE_CURRENT_BINARY_DIR}/Oblufftemp4_ExSec3600_Grid0.1_DBC_2011.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Oblufftemp5_ExSec3600_Grid0.1_DBC_2011.txt
    ${CMAKE_CURRENT_BINARY_DIR}/Oblufftemp5_ExSec3600_Grid0.1_DBC_2011.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ocean_salinity_ExSec3600_Grid0.02_2011.txt
    ${CMAKE_CURRENT_BINARY_DIR}/ocean_salinity_ExSec3600_Grid0.02_2011.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/peat_frac.txt
    ${CMAKE_CURRENT_BINARY_DIR}/peat_frac.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/porosity.txt
    ${CMAKE_CURRENT_BINARY_DIR}/porosity.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/quasi2D.g
    ${CMAKE_CURRENT_BINARY_DIR}/quasi2D.g COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/quasi2D.g.4.0
    ${CMAKE_CURRENT_BINARY_DIR}/quasi2D.g.4.0 COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/quasi2D.g.4.1
    ${CMAKE_CURRENT_BINARY_DIR}/quasi2D.g.4.1 COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/quasi2D.g.4.2
    ${CMAKE_CURRENT_BINARY_DIR}/quasi2D.g.4.2 COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/quasi2D.g.4.3
    ${CMAKE_CURRENT_BINARY_DIR}/quasi2D.g.4.3 COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/sand_frac.txt
    ${CMAKE_CURRENT_BINARY_DIR}/sand_frac.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/silt_frac.txt
    ${CMAKE_CURRENT_BINARY_DIR}/silt_frac.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation_unstab.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/simulation_unstab.yaml COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation_unstab_explicit.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/simulation_unstab_explicit.yaml COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation_stab.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/simulation_stab.yaml COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/TimeHis_ExSec3600_2011.txt
    ${CMAKE_CURRENT_BINARY_DIR}/TimeHis_ExSec3600_2011.txt COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/waterH_ExSec3600_Grid0.02_2011.txt
    ${CMAKE_CURRENT_BINARY_DIR}/waterH_ExSec3600_Grid0.02_2011.txt COPYONLY)

  # Create the test with this name and standard executable
  add_test(ACE_${testName}_unstab ${Albany.exe} simulation_unstab.yaml)
  set_tests_properties(ACE_${testName}_unstab PROPERTIES LABELS
                                                      "Demo;Tpetra;Forward")
  
  add_test(ACE_${testName}_unstab_Explicit ${Albany.exe} simulation_unstab_explicit.yaml)
  set_tests_properties(ACE_${testName}_unstab_Explicit PROPERTIES LABELS
                                                      "Demo;Tpetra;Forward")

  add_test(ACE_${testName}_stab ${Albany.exe} simulation_stab.yaml)
  set_tests_properties(ACE_${testName}_stab PROPERTIES LABELS
                                                      "Demo;Tpetra;Forward")

endif()
