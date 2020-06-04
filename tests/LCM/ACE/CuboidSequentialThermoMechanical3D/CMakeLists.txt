if(NOT ALBANY_PARALLEL_ONLY)
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/coupled.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/thermal.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/mechanical.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/bluff_salinity.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/bluff_salinity.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clay_frac.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/clay_frac.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/depth.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/depth.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/grid.g
                 ${CMAKE_CURRENT_BINARY_DIR}/grid.g COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/grid.g.4.0
                 ${CMAKE_CURRENT_BINARY_DIR}/grid.g.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/grid.g.4.1
                 ${CMAKE_CURRENT_BINARY_DIR}/grid.g.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/grid.g.4.2
                 ${CMAKE_CURRENT_BINARY_DIR}/grid.g.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/grid.g.4.3
                 ${CMAKE_CURRENT_BINARY_DIR}/grid.g.4.3 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_thermal.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/materials_thermal.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_mechanical.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/materials_mechanical.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/peat_frac.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/peat_frac.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/porosity.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/porosity.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/sand_frac.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/sand_frac.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/silt_frac.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/silt_frac.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/time_fake.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/time_fake.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/waterH_fake.txt
         	  ${CMAKE_CURRENT_BINARY_DIR}/waterH_fake.txt COPYONLY)

  # Create the test with this name and standard executable
  #add_test(${testName} ${Albany.exe} coupled.yaml)
  #set_tests_properties(${testName} PROPERTIES LABELS "Demo;Tpetra;Forward")
  
endif()
