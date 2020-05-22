if(NOT ALBANY_PARALLEL_ONLY)
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/simulation.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/simulation.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
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
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/time_fake.txt
                 ${CMAKE_CURRENT_BINARY_DIR}/time_fake.txt COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/waterH_fake.txt
         	  ${CMAKE_CURRENT_BINARY_DIR}/waterH_fake.txt COPYONLY)

  # Create the test with this name and standard executable
  #add_test(${testName} ${Albany.exe} simulation.yaml)
  #set_tests_properties(${testName} PROPERTIES LABELS "Demo;Tpetra;Forward")
  
endif()
