if(ALBANY_EPETRA)
  # 1. Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_2D.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_2D.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_3D.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_3D.yaml COPYONLY)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  add_test(${testName}_2D ${Albany.exe} input_2D.yaml)
  set_tests_properties(${testName}_2D PROPERTIES LABELS "Basic;Epetra;Forward")

  add_test(${testName}_3D ${Albany.exe} input_3D.yaml)
  set_tests_properties(${testName}_3D PROPERTIES LABELS "Basic;Epetra;Forward")
endif()
