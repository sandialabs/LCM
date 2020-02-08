# Name the test with the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# ##############################################################################
# 1D Standing Wave test     ###
# ##############################################################################


# ##############################################################################
# 2D Steady test         ###
# ##############################################################################


# ##############################################################################
# 2D Press Pulse test     ###
# ##############################################################################

if(ALBANY_IFPACK2)
  set(testName ${testNameRoot}_2DUnsteadyInvPressPulse)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2Dunsteady.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input2Dunsteady.yaml COPYONLY)

  add_test(${testName} ${Albany.exe} input2Dunsteady.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Demo;Tpetra;Forward")
endif()

# ##############################################################################
# 2D Euler MMS test       ###
# ##############################################################################


# ##############################################################################
# 2D Driven Pulse test     ###
# ##############################################################################


# ##############################################################################
# 3D Unsteady Euler test    ###
# ##############################################################################


# ##############################################################################
# 3D Unsteady NS test     ###
# ##############################################################################

# IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA) set(testName
# ${testNameRoot}_3DUnsteadyNS)

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input3DunsteadyNS.yaml
# ${CMAKE_CURRENT_BINARY_DIR}/input3DunsteadyNS.yaml COPYONLY)

# add_test(${testName} ${SerialAlbany.exe} input3DunsteadyNS.yaml)
# set_tests_properties(${testName} PROPERTIES LABELS
# "Demo;Epetra;Forward;Serial") ENDIF()
