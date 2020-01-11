# LCM ###############

IF(ALBANY_LCM)
  add_subdirectory(LCM)
ENDIF(ALBANY_LCM)

IF(ALBANY_PERFORMANCE_TESTS)
  add_subdirectory(PerformanceTests)
ENDIF()
