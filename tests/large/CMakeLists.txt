# LCM ###############

if(ALBANY_LCM)
  add_subdirectory(LCM)
endif(ALBANY_LCM)

if(ALBANY_PERFORMANCE_TESTS)
  add_subdirectory(PerformanceTests)
endif()
