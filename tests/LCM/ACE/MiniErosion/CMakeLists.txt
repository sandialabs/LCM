#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

add_subdirectory(bcs)
add_subdirectory(grid)
add_subdirectory(external_files)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_thermal_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_thermal_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_serial.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_np4.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_np4.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/coupled_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_mechanical_denudation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_mechanical_denudation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/denudation.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/denudation.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/denudation_gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/denudation_gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/denudation_np4_gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/denudation_np4_gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_restart.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_restart.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_restart_base.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/coupled_restart_base.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_restart_base.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_restart_base.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_restart_base.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_restart_base.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_restart.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/coupled_restart.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_restart.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_restart.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_restart.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_restart.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled_restart_pd.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/coupled_restart_pd.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thermal_restart_pd.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/thermal_restart_pd.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanical_restart_pd.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/mechanical_restart_pd.yaml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Every test here runs in this one build directory and writes the same Exodus
# file names, so they are serialized against each other with a resource lock
# rather than left to collide whenever ctest happens to schedule two at once.

# Serial test - denudation with element death (short run: 20 steps).
# M4: the ACE thermo-mechanical solver writes a single coupled Exodus
# output file (denudation.e) containing both physics' fields on the
# shared mesh, so the test does one exodiff against denudation_gold.e.
set(OUTFILE "denudation.e")
set(REF_FILE "denudation_gold.e")
set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest_serial.cmake)
add_test(
  NAME ACE_${testName}_Denudation_Serial
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME=denudation
    -DTEST_ARGS=coupled_denudation.yaml -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE} -P
    ${runtest.cmake})
set_tests_properties(ACE_${testName}_Denudation_Serial
                     PROPERTIES LABELS "LCM;Tpetra;Forward"
                                RESOURCE_LOCK ACE_${testName}_workdir)

# 4-rank parallel test - exodiff after merging with epu, against an
# np4-specific gold. Guards against regressions in the
# clone-before-disconnect element-death algorithm
# (Albany::applyElementDeath) which is what makes parallel erosion work;
# see doc/element-death.md. The np4 gold is separate from the serial one
# because a death cascade amplifies the epsilon-level differences between
# serial and parallel reduction order into discrete failure-threshold
# crossings: serial and np4 agree exactly through the first death waves,
# then individual cell death times shift by a step. A fixed 4-rank run is
# reproducible run-to-run, so it regresses against its own gold.
set(REF_FILE "denudation_np4_gold.e")
set(runtest_np4.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest_np4.cmake)
add_test(
  NAME ACE_${testName}_Denudation_np4
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}" -DTEST_NAME=denudation
    -DTEST_ARGS=coupled_denudation.yaml -DMPIMNP=${MPIMNP}
    -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -DSEACAS_EPU=${SEACAS_EPU}
    -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE} -P
    ${runtest_np4.cmake})
set_tests_properties(ACE_${testName}_Denudation_np4
                     PROPERTIES LABELS "LCM;Tpetra;Forward"
                                PROCESSORS ${MPIMNP}
                                RESOURCE_LOCK ACE_${testName}_workdir)

# Restart test - run the reference case, restart it from its tenth frame and
# require the two final states to agree. Self-comparing, so there is no gold
# file to regenerate when erosion timing changes. See runtest_restart.cmake.
set(runtest_restart.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest_restart.cmake)
add_test(
  NAME ACE_${testName}_Denudation_Restart
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME=denudation
    -DBASE_ARGS=coupled_restart_base.yaml -DRESTART_ARGS=coupled_restart.yaml
    -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DBASE_FILENAME=restart_base.e -DRESTART_FILENAME=restart.e -P
    ${runtest_restart.cmake})
set_tests_properties(ACE_${testName}_Denudation_Restart
                     PROPERTIES LABELS "LCM;Tpetra;Forward"
                                RESOURCE_LOCK ACE_${testName}_workdir)

# The same restart consistency check on 4 ranks. Worth its own test: on several
# ranks the restart has to land on the overlap maps and the combine/scatter
# manager as well, which the serial run never exercises.
add_test(
  NAME ACE_${testName}_Denudation_Restart_np4
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}" -DTEST_NAME=denudation
    -DBASE_ARGS=coupled_restart_base.yaml -DRESTART_ARGS=coupled_restart.yaml
    -DMPIMNP=${MPIMNP} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -DSEACAS_EPU=${SEACAS_EPU}
    -DBASE_FILENAME=restart_base.e -DRESTART_FILENAME=restart.e -P
    ${runtest_restart.cmake})
set_tests_properties(ACE_${testName}_Denudation_Restart_np4
                     PROPERTIES LABELS "LCM;Tpetra;Forward"
                                PROCESSORS ${MPIMNP}
                                RESOURCE_LOCK ACE_${testName}_workdir)

# Restart test from a POST-death frame: same self-comparing structure, restart
# point after the first three death events. Exercises the restart
# death-history replay (frame-walked death surgery, erodible-set painting,
# post-surgery rebuild); see issue #119. Restarting before any death (the
# tests above) does not cover any of that machinery.
add_test(
  NAME ACE_${testName}_Denudation_Restart_PostDeath
  COMMAND
  ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_NAME=denudation
    -DBASE_ARGS=coupled_restart_base.yaml -DRESTART_ARGS=coupled_restart_pd.yaml
    -DMPIMNP=1 -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
    -DBASE_FILENAME=restart_base.e -DRESTART_FILENAME=restart_pd.e -P
    ${runtest_restart.cmake})
set_tests_properties(ACE_${testName}_Denudation_Restart_PostDeath
                     PROPERTIES LABELS "LCM;Tpetra;Forward"
                                RESOURCE_LOCK ACE_${testName}_workdir)
