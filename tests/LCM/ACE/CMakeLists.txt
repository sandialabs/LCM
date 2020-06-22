#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
add_subdirectory(Cube)
add_subdirectory(Cuboid)
add_subdirectory(StandAloneThermal2D)
add_subdirectory(CuboidStandAloneThermal3D)
add_subdirectory(CuboidThermoMechanical3D)
add_subdirectory(CuboidSequentialThermoMechanical3D)
add_subdirectory(StandAloneThermal2.5DSlice)
