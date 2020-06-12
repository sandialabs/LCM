#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory.
#

set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
add_subdirectory(Alternating)
#
# These are big examples. Any problems in the Schwarz implementation will be
# caught by the cubes problem above, so we disable them.
#
# add_subdirectory(Overlap) add_subdirectory(NotchedCylinder)
# add_subdirectory(LaserWeld)
