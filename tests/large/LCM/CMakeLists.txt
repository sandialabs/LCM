#
# Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). This Software is released under the BSD license detailed
# in the file license.txt in the top-level Albany directory
#

if(ALBANY_DTK)
  add_subdirectory(DTKInterp)
endif()

if(ALBANY_STK)
  if(ALBANY_SEACAS)
    add_subdirectory(CrystalPlasticity)
    # amota: disabled because enters infinite loop add_subdirectory(HMC)
    add_subdirectory(Schwarz)
    add_subdirectory(MechanicsWithHydrogenBar)
  endif()
endif()
