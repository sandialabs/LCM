// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <string>

#include "Albany_Macros.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM {

//*****
template <typename EvalT, typename Traits>
ACEWavePressureBC_Base<EvalT, Traits>::ACEWavePressureBC_Base(Teuchos::ParameterList& p)
    : PHAL::Neumann<EvalT, Traits>(p)
{
  timeValues        = p.get<Teuchos::Array<RealType>>("Time Values").toVector();
  WaterHeightValues = p.get<Teuchos::Array<RealType>>("Water Height Values").toVector();

  ALBANY_PANIC(
      !(timeValues.size() == WaterHeightValues.size()), "Dimension of \"Time Values\" and \"BC Values\" do not match");
}

//*****
template <typename EvalT, typename Traits>
void
ACEWavePressureBC_Base<EvalT, Traits>::computeVal(RealType time)
{
  ALBANY_PANIC(time > timeValues.back(), "Time is growing unbounded!");
  ScalarT      Val;
  RealType     slope;
  unsigned int Index(0);

  while (timeValues[Index] < time) Index++;

  if (Index == 0)
    this->water_height_val = WaterHeightValues[Index];
  else {
    slope = (WaterHeightValues[Index] - WaterHeightValues[Index - 1]) / (timeValues[Index] - timeValues[Index - 1]);
    this->water_height_val = WaterHeightValues[Index - 1] + slope * (time - timeValues[Index - 1]);
    // std::cout << "IKT computeVal water_height_val = " << this->water_height_val << "\n";
  }

  return;
}

template <typename EvalT, typename Traits>
ACEWavePressureBC<EvalT, Traits>::ACEWavePressureBC(Teuchos::ParameterList& p)
    : ACEWavePressureBC_Base<EvalT, Traits>(p)
{
}

template <typename EvalT, typename Traits>
void
ACEWavePressureBC<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  RealType time = workset.current_time;

  switch (this->bc_type) {
    case PHAL::NeumannBase<EvalT, Traits>::ACEPRESS:
      // calculate scalar value of BC based on current time
      this->computeVal(time);
      break;

    default:

      ALBANY_ABORT(
          "ACE Time dependent Neumann boundary condition of type - "
          << this->bc_type << " is not supported.  Only 'ACE P' NBC is supported.");
      break;
  }

  PHAL::Neumann<EvalT, Traits>::evaluateFields(workset);
}

}  // namespace LCM
