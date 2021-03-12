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
  waterHeightValues = p.get<Teuchos::Array<RealType>>("Water Height Values").toVector();
  waveBreakingHeightValues = p.get<Teuchos::Array<RealType>>("Wave Breaking Height Values").toVector();
  waveLengthValues = p.get<Teuchos::Array<RealType>>("Wave Length Values").toVector();
  waveNumberValues = p.get<Teuchos::Array<RealType>>("Wave Number Values").toVector();

  ALBANY_PANIC(
      !(timeValues.size() == waterHeightValues.size()), "Dimension of \"Time Values\" and \"water Height Values\" do not match");
  ALBANY_PANIC(
      !(timeValues.size() == waveBreakingHeightValues.size()), "Dimension of \"Time Values\" and \"Wave Breaking Height Values\" do not match");
  ALBANY_PANIC(
      !(timeValues.size() == waveLengthValues.size()), "Dimension of \"Time Values\" and \"Wave Length Values\" do not match");
  ALBANY_PANIC(
      !(timeValues.size() == waveNumberValues.size()), "Dimension of \"Time Values\" and \"Wave Number Values\" do not match");
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

  if (Index == 0) {
    this->water_height_val = waterHeightValues[Index];
    this->wave_breaking_height_val = waveBreakingHeightValues[Index]; 
    this->wave_length_val = waveLengthValues[Index]; 
    this->wave_number_val = waveNumberValues[Index]; 
  }
  else {
    slope = (waterHeightValues[Index] - waterHeightValues[Index - 1]) / (timeValues[Index] - timeValues[Index - 1]);
    this->water_height_val = waterHeightValues[Index - 1] + slope * (time - timeValues[Index - 1]);
    slope = (waveBreakingHeightValues[Index] - waveBreakingHeightValues[Index - 1]) / (timeValues[Index] - timeValues[Index - 1]);
    this->wave_breaking_height_val = waveBreakingHeightValues[Index - 1] + slope * (time - timeValues[Index - 1]);
    slope = (waveLengthValues[Index] - waveLengthValues[Index - 1]) / (timeValues[Index] - timeValues[Index - 1]);
    this->wave_length_val = waveLengthValues[Index - 1] + slope * (time - timeValues[Index - 1]);
    slope = (waveNumberValues[Index] - waveNumberValues[Index - 1]) / (timeValues[Index] - timeValues[Index - 1]);
    this->wave_number_val = waveNumberValues[Index - 1] + slope * (time - timeValues[Index - 1]);
    // std::cout << "IKT computeVal water_height_val = " << this->water_height_val << "\n";
  }
  
  //IKT question: can water height be non-positive?  If not, add throw.
  ALBANY_PANIC(this->wave_breaking_height_val <= 0, "Wave breaking height is non-positive!");
  ALBANY_PANIC(this->wave_length_val <= 0, "Wave length is non-positive!");
  ALBANY_PANIC(this->wave_number_val <= 0, "Wave number is non-positive!");

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
