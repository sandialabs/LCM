// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <string>

#include "Albany_Macros.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM {

//*****
template <typename EvalT, typename Traits>
ACEWavePressureBC_Base<EvalT, Traits>::ACEWavePressureBC_Base(Teuchos::ParameterList& p) : PHAL::Neumann<EvalT, Traits>(p)
{
  timeValues       = p.get<Teuchos::Array<RealType>>("Time Values").toVector();
  waveLengthValues = p.get<Teuchos::Array<RealType>>("Wave Length Values").toVector();
  waveNumberValues = p.get<Teuchos::Array<RealType>>("Wave Number Values").toVector();
  sValues          = p.get<Teuchos::Array<RealType>>("Still Water Level Values").toVector();
  wValues          = p.get<Teuchos::Array<RealType>>("Wave Height Values").toVector();
  waterHValues     = p.get<Teuchos::Array<RealType>>("WaterH Values").toVector();

  auto bc_type = this->bc_type;
  // IKT, 8/19/2021: the following checks are overkill, as we do the same checks
  // in Albany::BCUtils
  if (bc_type == PHAL::NeumannBase<EvalT, Traits>::ACEPRESS) {
    ALBANY_PANIC(!(timeValues.size() == waveLengthValues.size()), "Dimension of \"Time Values\" and \"Wave Length Values\" do not match\n");
    ALBANY_PANIC(!(timeValues.size() == waveNumberValues.size()), "Dimension of \"Time Values\" and \"Wave Number Values\" do not match\n");
    ALBANY_PANIC(!(timeValues.size() == sValues.size()), "Dimension of \"Time Values\" and \"Still Water Level Values\" do not match\n");
    ALBANY_PANIC(!(timeValues.size() == wValues.size()), "Dimension of \"Time Values\" and \"Wave Height Values\" do not match\n");
  }
  else if (bc_type == PHAL::NeumannBase<EvalT, Traits>::ACEPRESS_HYDROSTATIC) {
    ALBANY_PANIC(!(timeValues.size() == waterHValues.size()), "Dimension of \"Time Values\" and \"WaterH Values\" do not match\n");
  }
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
    this->wave_length_val = waveLengthValues[Index];
    this->wave_number_val = waveNumberValues[Index];
    this->s_val           = sValues[Index];
    this->w_val           = wValues[Index];
  } else {
    slope                 = (waveLengthValues[Index] - waveLengthValues[Index - 1]) / (timeValues[Index] - timeValues[Index - 1]);
    this->wave_length_val = waveLengthValues[Index - 1] + slope * (time - timeValues[Index - 1]);
    slope                 = (waveNumberValues[Index] - waveNumberValues[Index - 1]) / (timeValues[Index] - timeValues[Index - 1]);
    this->wave_number_val = waveNumberValues[Index - 1] + slope * (time - timeValues[Index - 1]);
    slope                 = (sValues[Index] - sValues[Index - 1]) / (timeValues[Index] - timeValues[Index - 1]);
    this->s_val           = sValues[Index - 1] + slope * (time - timeValues[Index - 1]);
    slope                 = (wValues[Index] - wValues[Index - 1]) / (timeValues[Index] - timeValues[Index - 1]);
    this->w_val           = wValues[Index - 1] + slope * (time - timeValues[Index - 1]);
    // std::cout << "IKT computeVal water_height_val = " << this->water_height_val << "\n";

    ALBANY_PANIC(this->wave_length_val <= 0, "Wave length is non-positive!");
    ALBANY_PANIC(this->wave_number_val <= 0, "Wave number is non-positive!");
  }

  return;
}

template <typename EvalT, typename Traits>
void
ACEWavePressureBC_Base<EvalT, Traits>::computeValHydrostatic(RealType time)
{
  ALBANY_PANIC(time > timeValues.back(), "Time is growing unbounded!");
  ScalarT      Val;
  RealType     slope;
  unsigned int Index(0);

  while (timeValues[Index] < time) Index++;

  if (Index == 0) {
    this->waterH_val      = waterHValues[Index];
  } else {
    slope                 = (waterHValues[Index] - waterHValues[Index - 1]) / (timeValues[Index] - timeValues[Index - 1]);
    this->waterH_val       = waterHValues[Index - 1] + slope * (time - timeValues[Index - 1]);
    // std::cout << "IKT computeVal waterH_val = " << this->waterH_val << "\n";
    if (this->waterH_val < 0.){
      this->waterH_val = 0.;
    }
  }

  return;
}

template <typename EvalT, typename Traits>
ACEWavePressureBC<EvalT, Traits>::ACEWavePressureBC(Teuchos::ParameterList& p) : ACEWavePressureBC_Base<EvalT, Traits>(p)
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

    case PHAL::NeumannBase<EvalT, Traits>::ACEPRESS_HYDROSTATIC:
      // calculate scalar value of BC based on current time
      this->computeValHydrostatic(time);
      break;

    default:

      ALBANY_ABORT("ACE Time dependent Neumann boundary condition of type - " << this->bc_type << " is not supported.  Only 'ACE P' NBC is supported.");
      break;
  }

  PHAL::Neumann<EvalT, Traits>::evaluateFields(workset);
}

}  // namespace LCM
