// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <string>

#include "Albany_Macros.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM {

//*****
template <typename EvalT, typename Traits>
ACETimeTracBC_Base<EvalT, Traits>::ACETimeTracBC_Base(Teuchos::ParameterList& p) : PHAL::Neumann<EvalT, Traits>(p)
{
  timeValues = p.get<Teuchos::Array<RealType>>("Time Values").toVector();
  WaterHeightValues   = p.get<Teuchos::Array<RealType>>("Water Height Values").toVector();

  //IKT is the following needed?
  /*if (this->bc_type == PHAL::NeumannBase<EvalT, Traits>::COORD)

    ALBANY_PANIC(
        !(this->cellDims == WaterHeightValues.getNumCols()), "Dimension of the current problem and \"BC Values\" do not match");
  */
  ALBANY_PANIC(
      !(timeValues.size() == WaterHeightValues.size()), "Dimension of \"Time Values\" and \"BC Values\" do not match");
}

//*****
template <typename EvalT, typename Traits>
void
ACETimeTracBC_Base<EvalT, Traits>::computeVal(RealType time)
{
  ALBANY_PANIC(time > timeValues.back(), "Time is growing unbounded!");
  ScalarT      Val;
  RealType     slope;
  unsigned int Index(0);

  while (timeValues[Index] < time) Index++;

  if (Index == 0)
    this->const_val = WaterHeightValues[Index];
  else {
    slope           = (WaterHeightValues[Index] - WaterHeightValues[Index - 1]) / (timeValues[Index] - timeValues[Index - 1]);
    this->const_val = WaterHeightValues[Index - 1] + slope * (time - timeValues[Index - 1]);
    std::cout << "IKT computeVal const_val = " << this->const_val << "\n"; 
  }

  return;
}

template <typename EvalT, typename Traits>
void
ACETimeTracBC_Base<EvalT, Traits>::computeCoordVal(RealType time)
{
  //IKT FIXME - implement??
  /*ALBANY_PANIC(time > timeValues.back(), "Time is growing unbounded!");
  ScalarT      Val;
  RealType     slope;
  unsigned int Index(0);

  while (timeValues[Index] < time) Index++;

  if (Index == 0)
    for (int dim = 0; dim < this->cellDims; dim++) this->dudx[dim] = WaterHeightValues(dim, Index);
  else {
    for (size_t dim = 0; dim < this->cellDims; dim++) {
      slope           = (WaterHeightValues(dim, Index) - WaterHeightValues(dim, Index - 1)) / (timeValues[Index] - timeValues[Index - 1]);
      this->dudx[dim] = WaterHeightValues(dim, Index - 1) + slope * (time - timeValues[Index - 1]);
    }
  }*/

  return;
}

template <typename EvalT, typename Traits>
ACETimeTracBC<EvalT, Traits>::ACETimeTracBC(Teuchos::ParameterList& p) : ACETimeTracBC_Base<EvalT, Traits>(p)
{
}

template <typename EvalT, typename Traits>
void
ACETimeTracBC<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  std::cout << "IKT in ACETimeTracBC evaluateFields\n"; 

  RealType time = workset.current_time;

  switch (this->bc_type) {
    case PHAL::NeumannBase<EvalT, Traits>::ACEPRESS:
      // calculate scalar value of BC based on current time
      this->computeVal(time);
      break;

    default:

      ALBANY_ABORT("ACE Time dependent Neumann boundary condition of type - " << this->bc_type << " is not supported.  Only 'ACE P' NBC is supported.");
      break;
  }

  PHAL::Neumann<EvalT, Traits>::evaluateFields(workset);
}

}  // namespace LCM
