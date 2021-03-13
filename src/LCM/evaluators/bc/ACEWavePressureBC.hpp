// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ACETIMETRACBC_HPP
#define ACETIMETRACBC_HPP

#include "PHAL_Neumann.hpp"

namespace LCM {

/** \brief ACE time dependent Neumann boundary condition evaluator

*/

template <typename EvalT, typename Traits>
class ACEWavePressureBC_Base : public PHAL::Neumann<EvalT, Traits>
{
 public:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  ACEWavePressureBC_Base(Teuchos::ParameterList& p);

  void
  computeVal(RealType time);

 protected:
  std::vector<RealType> timeValues;
  std::vector<RealType> waterHeightValues;
  std::vector<RealType> heightAboveWaterOfMaxPressure;
  std::vector<RealType> waveLengthValues;
  std::vector<RealType> waveNumberValues;
};

template <typename EvalT, typename Traits>
class ACEWavePressureBC : public ACEWavePressureBC_Base<EvalT, Traits>
{
 public:
  ACEWavePressureBC(Teuchos::ParameterList& p);
  void
  evaluateFields(typename Traits::EvalData d);

 private:
  using ScalarT = typename EvalT::ScalarT;
};

}  // namespace LCM

#endif
