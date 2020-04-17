// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef MIXTURE_THERMAL_EXPANSION_HPP
#define MIXTURE_THERMAL_EXPANSION_HPP

#include "Albany_Types.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
/** \brief

    This evaluator calculates thermal expansion of a bi-phase
    mixture through volume averaging

*/

template <typename EvalT, typename Traits>
class MixtureThermalExpansion : public PHX::EvaluatorWithBaseImpl<Traits>,
                                public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  MixtureThermalExpansion(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  // Input:
  PHX::MDField<ScalarT const, Cell, QuadPoint> biotCoefficient;
  PHX::MDField<ScalarT const, Cell, QuadPoint> porosity;
  PHX::MDField<ScalarT const, Cell, QuadPoint> J;
  PHX::MDField<ScalarT const, Cell, QuadPoint> alphaSkeleton;
  PHX::MDField<ScalarT const, Cell, QuadPoint> alphaPoreFluid;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint> mixtureThermalExpansion;

  unsigned int numQPs;
  //  unsigned int numDims;
};
}  // namespace LCM

#endif
