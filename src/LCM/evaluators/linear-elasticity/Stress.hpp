// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef STRESS_HPP
#define STRESS_HPP

#include "Albany_Types.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template <typename EvalT, typename Traits>
class Stress : public PHX::EvaluatorWithBaseImpl<Traits>,
               public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  Stress(Teuchos::ParameterList const& p);

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
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim, Dim> strain;
  PHX::MDField<ScalarT const, Cell, QuadPoint>           elasticModulus;
  PHX::MDField<ScalarT const, Cell, QuadPoint>           poissonsRatio;

  unsigned int numQPs;
  unsigned int numDims;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;
};
}  // namespace LCM

#endif
