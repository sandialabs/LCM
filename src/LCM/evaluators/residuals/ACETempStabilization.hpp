// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ACETEMPSTABILIZATION_HPP
#define ACETEMPSTABILIZATION_HPP

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
class ACETempStabilization : public PHX::EvaluatorWithBaseImpl<Traits>, public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ACETempStabilization(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT const, Cell, QuadPoint>                tdot_;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wgradbf_;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>           tgrad_;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                thermal_conductivity_;  // thermal conductivity
  PHX::MDField<const ScalarT, Cell, QuadPoint>                thermal_inertia_;       // thermal inertia = rho * C

  // Output:
  PHX::MDField<ScalarT, Cell, Node> stab_;

  unsigned int num_qps_{0}, num_dims_{0}, num_nodes_{0}, workset_size_{0};
};
}  // namespace LCM

#endif
