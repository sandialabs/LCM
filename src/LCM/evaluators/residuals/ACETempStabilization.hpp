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
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim> thermal_cond_grad_at_qps_;  // thermal conductivity
                                                                                // grad at qps
  PHX::MDField<const MeshScalarT, Cell, QuadPoint> jacobian_det_;               // jacobian determinant - for getting mesh size h
  // Output:
  PHX::MDField<ScalarT, Cell, Node> tau_;

  unsigned int num_qps_{0}, num_dims_{0}, num_nodes_{0}, workset_size_{0};

  Teuchos::RCP<Teuchos::FancyOStream> fos_;

  // Stabilization parameters
  double stab_value_{0.0};

  enum TAU_TYPE
  {
    NONE,
    SUPG,
    PROP_TO_H
  };
  TAU_TYPE tau_type_;
};
}  // namespace LCM

#endif
