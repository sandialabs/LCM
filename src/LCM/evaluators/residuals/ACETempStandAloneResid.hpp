// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ACETEMPSTANDALONERESID_HPP
#define ACETEMPSTANDALONERESID_HPP

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
class ACETempStandAloneResid : public PHX::EvaluatorWithBaseImpl<Traits>, public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ACETempStandAloneResid(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint>      wbf_;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                tdot_;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wgradbf_;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>           tgrad_;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                thermal_conductivity_;      // thermal conductivity
  PHX::MDField<const ScalarT, Cell, QuadPoint>                thermal_inertia_;           // thermal inertia = rho * C
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim>           thermal_cond_grad_at_qps_;  // thermal conductivity
                                                                                          // grad at qps
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> coord_vec_;
  PHX::MDField<ScalarT, Cell, Node>                     tau_;
  PHX::MDField<const MeshScalarT, Cell, QuadPoint>      jacobian_det_;  // jacobian determinant - for getting mesh size h

  // Output:
  PHX::MDField<ScalarT, Cell, Node> residual_;

  unsigned int                        num_qps_{0}, num_dims_{0}, num_nodes_{0}, workset_size_{0};
  Teuchos::RCP<Teuchos::FancyOStream> fos_;

  // Parameters relevant to stabilization
  bool        use_stab_{false};
  double      x_max_{0.0}, z_max_{0.0};
  double      max_time_stab_{1.0e10};
  std::string stab_type_;
};
}  // namespace LCM

#endif
