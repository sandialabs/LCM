// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef TLPOROSTRESS_HPP
#define TLPOROSTRESS_HPP

#include "Albany_Types.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
/** \brief

    This evaluator obtains effective stress and return
    total stress (i.e. with pore-fluid contribution)
    For now, it does not work for Neohookean AD

*/

template <typename EvalT, typename Traits>
class TLPoroStress : public PHX::EvaluatorWithBaseImpl<Traits>,
                     public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  TLPoroStress(Teuchos::ParameterList const& p);

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
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim, Dim> defGrad;
  PHX::MDField<ScalarT const, Cell, QuadPoint>           J;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim, Dim> stress;
  PHX::MDField<ScalarT const, Cell, QuadPoint>           biotCoefficient;
  PHX::MDField<ScalarT const, Cell, QuadPoint>           porePressure;

  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  // Work space FCs
  Kokkos::DynRankView<ScalarT, PHX::Device> F_inv;
  Kokkos::DynRankView<ScalarT, PHX::Device> F_invT;
  Kokkos::DynRankView<ScalarT, PHX::Device> JF_invT;
  Kokkos::DynRankView<ScalarT, PHX::Device> JpF_invT;
  Kokkos::DynRankView<ScalarT, PHX::Device> JBpF_invT;

  // Material Name
  std::string matModel;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> totstress;
};
}  // namespace LCM

#endif
