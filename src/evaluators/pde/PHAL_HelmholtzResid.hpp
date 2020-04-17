// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_HELMHOLTZRESID_HPP
#define PHAL_HELMHOLTZRESID_HPP

#include "Albany_SacadoTypes.hpp"
#include "PHAL_Dimension.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Sacado_ParameterRegistration.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template <typename EvalT, typename Traits>
class HelmholtzResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                       public PHX::EvaluatorDerived<EvalT, Traits>,
                       public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  HelmholtzResid(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

  virtual ScalarT&
  getValue(std::string const& n)
  {
    return ksqr;
  };

 private:
  // Input:
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint>      wBF;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                U;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                V;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wGradBF;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>           UGrad;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>           VGrad;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                USource;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                VSource;

  bool haveSource;

  ScalarT ksqr;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> UResidual;
  PHX::MDField<ScalarT, Cell, Node> VResidual;
};
}  // namespace PHAL

#endif
