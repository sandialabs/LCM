// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_NSRM_HPP
#define PHAL_NSRM_HPP

#include "PHAL_Dimension.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template <typename EvalT, typename Traits>
class NSRm : public PHX::EvaluatorWithBaseImpl<Traits>,
             public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;

  NSRm(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>      pGrad;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim, Dim> VGrad;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>      V;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>      V_Dot;
  PHX::MDField<ScalarT const, Cell, QuadPoint>           T;
  PHX::MDField<ScalarT const, Cell, QuadPoint>           rho;
  PHX::MDField<ScalarT const, Cell, QuadPoint>           phi;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>      force;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>      permTerm;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>      ForchTerm;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim> Rm;

  unsigned int numQPs, numDims, numNodes;
  bool         enableTransient;
  bool         haveHeat;
  bool         porousMedia;
};
}  // namespace PHAL

#endif
