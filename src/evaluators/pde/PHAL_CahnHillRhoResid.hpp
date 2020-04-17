// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_CAHNHILLRHORESID_HPP
#define PHAL_CAHNHILLRHORESID_HPP

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
class CahnHillRhoResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;

  CahnHillRhoResid(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

  ScalarT&
  getValue(std::string const& n);

 private:
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint>      wBF;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wGradBF;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>           rhoGrad;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                chemTerm;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                noiseTerm;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> rhoResidual;

  Kokkos::DynRankView<ScalarT, PHX::Device> gamma_term;

  unsigned int numQPs, numDims, numNodes, worksetSize;

  ScalarT gamma;

  // Langevin noise present
  bool haveNoise;
};
}  // namespace PHAL

#endif
