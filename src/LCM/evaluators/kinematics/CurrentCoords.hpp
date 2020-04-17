// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef CURRENTCOORDS_HPP
#define CURRENTCOORDS_HPP

#include "Albany_Layouts.hpp"
#include "Albany_Types.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
/** \brief

    Compute the current coordinates

**/

template <typename EvalT, typename Traits>
class CurrentCoords : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  CurrentCoords(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

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
  PHX::MDField<const MeshScalarT, Cell, Vertex, Dim> refCoords;
  PHX::MDField<ScalarT const, Cell, Vertex, Dim>     displacement;

  // Output:
  PHX::MDField<ScalarT, Cell, Vertex, Dim> currentCoords;

  unsigned int worksetSize;
  unsigned int numNodes;
  unsigned int numDims;
};
}  // namespace LCM

#endif
