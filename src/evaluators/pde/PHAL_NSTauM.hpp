// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_NSTAUM_HPP
#define PHAL_NSTAUM_HPP

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
class NSTauM : public PHX::EvaluatorWithBaseImpl<Traits>,
               public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  NSTauM(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>          V;
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim, Dim> Gc;
  PHX::MDField<ScalarT const, Cell, QuadPoint>               rho;
  PHX::MDField<ScalarT const, Cell, QuadPoint>               mu;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> TauM;

  unsigned int                                  numQPs, numDims, numCells;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> normGc;
};
}  // namespace PHAL

#endif
