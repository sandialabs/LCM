// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license
// detailed in the file license.txt in the top-level Albany directory.

#ifndef PHAL_MAP_TO_PHYSICAL_FRAME_SIDE_HPP
#define PHAL_MAP_TO_PHYSICAL_FRAME_SIDE_HPP

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Utilities.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace PHAL {
/** \brief Compute Side Quad points coordinates

    This evaluator computes the coordinates of the quad points
    on a side set, by interpolating the side vertices coordinates.

*/

template <typename EvalT, typename Traits>
class MapToPhysicalFrameSide : public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  MapToPhysicalFrameSide(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Accessory variables
  std::vector<Kokkos::DynRankView<RealType, PHX::Device>> phi_at_cub_points;
  std::vector<std::vector<int>>                           sideVertices;
  std::vector<int>                                        numSideVertices;

  int         numDim;
  int         numSideQPs;
  std::string sideSetName;

  // Input:
  //! Values at vertices
  PHX::MDField<const MeshScalarT, Cell, Side, Vertex, Dim> coords_side_vertices;

  // Output:
  //! Values at quadrature points
  PHX::MDField<MeshScalarT, Cell, Side, QuadPoint, Dim> coords_side_qp;
};

}  // Namespace PHAL

#endif  // PHAL_MAP_TO_PHYSICAL_FRAME_SIDE_HPP
