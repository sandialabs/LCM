// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_NEUMANN_HPP
#define PHAL_NEUMANN_HPP

#include "Albany_Layouts.hpp"
#include "Albany_MaterialDatabase.hpp"
#include "Albany_MeshSpecs.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Utilities.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Sacado_ParameterAccessor.hpp"

namespace PHAL {

/** \brief Neumann boundary condition evaluator

*/

template <typename EvalT, typename Traits>
class NeumannBase : public PHX::EvaluatorWithBaseImpl<Traits>, public PHX::EvaluatorDerived<EvalT, Traits>, public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  enum NEU_TYPE
  {
    COORD,
    NORMAL,
    INTJUMP,
    PRESS,
    ACEPRESS,
    ACEPRESS_HYDROSTATIC,
    ROBIN,
    TRACTION,
    CLOSED_FORM,
    STEFAN_BOLTZMANN
  };
  enum SIDE_TYPE
  {
    OTHER,
    LINE,
    TRI,
    QUAD
  };  // to calculate areas for pressure bc

  typedef typename EvalT::ScalarT      ScalarT;
  typedef typename EvalT::MeshScalarT  MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  NeumannBase(Teuchos::ParameterList& p);

  void
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d) = 0;

  ScalarT&
  getValue(std::string const& n);

 protected:
  using ICT  = Intrepid2::CellTools<PHX::Device>;
  using IRST = Intrepid2::RealSpaceTools<PHX::Device>;
  using IFST = Intrepid2::FunctionSpaceTools<PHX::Device>;

  Teuchos::RCP<Albany::Application>            app_{Teuchos::null};
  Teuchos::RCP<Albany::Layouts> const&         dl;
  Teuchos::RCP<Albany::MeshSpecsStruct> const& meshSpecs;

  int                                      cellDims{0}, numQPs{0}, numNodes{0}, numCells{0}, maxSideDim{0}, maxNumQpSide{0};
  mutable int                              numBlocks{0};
  Teuchos::Array<int>                      offset;
  int                                      numDOFsSet{0};
  mutable Teuchos::RCP<Teuchos_Comm const> commT;

  // Should only specify flux vector components (dudx, dudy, dudz), dudn, or
  // pressure P

  // dudn scaled
  void
  calc_dudn_const(Kokkos::DynRankView<ScalarT, PHX::Device>& qp_data_returned, ScalarT scale = 1.0) const;

  // robin (also uses flux scaling)
  void
  calc_dudn_robin(Kokkos::DynRankView<ScalarT, PHX::Device>& qp_data_returned, Kokkos::DynRankView<ScalarT, PHX::Device> const& dof_side) const;

  // Stefan-Boltzmann (also uses flux scaling)
  void
  calc_dudn_radiate(Kokkos::DynRankView<ScalarT, PHX::Device>& qp_data_returned, Kokkos::DynRankView<ScalarT, PHX::Device> const& dof_side) const;

  // (dudx, dudy, dudz)
  void
  calc_gradu_dotn_const(
      Kokkos::DynRankView<ScalarT, PHX::Device>&           qp_data_returned,
      Kokkos::DynRankView<MeshScalarT, PHX::Device> const& jacobian_side_refcell,
      shards::CellTopology const&                          celltopo,
      int                                                  local_side_id) const;

  // (t_x, t_y, t_z)
  void
  calc_traction_components(Kokkos::DynRankView<ScalarT, PHX::Device>& qp_data_returned) const;

  // Pressure P
  void
  calc_press(
      Kokkos::DynRankView<ScalarT, PHX::Device>&           qp_data_returned,
      Kokkos::DynRankView<MeshScalarT, PHX::Device> const& jacobian_side_refcell,
      shards::CellTopology const&                          celltopo,
      int                                                  local_side_id) const;

  // ACE Pressure P
  void
  calc_ace_press(
      Kokkos::DynRankView<ScalarT, PHX::Device>&           qp_data_returned,
      Kokkos::DynRankView<MeshScalarT, PHX::Device> const& physPointsSide,
      Kokkos::DynRankView<MeshScalarT, PHX::Device> const& jacobian_side_refcell,
      shards::CellTopology const&                          celltopo,
      int                                                  local_side_id,
      const int                                            workset_num,
      const double                                         current_time) const;

  // Basic hydrostaitc ACE wave pressure NBC
  void
  calc_ace_press_hydrostatic(
      Kokkos::DynRankView<ScalarT, PHX::Device>&           qp_data_returned,
      Kokkos::DynRankView<MeshScalarT, PHX::Device> const& physPointsSide,
      Kokkos::DynRankView<MeshScalarT, PHX::Device> const& jacobian_side_refcell,
      shards::CellTopology const&                          celltopo,
      int                                                  local_side_id,
      const int                                            workset_num,
      const double                                         current_time) const;

  // The following is for the breaking wave formulation of the ACE wave pressure NBC
  ScalarT
  calc_ace_press_at_z_point(
      const double  rho,
      const double  g,
      const double  tm,
      const ScalarT s,
      const ScalarT w,
      const ScalarT k,
      const ScalarT L,
      const ScalarT zval) const;

  ScalarT
  calc_ace_press_at_z_point(const double rho, const double g, const ScalarT s, const ScalarT w, const ScalarT k, const ScalarT zval) const;

  // closed_from bc assignment
  // closed_from bc assignment
  void
  calc_closed_form(
      Kokkos::DynRankView<ScalarT, PHX::Device>&           qp_data_returned,
      Kokkos::DynRankView<MeshScalarT, PHX::Device> const& physPointsSide,
      Kokkos::DynRankView<MeshScalarT, PHX::Device> const& jacobian_side_refcell,
      shards::CellTopology const&                          celltopo,
      int                                                  local_side_id,
      typename Traits::EvalData                            workset) const;

  // Do the side integration
  void
  evaluateNeumannContribution(typename Traits::EvalData d);

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<const MeshScalarT, Cell, Vertex, Dim> coordVec;
  PHX::MDField<ScalarT const>                        dof;

  Teuchos::RCP<shards::CellTopology>                                cellType;
  Teuchos::ArrayRCP<Teuchos::RCP<shards::CellTopology>>             sideType;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>                    cubatureCell;
  Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>> cubatureSide;

  // The basis
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis;

  // Temporary Views
  Kokkos::DynRankView<MeshScalarT, PHX::Device> physPointsCell_buffer;

  Kokkos::DynRankView<ScalarT, PHX::Device> dofCell_buffer;
  Kokkos::DynRankView<ScalarT, PHX::Device> dofCellVec_buffer;

  Kokkos::DynRankView<RealType, PHX::Device> cubPointsSide_buffer;
  Kokkos::DynRankView<RealType, PHX::Device> refPointsSide_buffer;
  Kokkos::DynRankView<RealType, PHX::Device> cubWeightsSide_buffer;
  Kokkos::DynRankView<RealType, PHX::Device> basis_refPointsSide_buffer;

  Kokkos::DynRankView<MeshScalarT, PHX::Device> physPointsSide_buffer;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> jacobianSide_buffer;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> jacobianSide_det_buffer;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> weighted_measure_buffer;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> trans_basis_refPointsSide_buffer;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> weighted_trans_basis_refPointsSide_buffer;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> side_normals_buffer;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> normal_lengths_buffer;

  Kokkos::DynRankView<ScalarT, PHX::Device> dofSide_buffer;
  Kokkos::DynRankView<ScalarT, PHX::Device> dofSideVec_buffer;

  Kokkos::DynRankView<MeshScalarT, PHX::Device> temporary_buffer;
  Kokkos::DynRankView<ScalarT, PHX::Device>     data_buffer;

  // Output:
  Kokkos::DynRankView<ScalarT, PHX::Device> neumann;

  int  numSidesOnElem;
  bool vectorDOF;

  std::string              sideSetID;
  Teuchos::Array<RealType> inputValues;
  std::string              inputConditions;
  std::string              name;

  NEU_TYPE                  bc_type;
  Teuchos::Array<SIDE_TYPE> side_type;
  ScalarT                   const_val;
  ScalarT                   robin_vals[5];  // (dof_value, coeff multiplying difference (dof -
                                            // dof_value), jump)
  // The following are specific to ACE wave pressure BC
  ScalarT water_height_val;
  ScalarT height_above_water_of_max_pressure_val;
  ScalarT wave_length_val;
  ScalarT wave_number_val;
  ScalarT waterH_val;
  ScalarT s_val;
  ScalarT w_val;

  std::vector<ScalarT> dudx;

  std::vector<ScalarT> matScaling;
};

template <typename EvalT, typename Traits>
class Neumann;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************

// **************************************************************
// Residual
// **************************************************************
template <typename Traits>
class Neumann<PHAL::AlbanyTraits::Residual, Traits> : public NeumannBase<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  Neumann(Teuchos::ParameterList& p);
  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
};

// **************************************************************
// Jacobian
// **************************************************************
template <typename Traits>
class Neumann<PHAL::AlbanyTraits::Jacobian, Traits> : public NeumannBase<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  Neumann(Teuchos::ParameterList& p);
  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};

// **************************************************************
// Evaluator to aggregate all Neumann BCs into one "field"
// **************************************************************
template <typename EvalT, typename Traits>
class NeumannAggregator : public PHX::EvaluatorWithBaseImpl<Traits>, public PHX::EvaluatorDerived<EvalT, Traits>
{
 private:
  typedef typename EvalT::ScalarT ScalarT;

 public:
  NeumannAggregator(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData /* d */){};
};

}  // namespace PHAL

#endif
