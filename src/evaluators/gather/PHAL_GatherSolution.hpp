// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_GATHER_SOLUTION_HPP
#define PHAL_GATHER_SOLUTION_HPP

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_Layouts.hpp"
#include "Kokkos_Vector.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Teuchos_ParameterList.hpp"

namespace PHAL {
/** \brief Gathers solution values from the Newton solution vector into
    the nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the solution
    names vector.

*/
// **************************************************************
// Base Class with Generic Implementations: Specializations for
// Automatic Differentiation Below
// **************************************************************

template <typename EvalT, typename Traits>
class GatherSolutionBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  GatherSolutionBase(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  // This function requires template specialization, in derived class below
  virtual void
  evaluateFields(typename Traits::EvalData d) = 0;

 protected:
  typedef typename EvalT::ScalarT                   ScalarT;
  std::vector<PHX::MDField<ScalarT, Cell, Node>>    val;
  std::vector<PHX::MDField<ScalarT, Cell, Node>>    val_dot;
  std::vector<PHX::MDField<ScalarT, Cell, Node>>    val_dotdot;
  PHX::MDField<ScalarT, Cell, Node, VecDim>         valVec;
  PHX::MDField<ScalarT, Cell, Node, VecDim>         valVec_dot;
  PHX::MDField<ScalarT, Cell, Node, VecDim>         valVec_dotdot;
  PHX::MDField<ScalarT, Cell, Node, VecDim, VecDim> valTensor;
  PHX::MDField<ScalarT, Cell, Node, VecDim, VecDim> valTensor_dot;
  PHX::MDField<ScalarT, Cell, Node, VecDim, VecDim> valTensor_dotdot;
  std::size_t                                       numNodes;
  std::size_t numFieldsBase;  // Number of fields gathered in this call
  std::size_t offset;  // Offset of first DOF being gathered when numFields<neq
  unsigned short int tensorRank;
  bool               enableTransient;
  bool               enableAcceleration;

 protected:
  Albany::WorksetConn            nodeID;
  Albany::DeviceView1d<const ST> x_constView, xdot_constView, xdotdot_constView;

  typedef Kokkos::vector<Kokkos::DynRankView<ScalarT, PHX::Device>, PHX::Device>
                     KV;
  KV                 val_kokkos, val_dot_kokkos, val_dotdot_kokkos;
  typename KV::t_dev d_val, d_val_dot, d_val_dotdot;
};

template <typename EvalT, typename Traits>
class GatherSolution;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************

// **************************************************************
// Residual
// **************************************************************
template <typename Traits>
class GatherSolution<PHAL::AlbanyTraits::Residual, Traits>
    : public GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  GatherSolution(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);
  // Old constructor, still needed by BCs that use PHX Factory
  GatherSolution(Teuchos::ParameterList const& p);
  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  int const                                              numFields;

 public:
  struct PHAL_GatherSolRank2_Tag
  {
  };
  struct PHAL_GatherSolRank2_Transient_Tag
  {
  };
  struct PHAL_GatherSolRank2_Acceleration_Tag
  {
  };

  struct PHAL_GatherSolRank1_Tag
  {
  };
  struct PHAL_GatherSolRank1_Transient_Tag
  {
  };
  struct PHAL_GatherSolRank1_Acceleration_Tag
  {
  };

  struct PHAL_GatherSolRank0_Tag
  {
  };
  struct PHAL_GatherSolRank0_Transient_Tag
  {
  };
  struct PHAL_GatherSolRank0_Acceleration_Tag
  {
  };

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherSolRank2_Tag&, int const& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherSolRank2_Transient_Tag&, int const& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherSolRank2_Acceleration_Tag&, int const& cell)
      const;

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherSolRank1_Tag&, int const& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherSolRank1_Transient_Tag&, int const& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherSolRank1_Acceleration_Tag&, int const& cell)
      const;

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherSolRank0_Tag&, int const& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherSolRank0_Transient_Tag&, int const& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherSolRank0_Acceleration_Tag&, int const& cell)
      const;

 private:
  int numDim;

  typedef GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits> Base;
  using Base::d_val;
  using Base::d_val_dot;
  using Base::d_val_dotdot;
  using Base::nodeID;
  using Base::val_dot_kokkos;
  using Base::val_dotdot_kokkos;
  using Base::val_kokkos;
  using Base::x_constView;
  using Base::xdot_constView;
  using Base::xdotdot_constView;

  typedef typename PHX::Device::execution_space ExecutionSpace;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_GatherSolRank2_Tag>
      PHAL_GatherSolRank2_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_GatherSolRank2_Transient_Tag>
      PHAL_GatherSolRank2_Transient_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, PHAL_GatherSolRank2_Acceleration_Tag>
          PHAL_GatherSolRank2_Acceleration_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_GatherSolRank1_Tag>
      PHAL_GatherSolRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_GatherSolRank1_Transient_Tag>
      PHAL_GatherSolRank1_Transient_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, PHAL_GatherSolRank1_Acceleration_Tag>
          PHAL_GatherSolRank1_Acceleration_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_GatherSolRank0_Tag>
      PHAL_GatherSolRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_GatherSolRank0_Transient_Tag>
      PHAL_GatherSolRank0_Transient_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, PHAL_GatherSolRank0_Acceleration_Tag>
          PHAL_GatherSolRank0_Acceleration_Policy;
};

// **************************************************************
// Jacobian
// **************************************************************
template <typename Traits>
class GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>
    : public GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  GatherSolution(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(Teuchos::ParameterList const& p);
  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  int const                                              numFields;

 public:
  struct PHAL_GatherJacRank2_Tag
  {
  };
  struct PHAL_GatherJacRank2_Transient_Tag
  {
  };
  struct PHAL_GatherJacRank2_Acceleration_Tag
  {
  };

  struct PHAL_GatherJacRank1_Tag
  {
  };
  struct PHAL_GatherJacRank1_Transient_Tag
  {
  };
  struct PHAL_GatherJacRank1_Acceleration_Tag
  {
  };

  struct PHAL_GatherJacRank0_Tag
  {
  };
  struct PHAL_GatherJacRank0_Transient_Tag
  {
  };
  struct PHAL_GatherJacRank0_Acceleration_Tag
  {
  };

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherJacRank2_Tag&, int const& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherJacRank2_Transient_Tag&, int const& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherJacRank2_Acceleration_Tag&, int const& cell)
      const;

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherJacRank1_Tag&, int const& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherJacRank1_Transient_Tag&, int const& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherJacRank1_Acceleration_Tag&, int const& cell)
      const;

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherJacRank0_Tag&, int const& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherJacRank0_Transient_Tag&, int const& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_GatherJacRank0_Acceleration_Tag&, int const& cell)
      const;

 private:
  int    neq, numDim;
  double j_coeff, n_coeff, m_coeff;

  typedef GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits> Base;
  using Base::d_val;
  using Base::d_val_dot;
  using Base::d_val_dotdot;
  using Base::nodeID;
  using Base::val_dot_kokkos;
  using Base::val_dotdot_kokkos;
  using Base::val_kokkos;
  using Base::x_constView;
  using Base::xdot_constView;
  using Base::xdotdot_constView;

  typedef typename PHX::Device::execution_space ExecutionSpace;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_GatherJacRank2_Tag>
      PHAL_GatherJacRank2_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_GatherJacRank2_Transient_Tag>
      PHAL_GatherJacRank2_Transient_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, PHAL_GatherJacRank2_Acceleration_Tag>
          PHAL_GatherJacRank2_Acceleration_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_GatherJacRank1_Tag>
      PHAL_GatherJacRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_GatherJacRank1_Transient_Tag>
      PHAL_GatherJacRank1_Transient_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, PHAL_GatherJacRank1_Acceleration_Tag>
          PHAL_GatherJacRank1_Acceleration_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_GatherJacRank0_Tag>
      PHAL_GatherJacRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_GatherJacRank0_Transient_Tag>
      PHAL_GatherJacRank0_Transient_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, PHAL_GatherJacRank0_Acceleration_Tag>
          PHAL_GatherJacRank0_Acceleration_Policy;
};

}  // namespace PHAL

#endif  // PHAL_GATHER_SOLUTION_HPP
