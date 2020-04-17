// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_SCATTER_RESIDUAL_HPP
#define PHAL_SCATTER_RESIDUAL_HPP

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_KokkosTypes.hpp"
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
/** \brief Scatters result from the residual fields into the global data
    structurs. This includes the post-processing of the AD data type for all
    evaluation types besides Residual.

*/
// **************************************************************
// Base Class for code that is independent of evaluation type
// **************************************************************

template <typename EvalT, typename Traits>
class ScatterResidualBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ScatterResidualBase(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  virtual void
  evaluateFields(typename Traits::EvalData d) = 0;

 protected:
  typedef typename EvalT::ScalarT                      ScalarT;
  Teuchos::RCP<PHX::FieldTag>                          scatter_operation;
  std::vector<PHX::MDField<ScalarT const, Cell, Node>> val;
  PHX::MDField<ScalarT const, Cell, Node, Dim>         valVec;
  PHX::MDField<ScalarT const, Cell, Node, Dim, Dim>    valTensor;
  std::size_t                                          numNodes;
  std::size_t numFieldsBase;  // Number of fields gathered in this call
  std::size_t offset;  // Offset of first DOF being gathered when numFields<neq

  unsigned short int tensorRank;

 protected:
  Albany::WorksetConn      nodeID;
  Albany::DeviceView1d<ST> f_kokkos;
  Kokkos::vector<Kokkos::DynRankView<ScalarT const, PHX::Device>, PHX::Device>
      val_kokkos;
};

template <typename EvalT, typename Traits>
class ScatterResidual;

template <typename EvalT, typename Traits>
class ScatterResidualWithExtrudedParams : public ScatterResidual<EvalT, Traits>
{
 public:
  ScatterResidualWithExtrudedParams(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl)
      : ScatterResidual<EvalT, Traits>(p, dl)
  {
    extruded_params_levels = p.get<Teuchos::RCP<std::map<std::string, int>>>(
        "Extruded Params Levels");
  };

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm)
  {
    ScatterResidual<EvalT, Traits>::postRegistrationSetup(d, vm);
  }

  void
  evaluateFields(typename Traits::EvalData d)
  {
    ScatterResidual<EvalT, Traits>::evaluateFields(d);
  }

 protected:
  typedef typename EvalT::ScalarT          ScalarT;
  Teuchos::RCP<std::map<std::string, int>> extruded_params_levels;
};

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************

// **************************************************************
// Residual
// **************************************************************
template <typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::Residual, Traits>
    : public ScatterResidualBase<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  ScatterResidual(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);
  void
  evaluateFields(typename Traits::EvalData d);

 protected:
  std::size_t const numFields;

 private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;

 public:
  struct PHAL_ScatterResRank0_Tag
  {
  };
  struct PHAL_ScatterResRank1_Tag
  {
  };
  struct PHAL_ScatterResRank2_Tag
  {
  };

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_ScatterResRank0_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_ScatterResRank1_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_ScatterResRank2_Tag&, const int& cell) const;

 private:
  int numDims;

  typedef ScatterResidualBase<PHAL::AlbanyTraits::Residual, Traits> Base;
  using Base::f_kokkos;
  using Base::nodeID;
  using Base::val_kokkos;

  typedef typename PHX::Device::execution_space ExecutionSpace;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank0_Tag>
      PHAL_ScatterResRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank1_Tag>
      PHAL_ScatterResRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank2_Tag>
      PHAL_ScatterResRank2_Policy;
};

// **************************************************************
// Jacobian
// **************************************************************
template <typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>
    : public ScatterResidualBase<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  ScatterResidual(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);
  void
  evaluateFields(typename Traits::EvalData d);

 protected:
  std::size_t const numFields;

 private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;

 public:
  struct PHAL_ScatterResRank0_Tag
  {
  };
  struct PHAL_ScatterJacRank0_Adjoint_Tag
  {
  };
  struct PHAL_ScatterJacRank0_Tag
  {
  };
  struct PHAL_ScatterResRank1_Tag
  {
  };
  struct PHAL_ScatterJacRank1_Adjoint_Tag
  {
  };
  struct PHAL_ScatterJacRank1_Tag
  {
  };
  struct PHAL_ScatterResRank2_Tag
  {
  };
  struct PHAL_ScatterJacRank2_Adjoint_Tag
  {
  };
  struct PHAL_ScatterJacRank2_Tag
  {
  };

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_ScatterResRank0_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_ScatterJacRank0_Adjoint_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_ScatterJacRank0_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_ScatterResRank1_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_ScatterJacRank1_Adjoint_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_ScatterJacRank1_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_ScatterResRank2_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_ScatterJacRank2_Adjoint_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_ScatterJacRank2_Tag&, const int& cell) const;

 private:
  int                           neq, nunk, numDims;
  Albany::DeviceLocalMatrix<ST> Jac_kokkos;

  typedef ScatterResidualBase<PHAL::AlbanyTraits::Jacobian, Traits> Base;
  using Base::f_kokkos;
  using Base::nodeID;
  using Base::val_kokkos;

  typedef typename PHX::Device::execution_space ExecutionSpace;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank0_Tag>
      PHAL_ScatterResRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank0_Adjoint_Tag>
      PHAL_ScatterJacRank0_Adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank0_Tag>
      PHAL_ScatterJacRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank1_Tag>
      PHAL_ScatterResRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank1_Adjoint_Tag>
      PHAL_ScatterJacRank1_Adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank1_Tag>
      PHAL_ScatterJacRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank2_Tag>
      PHAL_ScatterResRank2_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank2_Adjoint_Tag>
      PHAL_ScatterJacRank2_Adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank2_Tag>
      PHAL_ScatterJacRank2_Policy;
};

}  // namespace PHAL

#endif
