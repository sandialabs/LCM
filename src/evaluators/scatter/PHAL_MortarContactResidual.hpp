// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_MORTAR_CONTACT_RESIDUAL_HPP
#define PHAL_MORTAR_CONTACT_RESIDUAL_HPP

#include "Albany_Layouts.hpp"
#include "Kokkos_Vector.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Teuchos_ParameterList.hpp"

namespace PHAL {
/** \brief Scatters Lagrange Multipliers from the residual fields into the
    global data structure.  This includes the
    post-processing of the AD data type for all evaluation
    types besides Residual.

*/
// **************************************************************
// Base Class for code that is independent of evaluation type
// **************************************************************

template <typename EvalT, typename Traits>
class MortarContactResidualBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                  public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  MortarContactResidualBase(
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

 protected:
  Albany::AbstractDiscretization::WorksetConn nodeID;
  Albany::DeviceView1d<ST>                    f_kokkos;
  Kokkos::vector<Kokkos::DynRankView<ScalarT const, PHX::Device>, PHX::Device>
      val_kokkos;
};

template <typename EvalT, typename Traits>
class MortarContactResidual;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************

// **************************************************************
// Residual
// **************************************************************
template <typename Traits>
class MortarContactResidual<PHAL::AlbanyTraits::Residual, Traits>
    : public MortarContactResidualBase<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  MortarContactResidual(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);
  void
  evaluateFields(typename Traits::EvalData d);

 protected:
  std::size_t const numFields;

 private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;

 public:
  struct PHAL_MortarContactResRank0_Tag
  {
  };
  struct PHAL_MortarContactResRank1_Tag
  {
  };
  struct PHAL_MortarContactResRank2_Tag
  {
  };

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_MortarContactResRank0_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_MortarContactResRank1_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_MortarContactResRank2_Tag&, const int& cell) const;

 private:
  int numDims;

  typedef MortarContactResidualBase<PHAL::AlbanyTraits::Residual, Traits> Base;
  using Base::f_kokkos;
  using Base::nodeID;
  using Base::val_kokkos;

  typedef typename PHX::Device::execution_space ExecutionSpace;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_MortarContactResRank0_Tag>
      PHAL_MortarContactResRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_MortarContactResRank1_Tag>
      PHAL_MortarContactResRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_MortarContactResRank2_Tag>
      PHAL_MortarContactResRank2_Policy;
};

// **************************************************************
// Jacobian
// **************************************************************
template <typename Traits>
class MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>
    : public MortarContactResidualBase<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  MortarContactResidual(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);
  void
  evaluateFields(typename Traits::EvalData d);

 protected:
  std::size_t const numFields;

 private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;

 public:
  struct PHAL_MortarContactResRank0_Tag
  {
  };
  struct PHAL_MortarContactJacRank0_Adjoint_Tag
  {
  };
  struct PHAL_MortarContactJacRank0_Tag
  {
  };
  struct PHAL_MortarContactResRank1_Tag
  {
  };
  struct PHAL_MortarContactJacRank1_Adjoint_Tag
  {
  };
  struct PHAL_MortarContactJacRank1_Tag
  {
  };
  struct PHAL_MortarContactResRank2_Tag
  {
  };
  struct PHAL_MortarContactJacRank2_Adjoint_Tag
  {
  };
  struct PHAL_MortarContactJacRank2_Tag
  {
  };

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_MortarContactResRank0_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_MortarContactJacRank0_Adjoint_Tag&, const int& cell)
      const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_MortarContactJacRank0_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_MortarContactResRank1_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_MortarContactJacRank1_Adjoint_Tag&, const int& cell)
      const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_MortarContactJacRank1_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_MortarContactResRank2_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_MortarContactJacRank2_Adjoint_Tag&, const int& cell)
      const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const PHAL_MortarContactJacRank2_Tag&, const int& cell) const;

 private:
  int                           neq, nunk, numDims;
  Albany::DeviceLocalMatrix<ST> Jac_kokkos;

  typedef MortarContactResidualBase<PHAL::AlbanyTraits::Jacobian, Traits> Base;
  using Base::f_kokkos;
  using Base::nodeID;
  using Base::val_kokkos;

  typedef typename PHX::Device::execution_space ExecutionSpace;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_MortarContactResRank0_Tag>
      PHAL_MortarContactResRank0_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, PHAL_MortarContactJacRank0_Adjoint_Tag>
          PHAL_MortarContactJacRank0_Adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_MortarContactJacRank0_Tag>
      PHAL_MortarContactJacRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_MortarContactResRank1_Tag>
      PHAL_MortarContactResRank1_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, PHAL_MortarContactJacRank1_Adjoint_Tag>
          PHAL_MortarContactJacRank1_Adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_MortarContactJacRank1_Tag>
      PHAL_MortarContactJacRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_MortarContactResRank2_Tag>
      PHAL_MortarContactResRank2_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, PHAL_MortarContactJacRank2_Adjoint_Tag>
          PHAL_MortarContactJacRank2_Adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_MortarContactJacRank2_Tag>
      PHAL_MortarContactJacRank2_Policy;
};

}  // namespace PHAL

#endif
