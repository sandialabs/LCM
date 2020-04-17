// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#if defined(ALBANY_TIMER)
#include <chrono>
#endif
#include "Albany_ContactManager.hpp"
#include "Albany_Macros.hpp"
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace PHAL {

template <typename EvalT, typename Traits>
MortarContactResidualBase<EvalT, Traits>::MortarContactResidualBase(
    Teuchos::ParameterList const&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
{
  scatter_operation =
      Teuchos::rcp(new PHX::Tag<ScalarT>("MortarContact", dl->dummy));

  const Teuchos::ArrayRCP<std::string>& names =
      p.get<Teuchos::ArrayRCP<std::string>>("Residual Names");

  numFieldsBase             = names.size();
  std::size_t const num_val = numFieldsBase;
  val.resize(num_val);
  for (std::size_t eq = 0; eq < numFieldsBase; ++eq) {
    PHX::MDField<ScalarT const, Cell, Node> mdf(names[eq], dl->node_scalar);
    val[eq] = mdf;
    this->addDependentField(val[eq]);
  }

  val_kokkos.resize(numFieldsBase);

  if (p.isType<int>("Offset of First DOF"))
    offset = p.get<int>("Offset of First DOF");
  else
    offset = 0;

  this->addEvaluatedField(*scatter_operation);

  this->setName("MortarContactResidual" + PHX::print<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
MortarContactResidualBase<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  for (std::size_t eq = 0; eq < numFieldsBase; ++eq)
    this->utils.setFieldData(val[eq], fm);
  numNodes = val[0].extent(1);
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template <typename Traits>
MortarContactResidual<PHAL::AlbanyTraits::Residual, Traits>::
    MortarContactResidual(
        Teuchos::ParameterList const&        p,
        const Teuchos::RCP<Albany::Layouts>& dl)
    : MortarContactResidualBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl),
      numFields(
          MortarContactResidualBase<PHAL::AlbanyTraits::Residual, Traits>::
              numFieldsBase)
{
}

// **********************************************************************
// Kokkos kernels
template <typename Traits>
KOKKOS_INLINE_FUNCTION void
MortarContactResidual<PHAL::AlbanyTraits::Residual, Traits>::operator()(
    const PHAL_MortarContactResRank0_Tag&,
    int const& cell) const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t eq = 0; eq < numFields; eq++) {
      const LO id = nodeID(cell, node, this->offset + eq);
      Kokkos::atomic_fetch_add(&f_kokkos(id), val_kokkos[eq](cell, node));
    }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
MortarContactResidual<PHAL::AlbanyTraits::Residual, Traits>::operator()(
    const PHAL_MortarContactResRank1_Tag&,
    int const& cell) const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t eq = 0; eq < numFields; eq++) {
      const LO id = nodeID(cell, node, this->offset + eq);
      Kokkos::atomic_fetch_add(&f_kokkos(id), this->valVec(cell, node, eq));
    }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
MortarContactResidual<PHAL::AlbanyTraits::Residual, Traits>::operator()(
    const PHAL_MortarContactResRank2_Tag&,
    int const& cell) const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t i = 0; i < numDims; i++)
      for (std::size_t j = 0; j < numDims; j++) {
        const LO id = nodeID(cell, node, this->offset + i * numDims + j);
        Kokkos::atomic_fetch_add(
            &f_kokkos(id), this->valTensor(cell, node, i, j));
      }
}

// **********************************************************************
template <typename Traits>
void
MortarContactResidual<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
#if defined(ALBANY_TIMER)
  auto start = std::chrono::high_resolution_clock::now();
#endif
  // Get map for local data structures
  nodeID = workset.wsElNodeEqID;

  // Get device view from thyra vector
  f_kokkos = Albany::getNonconstDeviceData(workset.f);

  // Get MDField views from std::vector
  for (int i = 0; i < numFields; i++) {
    val_kokkos[i] = this->val[i].get_view();
  }

  Kokkos::parallel_for(
      PHAL_MortarContactResRank0_Policy(0, workset.numCells), *this);
  cudaCheckError();

#if defined(ALBANY_TIMER)
  PHX::Device::fence();
  auto      elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec =
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout << "Mortar Contact Residual time = " << millisec << "  "
            << microseconds << std::endl;
#endif
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template <typename Traits>
MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
    MortarContactResidual(
        Teuchos::ParameterList const&        p,
        const Teuchos::RCP<Albany::Layouts>& dl)
    : MortarContactResidualBase<PHAL::AlbanyTraits::Jacobian, Traits>(p, dl),
      numFields(
          MortarContactResidualBase<PHAL::AlbanyTraits::Jacobian, Traits>::
              numFieldsBase)
{
}

// **********************************************************************
// Kokkos kernels
template <typename Traits>
KOKKOS_INLINE_FUNCTION void
MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(
    const PHAL_MortarContactResRank0_Tag&,
    int const& cell) const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t eq = 0; eq < numFields; eq++) {
      const LO id = nodeID(cell, node, this->offset + eq);
      Kokkos::atomic_fetch_add(
          &f_kokkos(id), (val_kokkos[eq](cell, node)).val());
    }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(
    const PHAL_MortarContactJacRank0_Adjoint_Tag&,
    int const& cell) const
{
  // int const neq = nodeID.extent(2);
  // int const nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO colT[500];
  LO rowT;
  // std::vector<LO> colT(nunk);
  // colT=(LO*) Kokkos::cuda_malloc<PHX::Device>(nunk*sizeof(LO));

  if (nunk > 500) Kokkos::abort("ERROR (MortarContactResidual): nunk > 500");

  for (int node_col = 0; node_col < this->numNodes; node_col++) {
    for (int eq_col = 0; eq_col < neq; eq_col++) {
      colT[neq * node_col + eq_col] = nodeID(cell, node_col, eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      rowT        = nodeID(cell, node, this->offset + eq);
      auto valptr = val_kokkos[eq](cell, node);
      for (int lunk = 0; lunk < nunk; lunk++) {
        ST val = valptr.fastAccessDx(lunk);
        Jac_kokkos.sumIntoValues(colT[lunk], &rowT, 1, &val, false, true);
      }
    }
  }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(
    const PHAL_MortarContactJacRank0_Tag&,
    int const& cell) const
{
  // int const neq = nodeID.extent(2);
  // int const nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  // colT=(LO*) Kokkos::cuda_malloc<PHX::Device>(nunk*sizeof(LO));
  LO rowT;
  LO colT[500];
  ST vals[500];
  // std::vector<LO> colT(nunk);
  // std::vector<ST> vals(nunk);

  if (nunk > 500) Kokkos::abort("ERROR (MortarContactResidual): nunk > 500");

  for (int node_col = 0, i = 0; node_col < this->numNodes; node_col++) {
    for (int eq_col = 0; eq_col < neq; eq_col++) {
      colT[neq * node_col + eq_col] = nodeID(cell, node_col, eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      rowT        = nodeID(cell, node, this->offset + eq);
      auto valptr = val_kokkos[eq](cell, node);
      for (int i = 0; i < nunk; ++i) vals[i] = valptr.fastAccessDx(i);
      Jac_kokkos.sumIntoValues(rowT, colT, nunk, vals, false, true);
    }
  }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(
    const PHAL_MortarContactResRank1_Tag&,
    int const& cell) const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t eq = 0; eq < numFields; eq++) {
      const LO id = nodeID(cell, node, this->offset + eq);
      Kokkos::atomic_fetch_add(
          &f_kokkos(id), (this->valVec(cell, node, eq)).val());
    }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(
    const PHAL_MortarContactJacRank1_Adjoint_Tag&,
    int const& cell) const
{
  // int const neq = nodeID.extent(2);
  // int const nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO colT[500];
  LO rowT;
  ST vals[500];
  // std::vector<ST> vals(nunk);
  // std::vector<LO> colT(nunk);
  // colT=(LO*) Kokkos::malloc<PHX::Device>(nunk*sizeof(LO));

  if (nunk > 500) Kokkos::abort("ERROR (MortarContactResidual): nunk > 500");

  for (int node_col = 0, i = 0; node_col < this->numNodes; node_col++) {
    for (int eq_col = 0; eq_col < neq; eq_col++) {
      colT[neq * node_col + eq_col] = nodeID(cell, node_col, eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      rowT = nodeID(cell, node, this->offset + eq);
      if (((this->valVec)(cell, node, eq)).hasFastAccess()) {
        for (int lunk = 0; lunk < nunk; lunk++) {
          ST val = ((this->valVec)(cell, node, eq)).fastAccessDx(lunk);
          Jac_kokkos.sumIntoValues(colT[lunk], &rowT, 1, &val, false, true);
        }
      }  // has fast access
    }
  }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(
    const PHAL_MortarContactJacRank1_Tag&,
    int const& cell) const
{
  // int const neq = nodeID.extent(2);
  // int const nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO colT[500];
  LO rowT;
  ST vals[500];
  // std::vector<LO> colT(nunk);
  // colT=(LO*) Kokkos::malloc<PHX::Device>(nunk*sizeof(LO));
  // std::vector<ST> vals(nunk);

  if (nunk > 500) Kokkos::abort("ERROR (MortarContactResidual): nunk > 500");

  for (int node_col = 0, i = 0; node_col < this->numNodes; node_col++) {
    for (int eq_col = 0; eq_col < neq; eq_col++) {
      colT[neq * node_col + eq_col] = nodeID(cell, node_col, eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      rowT = nodeID(cell, node, this->offset + eq);
      if (((this->valVec)(cell, node, eq)).hasFastAccess()) {
        for (int i = 0; i < nunk; ++i)
          vals[i] = (this->valVec)(cell, node, eq).fastAccessDx(i);
        Jac_kokkos.sumIntoValues(rowT, colT, nunk, vals, false, true);
      }
    }
  }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(
    const PHAL_MortarContactResRank2_Tag&,
    int const& cell) const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t i = 0; i < numDims; i++)
      for (std::size_t j = 0; j < numDims; j++) {
        const LO id = nodeID(cell, node, this->offset + i * numDims + j);
        Kokkos::atomic_fetch_add(
            &f_kokkos(id), (this->valTensor(cell, node, i, j)).val());
      }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(
    const PHAL_MortarContactJacRank2_Adjoint_Tag&,
    int const& cell) const
{
  // int const neq = nodeID.extent(2);
  // int const nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO colT[500];
  LO rowT;
  // std::vector<LO> colT(nunk);
  // colT=(LO*) Kokkos::malloc<PHX::Device>(nunk*sizeof(LO));

  if (nunk > 500) Kokkos::abort("ERROR (MortarContactResidual): nunk > 500");

  for (int node_col = 0, i = 0; node_col < this->numNodes; node_col++) {
    for (int eq_col = 0; eq_col < neq; eq_col++) {
      colT[neq * node_col + eq_col] = nodeID(cell, node_col, eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      rowT = nodeID(cell, node, this->offset + eq);
      if (((this->valTensor)(cell, node, eq / numDims, eq % numDims))
              .hasFastAccess()) {
        for (int lunk = 0; lunk < nunk; lunk++) {
          ST val = ((this->valTensor)(cell, node, eq / numDims, eq % numDims))
                       .fastAccessDx(lunk);
          Jac_kokkos.sumIntoValues(colT[lunk], &rowT, 1, &val, false, true);
        }
      }  // has fast access
    }
  }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(
    const PHAL_MortarContactJacRank2_Tag&,
    int const& cell) const
{
  // int const neq = nodeID.extent(2);
  // int const nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO colT[500];
  LO rowT;
  ST vals[500];
  // std::vector<LO> colT(nunk);
  // colT=(LO*) Kokkos::malloc<PHX::Device>(nunk*sizeof(LO));
  // std::vector<ST> vals(nunk);

  if (nunk > 500) Kokkos::abort("ERROR (MortarContactResidual): nunk > 500");

  for (int node_col = 0, i = 0; node_col < this->numNodes; node_col++) {
    for (int eq_col = 0; eq_col < neq; eq_col++) {
      colT[neq * node_col + eq_col] = nodeID(cell, node_col, eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      rowT = nodeID(cell, node, this->offset + eq);
      if (((this->valTensor)(cell, node, eq / numDims, eq % numDims))
              .hasFastAccess()) {
        for (int i = 0; i < nunk; ++i)
          vals[i] = (this->valTensor)(cell, node, eq / numDims, eq % numDims)
                        .fastAccessDx(i);
        Jac_kokkos.sumIntoValues(rowT, colT, nunk, vals, false, true);
      }
    }
  }
}

// **********************************************************************
template <typename Traits>
void
MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
#if defined(ALBANY_TIMER)
  auto start = std::chrono::high_resolution_clock::now();
#endif
  // Get map for local data structures
  nodeID = workset.wsElNodeEqID;

  // Get dimensions
  neq  = nodeID.extent(2);
  nunk = neq * this->numNodes;

  // Get Tpetra vector view and local matrix
  bool const loadResid = Teuchos::nonnull(workset.f);
  if (loadResid) { f_kokkos = Albany::getNonconstDeviceData(workset.f); }
  Jac_kokkos = Albany::getNonconstDeviceData(workset.Jac);

  // Get MDField views from std::vector
  for (int i = 0; i < numFields; i++) val_kokkos[i] = this->val[i].get_view();

  if (loadResid) {
    Kokkos::parallel_for(
        PHAL_MortarContactResRank0_Policy(0, workset.numCells), *this);
    cudaCheckError();
  }

  if (workset.is_adjoint) {
    Kokkos::parallel_for(
        PHAL_MortarContactJacRank0_Adjoint_Policy(0, workset.numCells), *this);
    cudaCheckError();
  } else {
    Kokkos::parallel_for(
        PHAL_MortarContactJacRank0_Policy(0, workset.numCells), *this);
    cudaCheckError();
  }

#if defined(ALBANY_TIMER)
  PHX::Device::fence();
  auto      elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec =
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout << "Mortar Contact Jacobian time = " << millisec << "  "
            << microseconds << std::endl;
#endif
}

}  // namespace PHAL
