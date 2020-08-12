// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if defined(ALBANY_TIMER)
#include <chrono>
#endif

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"
#include "PHAL_ScatterResidual.hpp"
#include "Phalanx_DataLayout.hpp"

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace PHAL {

template <typename EvalT, typename Traits>
ScatterResidualBase<EvalT, Traits>::ScatterResidualBase(
    Teuchos::ParameterList const&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
{
  std::string fieldName;
  if (p.isType<std::string>("Scatter Field Name"))
    fieldName = p.get<std::string>("Scatter Field Name");
  else
    fieldName = "Scatter";

  scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));

  Teuchos::ArrayRCP<std::string> names;
  if (p.isType<Teuchos::ArrayRCP<std::string>>("Residual Names")) {
    names = p.get<Teuchos::ArrayRCP<std::string>>("Residual Names");
  } else if (p.isType<std::string>("Residual Name")) {
    names = Teuchos::ArrayRCP<std::string>(1, p.get<std::string>("Residual Name"));
  } else {
    ALBANY_ABORT(
        "Error! You must specify either the std::string 'Residual Name', "
        "or the Teuchos::ArrayRCP<std::string> 'Residual Names'.\n");
  }

  tensorRank = p.get<int>("Tensor Rank");

  if (tensorRank == 0) {
    // scalar
    numFieldsBase             = names.size();
    std::size_t const num_val = numFieldsBase;
    val.resize(num_val);
    for (std::size_t eq = 0; eq < numFieldsBase; ++eq) {
      PHX::MDField<ScalarT const, Cell, Node> mdf(names[eq], dl->node_scalar);
      val[eq] = mdf;
      this->addDependentField(val[eq]);
    }
  } else if (tensorRank == 1) {
    // vector
    PHX::MDField<ScalarT const, Cell, Node, Dim> mdf(names[0], dl->node_vector);
    valVec = mdf;
    this->addDependentField(valVec);
    numFieldsBase = dl->node_vector->extent(2);
  } else if (tensorRank == 2) {
    // tensor
    PHX::MDField<ScalarT const, Cell, Node, Dim, Dim> mdf(names[0], dl->node_tensor);
    valTensor = mdf;
    this->addDependentField(valTensor);
    numFieldsBase = (dl->node_tensor->extent(2)) * (dl->node_tensor->extent(3));
  }

  if (tensorRank == 0) {
    val_kokkos.resize(numFieldsBase);
  }

  if (p.isType<int>("Offset of First DOF")) {
    offset = p.get<int>("Offset of First DOF");
  } else {
    offset = 0;
  }

  this->addEvaluatedField(*scatter_operation);

  this->setName(fieldName + PHX::print<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
ScatterResidualBase<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  if (tensorRank == 0) {
    for (std::size_t eq = 0; eq < numFieldsBase; ++eq) {
      this->utils.setFieldData(val[eq], fm);
    }
    numNodes = val[0].extent(1);
  } else if (tensorRank == 1) {
    this->utils.setFieldData(valVec, fm);
    numNodes = valVec.extent(1);
  } else if (tensorRank == 2) {
    this->utils.setFieldData(valTensor, fm);
    numNodes = valTensor.extent(1);
  }
  d.fill_field_dependencies(this->dependentFields(), this->evaluatedFields());
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template <typename Traits>
ScatterResidual<PHAL::AlbanyTraits::Residual, Traits>::ScatterResidual(
    Teuchos::ParameterList const&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : ScatterResidualBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl),
      numFields(ScatterResidualBase<PHAL::AlbanyTraits::Residual, Traits>::numFieldsBase)
{
}

// **********************************************************************
// Kokkos kernels
template <typename Traits>
KOKKOS_INLINE_FUNCTION void
ScatterResidual<PHAL::AlbanyTraits::Residual, Traits>::operator()(const PHAL_ScatterResRank0_Tag&, int const& cell)
    const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t eq = 0; eq < numFields; eq++) {
      const LO id = nodeID(cell, node, this->offset + eq);
      Kokkos::atomic_fetch_add(&f_kokkos(id), val_kokkos[eq](cell, node));
    }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
ScatterResidual<PHAL::AlbanyTraits::Residual, Traits>::operator()(const PHAL_ScatterResRank1_Tag&, int const& cell)
    const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t eq = 0; eq < numFields; eq++) {
      const LO id = nodeID(cell, node, this->offset + eq);
      Kokkos::atomic_fetch_add(&f_kokkos(id), this->valVec(cell, node, eq));
    }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
ScatterResidual<PHAL::AlbanyTraits::Residual, Traits>::operator()(const PHAL_ScatterResRank2_Tag&, int const& cell)
    const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t i = 0; i < numDims; i++)
      for (std::size_t j = 0; j < numDims; j++) {
        const LO id = nodeID(cell, node, this->offset + i * numDims + j);
        Kokkos::atomic_fetch_add(&f_kokkos(id), this->valTensor(cell, node, i, j));
      }
}

// **********************************************************************
template <typename Traits>
void
ScatterResidual<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Thyra_Vector> f = workset.f;

#if defined(ALBANY_TIMER)
  auto start = std::chrono::high_resolution_clock::now();
#endif
  // Get map for local data structures
  nodeID = workset.wsElNodeEqID;

  // Get Tpetra vector view from a specific device
  f_kokkos = Albany::getNonconstDeviceData(f);

  if (this->tensorRank == 0) {
    // Get MDField views from std::vector
    for (int i = 0; i < numFields; i++) val_kokkos[i] = this->val[i].get_view();

    Kokkos::parallel_for(PHAL_ScatterResRank0_Policy(0, workset.numCells), *this);
    cudaCheckError();
  } else if (this->tensorRank == 1) {
    Kokkos::parallel_for(PHAL_ScatterResRank1_Policy(0, workset.numCells), *this);
    cudaCheckError();
  } else if (this->tensorRank == 2) {
    numDims = this->valTensor.extent(2);
    Kokkos::parallel_for(PHAL_ScatterResRank2_Policy(0, workset.numCells), *this);
    cudaCheckError();
  }

#if defined(ALBANY_TIMER)
  PHX::Device::fence();
  auto      elapsed      = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec     = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout << "Scatter Residual time = " << millisec << "  " << microseconds << std::endl;
#endif
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template <typename Traits>
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::ScatterResidual(
    Teuchos::ParameterList const&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : ScatterResidualBase<PHAL::AlbanyTraits::Jacobian, Traits>(p, dl),
      numFields(ScatterResidualBase<PHAL::AlbanyTraits::Jacobian, Traits>::numFieldsBase)
{
}

// **********************************************************************
// Kokkos kernels
template <typename Traits>
KOKKOS_INLINE_FUNCTION void
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(const PHAL_ScatterResRank0_Tag&, int const& cell)
    const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t eq = 0; eq < numFields; eq++) {
      const LO id = nodeID(cell, node, this->offset + eq);
      Kokkos::atomic_fetch_add(&f_kokkos(id), (val_kokkos[eq](cell, node)).val());
    }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(
    const PHAL_ScatterJacRank0_Adjoint_Tag&,
    int const& cell) const
{
  // int const neq = nodeID.extent(2);
  // int const nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO col[500];
  LO row;

  if (nunk > 500) {
    Kokkos::abort("ERROR (ScatterResidual): nunk > 500");
  }

  for (int node_col = 0; node_col < this->numNodes; node_col++) {
    for (int eq_col = 0; eq_col < neq; eq_col++) {
      col[neq * node_col + eq_col] = nodeID(cell, node_col, eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      row         = nodeID(cell, node, this->offset + eq);
      auto valptr = val_kokkos[eq](cell, node);
      for (int lunk = 0; lunk < nunk; lunk++) {
        ST val = valptr.fastAccessDx(lunk);
        Jac_kokkos.sumIntoValues(col[lunk], &row, 1, &val, false, true);
      }
    }
  }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(const PHAL_ScatterJacRank0_Tag&, int const& cell)
    const
{
  // int const neq = nodeID.extent(2);
  // int const nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO row;
  LO col[500];
  ST vals[500];

  if (nunk > 500) {
    Kokkos::abort("ERROR (ScatterResidual): nunk > 500");
  }

  for (int node_col = 0; node_col < this->numNodes; node_col++) {
    for (int eq_col = 0; eq_col < neq; eq_col++) {
      col[neq * node_col + eq_col] = nodeID(cell, node_col, eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      row         = nodeID(cell, node, this->offset + eq);
      auto valptr = val_kokkos[eq](cell, node);
      for (int i = 0; i < nunk; ++i) vals[i] = valptr.fastAccessDx(i);
      Jac_kokkos.sumIntoValues(row, col, nunk, vals, false, true);
    }
  }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(const PHAL_ScatterResRank1_Tag&, int const& cell)
    const
{
  for (std::size_t node = 0; node < this->numNodes; node++) {
    for (std::size_t eq = 0; eq < numFields; eq++) {
      const LO id = nodeID(cell, node, this->offset + eq);
      Kokkos::atomic_fetch_add(&f_kokkos(id), (this->valVec(cell, node, eq)).val());
    }
  }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(
    const PHAL_ScatterJacRank1_Adjoint_Tag&,
    int const& cell) const
{
  // int const neq = nodeID.extent(2);
  // int const nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO col[500];
  LO row;

  if (nunk > 500) {
    Kokkos::abort("ERROR (ScatterResidual): nunk > 500");
  }

  for (int node_col = 0; node_col < this->numNodes; node_col++) {
    for (int eq_col = 0; eq_col < neq; eq_col++) {
      col[neq * node_col + eq_col] = nodeID(cell, node_col, eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      row = nodeID(cell, node, this->offset + eq);
      if (((this->valVec)(cell, node, eq)).hasFastAccess()) {
        for (int lunk = 0; lunk < nunk; lunk++) {
          ST val = ((this->valVec)(cell, node, eq)).fastAccessDx(lunk);
          Jac_kokkos.sumIntoValues(col[lunk], &row, 1, &val, false, true);
        }
      }  // has fast access
    }
  }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(const PHAL_ScatterJacRank1_Tag&, int const& cell)
    const
{
  // int const neq = nodeID.extent(2);
  // int const nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO col[500];
  LO row;
  ST vals[500];

  if (nunk > 500) {
    Kokkos::abort("ERROR (ScatterResidual): nunk > 500");
  }

  for (int node_col = 0; node_col < this->numNodes; node_col++) {
    for (int eq_col = 0; eq_col < neq; eq_col++) {
      col[neq * node_col + eq_col] = nodeID(cell, node_col, eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      row = nodeID(cell, node, this->offset + eq);
      if (((this->valVec)(cell, node, eq)).hasFastAccess()) {
        for (int i = 0; i < nunk; ++i) vals[i] = (this->valVec)(cell, node, eq).fastAccessDx(i);
        Jac_kokkos.sumIntoValues(row, col, nunk, vals, false, true);
      }
    }
  }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(const PHAL_ScatterResRank2_Tag&, int const& cell)
    const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t i = 0; i < numDims; i++)
      for (std::size_t j = 0; j < numDims; j++) {
        const LO id = nodeID(cell, node, this->offset + i * numDims + j);
        Kokkos::atomic_fetch_add(&f_kokkos(id), (this->valTensor(cell, node, i, j)).val());
      }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(
    const PHAL_ScatterJacRank2_Adjoint_Tag&,
    int const& cell) const
{
  // int const neq = nodeID.extent(2);
  // int const nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO col[500];
  LO row;

  if (nunk > 500) {
    Kokkos::abort("ERROR (ScatterResidual): nunk > 500");
  }

  for (int node_col = 0; node_col < this->numNodes; node_col++) {
    for (int eq_col = 0; eq_col < neq; eq_col++) {
      col[neq * node_col + eq_col] = nodeID(cell, node_col, eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      row = nodeID(cell, node, this->offset + eq);
      if (((this->valTensor)(cell, node, eq / numDims, eq % numDims)).hasFastAccess()) {
        for (int lunk = 0; lunk < nunk; lunk++) {
          ST val = ((this->valTensor)(cell, node, eq / numDims, eq % numDims)).fastAccessDx(lunk);
          Jac_kokkos.sumIntoValues(col[lunk], &row, 1, &val, false, true);
        }
      }  // has fast access
    }
  }
}

template <typename Traits>
KOKKOS_INLINE_FUNCTION void
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::operator()(const PHAL_ScatterJacRank2_Tag&, int const& cell)
    const
{
  // int const neq = nodeID.extent(2);
  // int const nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO col[500];
  LO row;
  ST vals[500];

  if (nunk > 500) {
    Kokkos::abort("ERROR (ScatterResidual): nunk > 500");
  }

  for (int node_col = 0; node_col < this->numNodes; node_col++) {
    for (int eq_col = 0; eq_col < neq; eq_col++) {
      col[neq * node_col + eq_col] = nodeID(cell, node_col, eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      row = nodeID(cell, node, this->offset + eq);
      if (((this->valTensor)(cell, node, eq / numDims, eq % numDims)).hasFastAccess()) {
        for (int i = 0; i < nunk; ++i)
          vals[i] = (this->valTensor)(cell, node, eq / numDims, eq % numDims).fastAccessDx(i);
        Jac_kokkos.sumIntoValues(row, col, nunk, vals, false, true);
      }
    }
  }
}

// **********************************************************************
template <typename Traits>
void
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(typename Traits::EvalData workset)
{
#if defined(ALBANY_TIMER)
  auto start = std::chrono::high_resolution_clock::now();
#endif
  // Get map for local data structures
  nodeID = workset.wsElNodeEqID;

  // Get dimensions
  neq  = nodeID.extent(2);
  nunk = neq * this->numNodes;

  // Get Kokkos vector view and local matrix
  bool const loadResid = Teuchos::nonnull(workset.f);
  if (loadResid) {
    f_kokkos = workset.f_kokkos;
  }
  Jac_kokkos = workset.Jac_kokkos;

  if (this->tensorRank == 0) {
    // Get MDField views from std::vector
    for (int i = 0; i < numFields; i++) val_kokkos[i] = this->val[i].get_view();

    if (loadResid) {
      Kokkos::parallel_for(PHAL_ScatterResRank0_Policy(0, workset.numCells), *this);
      cudaCheckError();
    }

    if (workset.is_adjoint) {
      Kokkos::parallel_for(PHAL_ScatterJacRank0_Adjoint_Policy(0, workset.numCells), *this);
      cudaCheckError();
    } else {
      Kokkos::parallel_for(PHAL_ScatterJacRank0_Policy(0, workset.numCells), *this);
      cudaCheckError();
    }
  } else if (this->tensorRank == 1) {
    if (loadResid) {
      Kokkos::parallel_for(PHAL_ScatterResRank1_Policy(0, workset.numCells), *this);
      cudaCheckError();
    }

    if (workset.is_adjoint) {
      Kokkos::parallel_for(PHAL_ScatterJacRank1_Adjoint_Policy(0, workset.numCells), *this);
      cudaCheckError();
    } else {
      Kokkos::parallel_for(PHAL_ScatterJacRank1_Policy(0, workset.numCells), *this);
      cudaCheckError();
    }
  } else if (this->tensorRank == 2) {
    numDims = this->valTensor.extent(2);

    if (loadResid) {
      Kokkos::parallel_for(PHAL_ScatterResRank2_Policy(0, workset.numCells), *this);
      cudaCheckError();
    }

    if (workset.is_adjoint) {
      Kokkos::parallel_for(PHAL_ScatterJacRank2_Adjoint_Policy(0, workset.numCells), *this);
    } else {
      Kokkos::parallel_for(PHAL_ScatterJacRank2_Policy(0, workset.numCells), *this);
      cudaCheckError();
    }
  }

#if defined(ALBANY_TIMER)
  PHX::Device::fence();
  auto      elapsed      = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec     = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout << "Scatter Jacobian time = " << millisec << "  " << microseconds << std::endl;
#endif
}

}  // namespace PHAL
