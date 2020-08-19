// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <set>

#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace PHAL {

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template <typename Traits>
DirichletOffNodeSet<PHAL::AlbanyTraits::Residual, Traits>::DirichletOffNodeSet(Teuchos::ParameterList& p)
    : DirichletBase<PHAL::AlbanyTraits::Residual, Traits>(p),
      nodeSets(*p.get<Teuchos::RCP<std::vector<std::string>>>("Node Sets"))
{
}

// **********************************************************************
template <typename Traits>
void
DirichletOffNodeSet<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  // Gather all node IDs from all the stored nodesets
  std::set<int> nodeSetsRows;
  for (int ins(0); ins < nodeSets.size(); ++ins) {
    std::vector<std::vector<int>> const& nsNodes = dirichletWorkset.nodeSets->find(nodeSets[ins])->second;
    for (int inode = 0; inode < nsNodes.size(); ++inode) {
      nodeSetsRows.insert(nsNodes[inode][this->offset]);
    }
  }

  Teuchos::RCP<Thyra_Vector const> x = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;

  Teuchos::ArrayRCP<const ST> x_constView    = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView = Albany::getNonconstLocalData(f);

  // Loop on all local dofs and set the BC on those not in nodeSetsRows
  LO num_local_dofs = Albany::getSpmdVectorSpace(f->space())->localSubDim();
  for (LO row = 0; row < num_local_dofs; ++row) {
    if (nodeSetsRows.find(row) == nodeSetsRows.end()) {
      f_nonconstView[row] = x_constView[row] - this->value;
    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template <typename Traits>
DirichletOffNodeSet<PHAL::AlbanyTraits::Jacobian, Traits>::DirichletOffNodeSet(Teuchos::ParameterList& p)
    : DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>(p),
      nodeSets(*p.get<Teuchos::RCP<std::vector<std::string>>>("Node Sets"))
{
}

// **********************************************************************
template <typename Traits>
void
DirichletOffNodeSet<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  // Gather all node IDs from all the stored nodesets
  std::set<int> nodeSetsRows;
  for (int ins(0); ins < nodeSets.size(); ++ins) {
    std::vector<std::vector<int>> const& nsNodes = dirichletWorkset.nodeSets->find(nodeSets[ins])->second;
    for (int inode = 0; inode < nsNodes.size(); ++inode) {
      nodeSetsRows.insert(nsNodes[inode][this->offset]);
    }
  }

  Teuchos::RCP<Thyra_Vector const> x   = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f   = dirichletWorkset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  const RealType j_coeff   = dirichletWorkset.j_coeff;
  bool           fillResid = (f != Teuchos::null);

  if (fillResid) {
    x_constView    = Albany::getLocalData(x);
    f_nonconstView = Albany::getNonconstLocalData(f);
  }

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntries;
  Teuchos::Array<LO> matrixIndices;

  // Loop on all local dofs and set the BC on those not in nodeSetsRows
  LO num_local_dofs = Albany::getSpmdVectorSpace(jac->range())->localSubDim();
  for (LO row = 0; row < num_local_dofs; ++row) {
    if (nodeSetsRows.find(row) == nodeSetsRows.end()) {
      // It's a row not on the given node sets
      index[0] = row;

      // Extract the row, zero it out, then put j_coeff on diagonal
      Albany::getLocalRowValues(jac, row, matrixIndices, matrixEntries);
      for (auto& val : matrixEntries) {
        val = 0.0;
      }
      Albany::setLocalRowValues(jac, row, matrixIndices(), matrixEntries());
      Albany::setLocalRowValues(jac, row, index(), value());

      if (fillResid) {
        f_nonconstView[row] = x_constView[row] - this->value.val();
      }
    }
  }
}

}  // Namespace PHAL
