// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
EquilibriumConcentrationBC_Base<EvalT, Traits>::EquilibriumConcentrationBC_Base(Teuchos::ParameterList& p)
    : coffset_(p.get<int>("Equation Offset")),
      poffset_(p.get<int>("Pressure Offset")),
      PHAL::DirichletBase<EvalT, Traits>(p),
      applied_conc_(p.get<RealType>("Applied Concentration")),
      pressure_fac_(p.get<RealType>("Pressure Factor"))
{
}
template <typename EvalT, typename Traits>
void
EquilibriumConcentrationBC_Base<EvalT, Traits>::computeBCs(ScalarT& pressure, ScalarT& Cval)
{
  Cval = applied_conc_ * std::exp(pressure_fac_ * pressure);
}
// Specialization: Residual
template <typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::Residual, Traits>::EquilibriumConcentrationBC(Teuchos::ParameterList& p)
    : EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}
template <typename Traits>
void
EquilibriumConcentrationBC<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Thyra_Vector const> x = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;

  Teuchos::ArrayRCP<const ST> x_constView    = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView = Albany::getNonconstLocalData(f);

  // Grab the vector of node GIDs for this Node Set ID from the std::map
  std::vector<std::vector<int>> const& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int     cunk, punk;
  ScalarT Cval;
  ScalarT pressure;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    cunk     = nsNodes[inode][this->coffset_];
    punk     = nsNodes[inode][this->poffset_];
    pressure = x_constView[punk];
    this->computeBCs(pressure, Cval);

    f_nonconstView[cunk] = x_constView[cunk] - Cval;
  }
}
// Specialization: Jacobian
template <typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::Jacobian, Traits>::EquilibriumConcentrationBC(Teuchos::ParameterList& p)
    : EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}
template <typename Traits>
void
EquilibriumConcentrationBC<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(
    typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Thyra_Vector const> x   = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f   = dirichletWorkset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;

  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  const RealType                       j_coeff = dirichletWorkset.j_coeff;
  std::vector<std::vector<int>> const& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  bool fillResid = (f != Teuchos::null);
  if (fillResid) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }

  int     cunk, punk;
  ScalarT Cval;
  ScalarT pressure;

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntries;
  Teuchos::Array<LO> matrixIndices;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    cunk     = nsNodes[inode][this->coffset_];
    punk     = nsNodes[inode][this->poffset_];
    pressure = x_constView[punk];
    this->computeBCs(pressure, Cval);

    // replace jac values for the C dof
    Albany::getLocalRowValues(jac, cunk, matrixIndices, matrixEntries);
    for (auto& val : matrixEntries) {
      val = 0.0;
    }
    Albany::setLocalRowValues(jac, cunk, matrixIndices(), matrixEntries());
    index[0] = cunk;
    Albany::setLocalRowValues(jac, cunk, index(), value());

    if (fillResid) {
      f_nonconstView[cunk] = x_constView[cunk] - Cval.val();
    }
  }
}

}  // namespace LCM
