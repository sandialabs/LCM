// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace LCM {

template <typename EvalT, typename Traits>
TorsionBC_Base<EvalT, Traits>::TorsionBC_Base(Teuchos::ParameterList& p)
    : PHAL::DirichletBase<EvalT, Traits>(p),
      thetaDot(p.get<RealType>("Theta Dot")),
      X0(p.get<RealType>("X0")),
      Y0(p.get<RealType>("Y0"))
{
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
TorsionBC_Base<EvalT, Traits>::computeBCs(double* coord, ScalarT& Xval, ScalarT& Yval, const RealType time)
{
  RealType X(coord[0]);
  RealType Y(coord[1]);
  RealType theta(thetaDot * time);

  // compute displace Xval and Yval. (X0,Y0) is the center of rotation/torsion
  Xval = X0 + (X - X0) * std::cos(theta) - (Y - Y0) * std::sin(theta) - X;
  Yval = Y0 + (X - X0) * std::sin(theta) + (Y - Y0) * std::cos(theta) - Y;
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template <typename Traits>
TorsionBC<PHAL::AlbanyTraits::Residual, Traits>::TorsionBC(Teuchos::ParameterList& p)
    : TorsionBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

// **********************************************************************
template <typename Traits>
void
TorsionBC<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Thyra_Vector const> x = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;

  Teuchos::ArrayRCP<const ST> x_constView    = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView = Albany::getNonconstLocalData(f);

  // Grab the vector off node GIDs for this Node Set ID from the std::map
  std::vector<std::vector<int>> const& nsNodes      = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  std::vector<double*> const&          nsNodeCoords = dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;

  int     xlunk, ylunk;  // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    f_nonconstView[xlunk] = x_constView[xlunk] - Xval;
    f_nonconstView[ylunk] = x_constView[ylunk] - Yval;

    // Record DOFs to avoid setting Schwarz BCs on them.
    dirichletWorkset.fixed_dofs_.insert(xlunk);
    dirichletWorkset.fixed_dofs_.insert(ylunk);
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template <typename Traits>
TorsionBC<PHAL::AlbanyTraits::Jacobian, Traits>::TorsionBC(Teuchos::ParameterList& p)
    : TorsionBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}
// **********************************************************************
template <typename Traits>
void
TorsionBC<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Thyra_Vector const> x   = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f   = dirichletWorkset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;

  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  const RealType                       j_coeff      = dirichletWorkset.j_coeff;
  std::vector<std::vector<int>> const& nsNodes      = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  std::vector<double*> const&          nsNodeCoords = dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  bool fillResid = (f != Teuchos::null);
  if (fillResid) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }

  RealType time = dirichletWorkset.current_time;

  int     xlunk, ylunk;  // local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  size_t             numEntries;
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntries;
  Teuchos::Array<LO> matrixIndices;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    // replace jac values for the X dof
    Albany::getLocalRowValues(jac, xlunk, matrixIndices, matrixEntries);
    for (auto& val : matrixEntries) {
      val = 0.0;
    }
    Albany::setLocalRowValues(jac, xlunk, matrixIndices(), matrixEntries());
    index[0] = xlunk;
    Albany::setLocalRowValues(jac, xlunk, index(), value());

    // replace jac values for the y dof
    Albany::getLocalRowValues(jac, ylunk, matrixIndices, matrixEntries);
    for (auto& val : matrixEntries) {
      val = 0.0;
    }
    Albany::setLocalRowValues(jac, ylunk, matrixIndices(), matrixEntries());
    index[0] = ylunk;
    Albany::setLocalRowValues(jac, ylunk, index(), value());

    if (fillResid) {
      f_nonconstView[xlunk] = x_constView[xlunk] - Xval.val();
      f_nonconstView[ylunk] = x_constView[ylunk] - Yval.val();
    }
  }
}

}  // namespace LCM
