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
KfieldBC_Base<EvalT, Traits>::KfieldBC_Base(Teuchos::ParameterList& p)
    : offset(p.get<int>("Equation Offset")),
      PHAL::DirichletBase<EvalT, Traits>(p),
      mu(p.get<RealType>("Shear Modulus")),
      nu(p.get<RealType>("Poissons Ratio"))
{
  KIval  = p.get<RealType>("KI Value");
  KIIval = p.get<RealType>("KII Value");

  KI  = KIval;
  KII = KIIval;

  KI_name  = p.get<std::string>("Kfield KI Name");
  KII_name = p.get<std::string>("Kfield KII Name");

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  this->registerSacadoParameter(KI_name, paramLib);
  this->registerSacadoParameter(KII_name, paramLib);

  timeValues = p.get<Teuchos::Array<RealType>>("Time Values").toVector();
  KIValues   = p.get<Teuchos::Array<RealType>>("KI Values").toVector();
  KIIValues  = p.get<Teuchos::Array<RealType>>("KII Values").toVector();

  ALBANY_PANIC(!(timeValues.size() == KIValues.size()), "Dimension of \"Time Values\" and \"KI Values\" do not match");

  ALBANY_PANIC(
      !(timeValues.size() == KIIValues.size()), "Dimension of \"Time Values\" and \"KII Values\" do not match");
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename KfieldBC_Base<EvalT, Traits>::ScalarT&
KfieldBC_Base<EvalT, Traits>::getValue(std::string const& n)
{
  if (n == KI_name) return KI;
  // else if (n== timeValues)
  //        return timeValues;
  else
    return KII;
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
KfieldBC_Base<EvalT, Traits>::computeBCs(double* coord, ScalarT& Xval, ScalarT& Yval, RealType time)
{
  ALBANY_PANIC(time > timeValues.back(), "Time is growing unbounded!");

  RealType X, Y, R, theta;
  ScalarT  coeff_1, coeff_2;
  RealType tau = 6.283185307179586;
  ScalarT  KI_X, KI_Y, KII_X, KII_Y;

  X     = coord[0];
  Y     = coord[1];
  R     = std::sqrt(X * X + Y * Y);
  theta = std::atan2(Y, X);

  ScalarT      KIFunctionVal, KIIFunctionVal;
  RealType     KIslope, KIIslope;
  unsigned int Index(0);

  while (timeValues[Index] < time) Index++;

  if (Index == 0) {
    KIFunctionVal  = KIValues[Index];
    KIIFunctionVal = KIIValues[Index];
  } else {
    KIslope       = (KIValues[Index] - KIValues[Index - 1]) / (timeValues[Index] - timeValues[Index - 1]);
    KIFunctionVal = KIValues[Index - 1] + KIslope * (time - timeValues[Index - 1]);

    KIIslope       = (KIIValues[Index] - KIIValues[Index - 1]) / (timeValues[Index] - timeValues[Index - 1]);
    KIIFunctionVal = KIIValues[Index - 1] + KIIslope * (time - timeValues[Index - 1]);
  }

  coeff_1 = (KI * KIFunctionVal / mu) * std::sqrt(R / tau);
  coeff_2 = (KII * KIIFunctionVal / mu) * std::sqrt(R / tau);

  KI_X = coeff_1 * (1.0 - 2.0 * nu + std::sin(theta / 2.0) * std::sin(theta / 2.0)) * std::cos(theta / 2.0);
  KI_Y = coeff_1 * (2.0 - 2.0 * nu - std::cos(theta / 2.0) * std::cos(theta / 2.0)) * std::sin(theta / 2.0);

  KII_X = coeff_2 * (2.0 - 2.0 * nu + std::cos(theta / 2.0) * std::cos(theta / 2.0)) * std::sin(theta / 2.0);
  KII_Y = coeff_2 * (-1.0 + 2.0 * nu + std::sin(theta / 2.0) * std::sin(theta / 2.0)) * std::cos(theta / 2.0);

  Xval = KI_X + KII_X;
  Yval = KI_Y + KII_Y;
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template <typename Traits>
KfieldBC<PHAL::AlbanyTraits::Residual, Traits>::KfieldBC(Teuchos::ParameterList& p)
    : KfieldBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

// **********************************************************************
template <typename Traits>
void
KfieldBC<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(typename Traits::EvalData dirichletWorkset)
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
KfieldBC<PHAL::AlbanyTraits::Jacobian, Traits>::KfieldBC(Teuchos::ParameterList& p)
    : KfieldBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}
// **********************************************************************
template <typename Traits>
void
KfieldBC<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Thyra_Vector const> x   = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f   = dirichletWorkset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;

  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  RealType time = dirichletWorkset.current_time;

  const RealType                       j_coeff      = dirichletWorkset.j_coeff;
  std::vector<std::vector<int>> const& nsNodes      = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  std::vector<double*> const&          nsNodeCoords = dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  bool fillResid = (f != Teuchos::null);
  if (fillResid) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }

  int     xlunk, ylunk;  // local indicies into unknown vector
  double* coord;

  ScalarT            Xval, Yval;
  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
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
