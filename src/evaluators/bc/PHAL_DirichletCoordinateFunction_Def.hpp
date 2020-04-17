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

namespace PHAL {

template <typename EvalT, typename Traits /*, typename cfunc_traits*/>
DirichletCoordFunction_Base<EvalT, Traits /*, cfunc_traits*/>::
    DirichletCoordFunction_Base(Teuchos::ParameterList& p)
    : PHAL::DirichletBase<EvalT, Traits>(p), func(p)
{
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template <typename Traits /*, typename cfunc_traits*/>
DirichletCoordFunction<
    PHAL::AlbanyTraits::Residual,
    Traits /*, cfunc_traits*/>::DirichletCoordFunction(Teuchos::ParameterList&
                                                           p)
    : DirichletCoordFunction_Base<
          PHAL::AlbanyTraits::Residual,
          Traits /*, cfunc_traits*/>(p)
{
}

// **********************************************************************
template <typename Traits /*, typename cfunc_traits*/>
void
DirichletCoordFunction<
    PHAL::AlbanyTraits::Residual,
    Traits /*, cfunc_traits*/>::evaluateFields(typename Traits::EvalData
                                                   dirichletWorkset)
{
  Teuchos::RCP<Thyra_Vector const> x = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;

  Teuchos::ArrayRCP<const ST> x_constView    = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView = Albany::getNonconstLocalData(f);

  // Grab the vector off node GIDs for this Node Set ID from the std::map
  std::vector<std::vector<int>> const& nsNodes =
      dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  std::vector<double*> const& nsNodeCoords =
      dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time                 = dirichletWorkset.current_time;
  int      number_of_components = this->func.getNumComponents();

  double*              coord;
  std::vector<ScalarT> BCVals(number_of_components);

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for (unsigned int j = 0; j < number_of_components; j++) {
      int offset             = nsNodes[inode][j];
      f_nonconstView[offset] = (x_constView[offset] - BCVals[j]);
    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template <typename Traits /*, typename cfunc_traits*/>
DirichletCoordFunction<
    PHAL::AlbanyTraits::Jacobian,
    Traits /*, cfunc_traits*/>::DirichletCoordFunction(Teuchos::ParameterList&
                                                           p)
    : DirichletCoordFunction_Base<
          PHAL::AlbanyTraits::Jacobian,
          Traits /*, cfunc_traits*/>(p)
{
}
// **********************************************************************
template <typename Traits /*, typename cfunc_traits*/>
void
DirichletCoordFunction<
    PHAL::AlbanyTraits::Jacobian,
    Traits /*, cfunc_traits*/>::evaluateFields(typename Traits::EvalData
                                                   dirichletWorkset)
{
  Teuchos::RCP<Thyra_Vector const> x   = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f   = dirichletWorkset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  const RealType j_coeff = dirichletWorkset.j_coeff;

  std::vector<std::vector<int>> const& nsNodes =
      dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  std::vector<double*> const& nsNodeCoords =
      dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntries;
  Teuchos::Array<LO> matrixIndices;

  bool fillResid = (f != Teuchos::null);
  if (fillResid) {
    x_constView    = Albany::getLocalData(x);
    f_nonconstView = Albany::getNonconstLocalData(f);
  }

  RealType time                 = dirichletWorkset.current_time;
  int      number_of_components = this->func.getNumComponents();

  double*              coord;
  std::vector<ScalarT> BCVals(number_of_components);

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for (unsigned int j = 0; j < number_of_components; j++) {
      int offset = nsNodes[inode][j];
      index[0]   = offset;

      // Extract the row, zero it out, then put j_coeff on diagonal
      Albany::getLocalRowValues(jac, offset, matrixIndices, matrixEntries);
      for (auto& val : matrixEntries) { val = 0.0; }
      Albany::setLocalRowValues(jac, offset, matrixIndices(), matrixEntries());
      Albany::setLocalRowValues(jac, offset, index(), value());

      if (fillResid) {
        f_nonconstView[offset] = (x_constView[offset] - BCVals[j].val());
      }
    }
  }
}

}  // namespace PHAL
