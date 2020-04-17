// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "AAdapt_RC_Projector_impl.hpp"

#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_Utils.hpp"
#include "BelosBlockCGSolMgr.hpp"
#include "BelosThyraAdapter.hpp"
#include "BelosTpetraAdapter.hpp"
#include "Ifpack2_RILUK.hpp"

namespace AAdapt {
namespace rc {

Teuchos::RCP<Thyra_MultiVector>
solve(
    const Teuchos::RCP<const Thyra_LinearOp>&    A,
    Teuchos::RCP<Thyra_LinearOp>&                P,
    const Teuchos::RCP<const Thyra_MultiVector>& b,
    Teuchos::ParameterList&                      pl)
{
  int const                       nrhs = b->domain()->dim();
  Teuchos::RCP<Thyra_MultiVector> x = Thyra::createMembers(A->domain(), nrhs);

  if (P.is_null()) {
    Teuchos::ParameterList pl_;
    pl_.set<int>("fact: iluk level-of-fill", 0);
    Teuchos::RCP<Ifpack2::RILUK<Tpetra_RowMatrix>> prec;
    Teuchos::RCP<const Tpetra_CrsMatrix> tA = Albany::getConstTpetraMatrix(A);
    prec = Teuchos::rcp(new Ifpack2::RILUK<Tpetra_RowMatrix>(tA));
    prec->setParameters(pl_);
    prec->initialize();
    prec->compute();
    P = Albany::createThyraLinearOp(
        Teuchos::rcp_implicit_cast<Tpetra_Operator>(prec));
  }

  typedef Thyra_MultiVector MV;
  typedef Thyra_LinearOp    Op;

  typedef Belos::LinearProblem<RealType, MV, Op> LinearProblem;
  Teuchos::RCP<LinearProblem>                    problem =
      Teuchos::rcp(new LinearProblem(A, x, b));
  problem->setRightPrec(P);
  problem->setProblem();

  Belos::BlockCGSolMgr<RealType, MV, Op> solver(
      problem, Teuchos::rcp(&pl, false));
  solver.solve();

  return x;
}

}  // namespace rc
}  // namespace AAdapt
