// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Schwarz_CoupledJacobian.hpp"

#include "Albany_Macros.hpp"
#include "Albany_Utils.hpp"
#include "Teuchos_ParameterListExceptions.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Thyra_DefaultBlockedLinearOp.hpp"

using Thyra::PhysicallyBlockedLinearOpBase;

namespace LCM {

Schwarz_CoupledJacobian::Schwarz_CoupledJacobian(
    Teuchos::RCP<Teuchos_Comm const> const& comm)
{
  comm_ = comm;
}

Schwarz_CoupledJacobian::~Schwarz_CoupledJacobian() { return; }

//#define USE_OFF_DIAGONAL
//#define EXPLICIT_OFF_DIAGONAL

// getThyraCoupledJacobian method is similar to getThyraMatrix in panzer
//(Panzer_BlockedTpetraLinearObjFactory_impl.hpp).
Teuchos::RCP<Thyra::LinearOpBase<ST>>
Schwarz_CoupledJacobian::getThyraCoupledJacobian(
    Teuchos::Array<Teuchos::RCP<Thyra_LinearOp>>                jacs,
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> const& ca) const
{
  auto const block_dim = jacs.size();

  // get the block dimension
  // this operator will be square
  Teuchos::RCP<Thyra::PhysicallyBlockedLinearOpBase<ST>> blocked_op =
      Thyra::defaultBlockedLinearOp<ST>();

  blocked_op->beginBlockFill(block_dim, block_dim);

  // loop over each block
  for (std::size_t i = 0; i < block_dim; i++) {
    for (std::size_t j = 0; j < block_dim; j++) {
      // build (i,j) block matrix and add it to blocked operator
      if (i == j) {  // Diagonal blocks
        blocked_op->setNonconstBlock(i, j, jacs[i]);
      } else {  // Off-diagonal blocks
#if defined(USE_OFF_DIAGONAL)
#if defined(EXPLICIT_OFF_DIAGONAL)

        Teuchos::RCP<Schwarz_BoundaryJacobian> jac_boundary =
            Teuchos::rcp(new Schwarz_BoundaryJacobian(comm_, ca, jacs, i, j));

        Teuchos::RCP<Thyra_LinearOp> exp_jac =
            jac_boundary->getExplicitOperator();

        Teuchos::RCP<Thyra::LinearOpBase<ST>> block = exp_jac;

#else

        Teuchos::RCP<Thyra_LinearOp> jac_boundary =
            Teuchos::rcp(new Schwarz_BoundaryJacobian(comm_, ca, jacs, i, j));

        Teuchos::RCP<Thyra::LinearOpBase<ST>> block = jac_boundary;

#endif  // EXPLICIT_OFF_DIAGONAL

        blocked_op->setNonconstBlock(i, j, block);
#endif  // USE_OFF_DIAGONAL
      }
    }
  }

  // all done
  blocked_op->endBlockFill();
  return blocked_op;
}

}  // namespace LCM
