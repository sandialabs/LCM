// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_DistributedResponseFunction.hpp"

void
Albany::DistributedResponseFunction::evaluateDerivative(
    double const                                     current_time,
    Teuchos::RCP<Thyra_Vector const> const&          x,
    Teuchos::RCP<Thyra_Vector const> const&          xdot,
    Teuchos::RCP<Thyra_Vector const> const&          xdotdot,
    const Teuchos::Array<ParamVec>&                  p,
    ParamVec*                                        deriv_p,
    Teuchos::RCP<Thyra_Vector> const&                g,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp)
{
  // Get the stored operator pointers (note: they may be null)
  const Teuchos::RCP<Thyra_LinearOp> dg_dx_op       = dg_dx.getLinearOp();
  const Teuchos::RCP<Thyra_LinearOp> dg_dxdot_op    = dg_dxdot.getLinearOp();
  const Teuchos::RCP<Thyra_LinearOp> dg_dxdotdot_op = dg_dxdotdot.getLinearOp();

  // Get the stored multivector pointer (note: it may be null)
  Teuchos::RCP<Thyra_MultiVector> const& dg_dp_mv = dg_dp.getMultiVector();

  this->evaluateGradient(
      current_time,
      x,
      xdot,
      xdotdot,
      p,
      deriv_p,
      g,
      dg_dx_op,
      dg_dxdot_op,
      dg_dxdotdot_op,
      dg_dp_mv);
}
