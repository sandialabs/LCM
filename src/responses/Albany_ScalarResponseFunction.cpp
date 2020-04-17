// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_ScalarResponseFunction.hpp"

#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"

namespace Albany {

Teuchos::RCP<Thyra_VectorSpace const>
ScalarResponseFunction::responseVectorSpace() const
{
  int num_responses = this->numResponses();
  return createLocallyReplicatedVectorSpace(num_responses, comm);
}

Teuchos::RCP<Thyra_LinearOp>
ScalarResponseFunction::createGradientOp() const
{
  ALBANY_ABORT(
      "Error!  Albany::ScalarResponseFunction::createGradientOpT():  "
      << "Operator form of dg/dx is not supported for scalar responses.");
  return Teuchos::null;
}

void
ScalarResponseFunction::evaluateDerivative(
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
  this->evaluateGradient(
      current_time,
      x,
      xdot,
      xdotdot,
      p,
      deriv_p,
      g,
      dg_dx.getMultiVector(),
      dg_dxdot.getMultiVector(),
      dg_dxdotdot.getMultiVector(),
      dg_dp.getMultiVector());
}

}  // namespace Albany
