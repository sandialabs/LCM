// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_KLResponseFunction.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_VerboseObject.hpp"

namespace Albany {

KLResponseFunction::KLResponseFunction(
    const Teuchos::RCP<Albany::AbstractResponseFunction>& response_,
    Teuchos::ParameterList&                               responseParams)
    : response(response_),
      responseParams(responseParams),
      out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  num_kl = responseParams.get("Number of KL Terms", 5);
}

void
KLResponseFunction::evaluateResponse(
    double const                            current_time,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& xdot,
    Teuchos::RCP<Thyra_Vector const> const& xdotdot,
    const Teuchos::Array<ParamVec>&         p,
    Teuchos::RCP<Thyra_Vector> const&       g)
{
  response->evaluateResponse(current_time, x, xdot, xdotdot, p, g);
}

void
KLResponseFunction::evaluateDerivative(
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
  response->evaluateDerivative(
      current_time,
      x,
      xdot,
      xdotdot,
      p,
      deriv_p,
      g,
      dg_dx,
      dg_dxdot,
      dg_dxdotdot,
      dg_dp);
}

}  // namespace Albany
