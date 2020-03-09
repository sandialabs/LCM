//
// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
//

#include "Albany_FieldManagerResidualOnlyResponseFunction.hpp"

Albany::FieldManagerResidualOnlyResponseFunction::
    FieldManagerResidualOnlyResponseFunction(
        const Teuchos::RCP<Albany::Application>&     application_,
        const Teuchos::RCP<Albany::AbstractProblem>& problem_,
        const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs_,
        const Teuchos::RCP<Albany::StateManager>&    stateMgr_,
        Teuchos::ParameterList&                      responseParams)
    : FieldManagerScalarResponseFunction(
          application_,
          problem_,
          meshSpecs_,
          stateMgr_,
          responseParams)
{
}

void
Albany::FieldManagerResidualOnlyResponseFunction::evaluateGradient(
    double const                            current_time,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& xdot,
    Teuchos::RCP<Thyra_Vector const> const& xdotdot,
    const Teuchos::Array<ParamVec>&         p,
    ParamVec* /*deriv_p*/,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& /*dg_dx*/,
    const Teuchos::RCP<Thyra_MultiVector>& /*dg_dxdot*/,
    const Teuchos::RCP<Thyra_MultiVector>& /*dg_dxdotdot*/,
    const Teuchos::RCP<Thyra_MultiVector>& /*dg_dp*/)
{
  if (!g.is_null()) {
    this->evaluateResponse(current_time, x, xdot, xdotdot, p, g);
  }
}
