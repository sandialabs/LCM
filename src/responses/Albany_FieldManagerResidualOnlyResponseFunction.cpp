// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_FieldManagerResidualOnlyResponseFunction.hpp"

Albany::FieldManagerResidualOnlyResponseFunction::FieldManagerResidualOnlyResponseFunction(
    const Teuchos::RCP<Albany::Application>&     application_,
    const Teuchos::RCP<Albany::AbstractProblem>& problem_,
    const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs_,
    const Teuchos::RCP<Albany::StateManager>&    stateMgr_,
    Teuchos::ParameterList&                      responseParams)
    : FieldManagerScalarResponseFunction(application_, problem_, meshSpecs_, stateMgr_, responseParams)
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
    Teuchos::RCP<Thyra_Vector> const& g,
    Teuchos::RCP<Thyra_MultiVector> const& /*dg_dx*/,
    Teuchos::RCP<Thyra_MultiVector> const& /*dg_dxdot*/,
    Teuchos::RCP<Thyra_MultiVector> const& /*dg_dxdotdot*/,
    Teuchos::RCP<Thyra_MultiVector> const& /*dg_dp*/)
{
  if (!g.is_null()) {
    this->evaluateResponse(current_time, x, xdot, xdotdot, p, g);
  }
}
