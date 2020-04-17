// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_SolutionTwoNormResponseFunction.hpp"

#include "Thyra_VectorStdOps.hpp"

Albany::SolutionTwoNormResponseFunction::SolutionTwoNormResponseFunction(
    const Teuchos::RCP<Teuchos_Comm const>& commT)
    : SamplingBasedScalarResponseFunction(commT)
{
}

Albany::SolutionTwoNormResponseFunction::~SolutionTwoNormResponseFunction() {}

unsigned int
Albany::SolutionTwoNormResponseFunction::numResponses() const
{
  return 1;
}

void
Albany::SolutionTwoNormResponseFunction::evaluateResponse(
    double const /*current_time*/,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& /*xdot*/,
    Teuchos::RCP<Thyra_Vector const> const& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*p*/,
    Teuchos::RCP<Thyra_Vector> const& g)
{
  Teuchos::ScalarTraits<ST>::magnitudeType twonorm = x->norm_2();
  g->assign(twonorm);
}

void
Albany::SolutionTwoNormResponseFunction::evaluateGradient(
    double const /*current_time*/,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& /*xdot*/,
    Teuchos::RCP<Thyra_Vector const> const& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*p*/,
    ParamVec* /*deriv_p*/,
    Teuchos::RCP<Thyra_Vector> const&      g,
    Teuchos::RCP<Thyra_MultiVector> const& dg_dx,
    Teuchos::RCP<Thyra_MultiVector> const& dg_dxdot,
    Teuchos::RCP<Thyra_MultiVector> const& dg_dxdotdot,
    Teuchos::RCP<Thyra_MultiVector> const& dg_dp)
{
  Teuchos::ScalarTraits<ST>::magnitudeType nrm = x->norm_2();

  // Evaluate response g
  if (!g.is_null()) { g->assign(nrm); }

  // Evaluate dg/dx
  if (!dg_dx.is_null()) {
    // V_StV stands for V_out = Scalar * V_in
    Thyra::V_StV(dg_dx->col(0).ptr(), 1.0 / nrm, *x);
  }

  // Evaluate dg/dxdot
  if (!dg_dxdot.is_null()) { dg_dxdot->assign(0.0); }

  // Evaluate dg/dxdot
  if (!dg_dxdotdot.is_null()) { dg_dxdotdot->assign(0.0); }

  // Evaluate dg/dp
  if (!dg_dp.is_null()) { dg_dp->assign(0.0); }
}
