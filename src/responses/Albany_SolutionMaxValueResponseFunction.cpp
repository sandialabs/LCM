// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_SolutionMaxValueResponseFunction.hpp"

#include <limits>

#include "Albany_ThyraUtils.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Thyra_SpmdVectorBase.hpp"

namespace Albany {

SolutionMaxValueResponseFunction::SolutionMaxValueResponseFunction(
    const Teuchos::RCP<Teuchos_Comm const>& comm,
    int                                     neq_,
    int                                     eq_,
    bool                                    interleavedOrdering_)
    : SamplingBasedScalarResponseFunction(comm),
      neq(neq_),
      eq(eq_),
      comm_(comm),
      interleavedOrdering(interleavedOrdering_)
{
  // Nothing to be done here
}

void
SolutionMaxValueResponseFunction::evaluateResponse(
    double const /*current_time*/,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& /*xdot*/,
    Teuchos::RCP<Thyra_Vector const> const& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*p*/,
    Teuchos::RCP<Thyra_Vector> const& g)
{
  Teuchos::ArrayRCP<ST> g_nonconstView = getNonconstLocalData(g);
  computeMaxValue(x, g_nonconstView[0]);
}

void
SolutionMaxValueResponseFunction::evaluateGradient(
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
  ST max_val;
  computeMaxValue(x, max_val);

  // Evaluate response g
  if (!g.is_null()) {
    Teuchos::ArrayRCP<ST> g_nonconstView = getNonconstLocalData(g);
    g_nonconstView[0]                    = max_val;
  }

  // Evaluate dg/dx
  if (!dg_dx.is_null()) {
    Teuchos::ArrayRCP<const ST> x_constView = getLocalData(x);
    Teuchos::ArrayRCP<ST>       dg_dx_nonconstView =
        getNonconstLocalData(dg_dx->col(0));
    for (int i = 0; i < x_constView.size(); ++i) {
      if (x_constView[i] == max_val) {
        dg_dx_nonconstView[i] = 1.0;
      } else {
        dg_dx_nonconstView[i] = 0.0;
      }
    }
  }

  // Evaluate dg/dxdot
  if (!dg_dxdot.is_null()) { dg_dxdot->assign(0.0); }

  // Evaluate dg/dxdotdot
  if (!dg_dxdotdot.is_null()) { dg_dxdotdot->assign(0.0); }

  // Evaluate dg/dp
  if (!dg_dp.is_null()) { dg_dp->assign(0.0); }
}

void
SolutionMaxValueResponseFunction::computeMaxValue(
    Teuchos::RCP<Thyra_Vector const> const& x,
    ST&                                     global_max)
{
  auto x_local = getLocalData(x);

  // Loop over nodes to find max value for equation eq
  int num_my_nodes = x_local.size() / neq;
  int index;
  ST  my_max = std::numeric_limits<ST>::lowest();
  for (int node = 0; node < num_my_nodes; node++) {
    if (interleavedOrdering) {
      index = node * neq + eq;
    } else {
      index = node + eq * num_my_nodes;
    }
    if (x_local[index] > my_max) { my_max = x_local[index]; }
  }

  // Check remainder (AGS: NOT SURE HOW THIS CODE GETS CALLED?)
  // LB: I believe this code would get called if equations at a given node are
  // not
  //     forced to be on the same process, in which case neq may not divide the
  //     local dimension. I also believe Albany makes sure this does not happen,
  //     so I *think* these lines *should* be safe to remove...
  if (num_my_nodes * neq + eq < x_local.size()) {
    if (interleavedOrdering) {
      index = num_my_nodes * neq + eq;
    } else {
      index = num_my_nodes + eq * num_my_nodes;
    }
    if (x_local[index] > my_max) { my_max = x_local[index]; }
  }

  // Get max value across all proc's
  Teuchos::reduceAll(*comm_, Teuchos::REDUCE_MAX, 1, &my_max, &global_max);
}

}  // namespace Albany
