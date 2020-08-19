// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_SolutionResponseFunction.hpp"

#include "Albany_Application.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_ThyraUtils.hpp"

namespace Albany {

SolutionResponseFunction::SolutionResponseFunction(
    const Teuchos::RCP<Albany::Application>& application,
    Teuchos::ParameterList const&            responseParams)
    : solution_vs(getSpmdVectorSpace(application->getVectorSpace()))
{
  // Build list of DOFs we want to keep
  // This should be replaced by DOF names eventually
  int numDOF = application->getProblem()->numEquations();
  if (responseParams.isType<Teuchos::Array<int>>("Keep DOF Indices")) {
    Teuchos::Array<int> dofs = responseParams.get<Teuchos::Array<int>>("Keep DOF Indices");
    keepDOF.resize(numDOF, false);
    numKeepDOF = 0;
    for (int i = 0; i < dofs.size(); i++) {
      keepDOF[dofs[i]] = true;
      ++numKeepDOF;
    }
  } else {
    keepDOF.resize(numDOF, true);
    numKeepDOF = numDOF;
  }
}

void
Albany::SolutionResponseFunction::setup()
{
  // Build culled vs
  int Neqns = keepDOF.size();
  int N     = solution_vs->localSubDim();

  TEUCHOS_ASSERT(!(N % Neqns));  // Assume that all the equations for
                                 // a given node are on the assigned
                                 // processor. I.e. need to ensure
                                 // that N is exactly Neqns-divisible

  int nnodes = N / Neqns;            // number of fem nodes
  int N_new  = nnodes * numKeepDOF;  // length of local x_new

  Teuchos::Array<LO> subspace_components(N_new);
  for (int ieqn = 0, idx = 0; ieqn < Neqns; ++ieqn) {
    if (keepDOF[ieqn]) {
      for (int inode = 0; inode < nnodes; ++inode, ++idx) {
        subspace_components[idx] = inode * Neqns + ieqn;
      }
    }
  }
  culled_vs = getSpmdVectorSpace(createSubspace(solution_vs, subspace_components));

  // Create graph for gradient operator -- diagonal matrix
  cull_op_factory        = Teuchos::rcp(new ThyraCrsMatrixFactory(solution_vs, culled_vs, 1));
  auto culled_vs_indexer = createGlobalLocalIndexer(culled_vs);
  for (int i = 0; i < culled_vs->localSubDim(); i++) {
    const GO row = culled_vs_indexer->getGlobalElement(i);
    cull_op_factory->insertGlobalIndices(row, Teuchos::arrayView(&row, 1));
  }
  cull_op_factory->fillComplete();

  // Create the culling operator
  cull_op = cull_op_factory->createOp();
  assign(cull_op, 1.0);
  fillComplete(cull_op);
}

Teuchos::RCP<Thyra_LinearOp>
SolutionResponseFunction::createGradientOp() const
{
  auto gradOp = cull_op_factory->createOp();
  fillComplete(gradOp);
  return gradOp;
}

void
SolutionResponseFunction::evaluateResponse(
    double const /*current_time*/,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& /*xdot*/,
    Teuchos::RCP<Thyra_Vector const> const& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*p*/,
    Teuchos::RCP<Thyra_Vector> const& g)
{
  cullSolution(x, g);
}

void
SolutionResponseFunction::evaluateGradient(
    double const /*current_time*/,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& /*xdot*/,
    Teuchos::RCP<Thyra_Vector const> const& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*p*/,
    ParamVec* /*deriv_p*/,
    Teuchos::RCP<Thyra_Vector> const&      g,
    const Teuchos::RCP<Thyra_LinearOp>&    dg_dx,
    const Teuchos::RCP<Thyra_LinearOp>&    dg_dxdot,
    const Teuchos::RCP<Thyra_LinearOp>&    dg_dxdotdot,
    Teuchos::RCP<Thyra_MultiVector> const& dg_dp)
{
  if (!g.is_null()) {
    cullSolution(x, g);
  }

  if (!dg_dx.is_null()) {
    assign(dg_dx, 1.0);  // matrix only stores the diagonal
  }

  if (!dg_dxdot.is_null()) {
    assign(dg_dxdot, 0.0);  // matrix only stores the diagonal
  }

  if (!dg_dxdotdot.is_null()) {
    assign(dg_dxdotdot, 0.0);  // matrix only stores the diagonal
  }

  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}

void
SolutionResponseFunction::cullSolution(
    const Teuchos::RCP<const Thyra_MultiVector>& x,
    Teuchos::RCP<Thyra_MultiVector> const&       x_culled) const
{
  cull_op->apply(Thyra::EOpTransp::NOTRANS, *x, x_culled.ptr(), 1.0, 0.0);
}

}  // namespace Albany
