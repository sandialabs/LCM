// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "SolutionSniffer.hpp"

#include "Albany_Application.hpp"
#include "LOCA_MultiContinuation_ExtendedVector.H"
#include "NOX_Abstract_Group.H"
#include "NOX_Solver_Generic.H"
#include "NOX_Thyra_Vector.H"
#include "Teuchos_VerboseObject.hpp"

namespace LCM {

SolutionSniffer::SolutionSniffer() {}

SolutionSniffer::~SolutionSniffer() {}

void
SolutionSniffer::setApplication(Teuchos::RCP<Albany::Application> const& app)
{
  app_ = app;
}

void
SolutionSniffer::runPreIterate(NOX::Solver::Generic const&)
{
}

void
SolutionSniffer::runPostIterate(NOX::Solver::Generic const&)
{
}

void
SolutionSniffer::runPreSolve(NOX::Solver::Generic const& solver)
{
  if (status_test_.is_null() == false) {
    status_test_->status_         = NOX::StatusTest::Unevaluated;
    status_test_->status_message_ = "";
  }

  // Linear-elastic warmstart: at the entry of every NOX solve (one per LOCA
  // step, one per Tempus stage's nonlinear stage solve), advance the
  // predictor from x_n_converged to the linear-elastic equilibrium for the
  // BCs at the new t/λ. The first nonlinear material evaluation then sees
  // a smooth elastic field instead of the BC-injection step jump that
  // otherwise spikes path-dependent state in CrystalPlasticity slip-rate
  // Newton, Ortiz-Pandolfi δ_max, and the SDirichlet/Tempus pre-warm.
  //
  // Mutate via const_cast: the NOX framework treats the group as const
  // through this observer interface, but the contract of runPreSolve is
  // explicitly that the observer may massage the initial iterate.
  // Warmstart is opt-in via the YAML Problem-list bool "Warmstart Predictor"
  // (default false). The kernel is correct linear-elastic equilibrium, but
  // it produces a warmer initial iterate that can push some nonlinear
  // solvers (notably Block CG on non-SPD cohesive tangents, or NOX line
  // searches tuned to the dfm-style cold start) into different convergence
  // paths from the gold trajectory. Restoring a previously-removed test
  // means turning the flag on for that YAML and validating numerics —
  // hence opt-in, not always-on.
  bool const warmstart_on =
      Teuchos::nonnull(app_) && Teuchos::nonnull(app_->getProblemPL()) && app_->getProblemPL()->isParameter("Warmstart Predictor") && app_->getProblemPL()->get<bool>("Warmstart Predictor");
  if (warmstart_on) {
    // LOCA wraps the solution group's X in a LOCA::MultiContinuation::
    // ExtendedVector (carrying the continuation parameter alongside x). For
    // non-LOCA NOX runs (transient / Schwarz), x_abstract is directly a
    // NOX::Thyra::Vector. Handle both: try ExtendedVector first to unwrap,
    // fall through to the NOX::Thyra::Vector cast otherwise.
    auto x_abstract  = solver.getSolutionGroupPtr()->getXPtr();
    auto x_ext_const = Teuchos::rcp_dynamic_cast<const LOCA::MultiContinuation::ExtendedVector>(x_abstract, false);
    Teuchos::RCP<const NOX::Thyra::Vector> x_nox_const =
        Teuchos::nonnull(x_ext_const)
            ? Teuchos::rcp_dynamic_cast<const NOX::Thyra::Vector>(x_ext_const->getXVec(), false)
            : Teuchos::rcp_dynamic_cast<const NOX::Thyra::Vector>(x_abstract, false);
    if (Teuchos::nonnull(x_nox_const)) {
      auto         x_nox   = Teuchos::rcp_const_cast<NOX::Thyra::Vector>(x_nox_const);
      auto         x_thyra = x_nox->getThyraRCPVector();
      double const t_new = app_->fixTime(0.0);
      app_->warmstartPredictor(x_thyra, t_new);
      // Mark the solution group's F/J as stale by re-setting X. Without
      // this, NOX would treat the cached F (computed from the pre-warmstart
      // X) as still valid and skip recomputing it at iter 0.
      auto group_const = solver.getSolutionGroupPtr();
      auto group       = Teuchos::rcp_const_cast<NOX::Abstract::Group>(group_const);
      if (Teuchos::nonnull(x_ext_const)) {
        // For LOCA, set the underlying ExtendedVector through getX().
        auto x_ext = Teuchos::rcp_const_cast<LOCA::MultiContinuation::ExtendedVector>(x_ext_const);
        group->setX(*x_ext);
      } else {
        group->setX(*x_nox);
      }
    }
  }

  NOX::Abstract::Vector const& x = solver.getPreviousSolutionGroup().getX();
  norm_init_                     = x.norm();
  soln_init_                     = x.clone(NOX::DeepCopy);
}

void
SolutionSniffer::runPostSolve(NOX::Solver::Generic const& solver)
{
  NOX::Abstract::Vector const& y = solver.getSolutionGroup().getX();

  // Save solution
  last_soln_  = y.clone();
  norm_final_ = y.norm();

  NOX::Abstract::Vector const&        x         = *(soln_init_);
  Teuchos::RCP<NOX::Abstract::Vector> soln_diff = x.clone();
  NOX::Abstract::Vector&              dx        = *(soln_diff);

  dx.update(1.0, y, -1.0, x, 0.0);
  norm_diff_ = dx.norm();
}

void
SolutionSniffer::setStatusTest(Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> status_test)
{
  status_test_ = status_test;
}

ST
SolutionSniffer::getInitialNorm()
{
  return norm_init_;
}

ST
SolutionSniffer::getFinalNorm()
{
  return norm_final_;
}

ST
SolutionSniffer::getDifferenceNorm()
{
  return norm_diff_;
}

Teuchos::RCP<NOX::Abstract::Vector>
SolutionSniffer::getLastSoln()
{
  return last_soln_;
}

}  // namespace LCM
