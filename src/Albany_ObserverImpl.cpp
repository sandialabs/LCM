// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include "Albany_ObserverImpl.hpp"

#include <cstdlib>
#include <string>
#include <vector>

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_Application.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_Utils.hpp"
#include "Teuchos_VerboseObject.hpp"

namespace Albany {

void
ObserverImpl::projectNodalFields(double stamp)
{
  // Build the projectors once, from the problem's "Nodal Field Projection"
  // sublist. Each reads the saved quadrature-point states and writes the
  // proj_nodal_* nodal states that Application::buildProblem registered, so the
  // projection is fully decoupled from the (now removed) response path.
  if (!projectors_built_) {
    projectors_built_ = true;
    auto problem_pl   = app_->getProblemPL();
    if (problem_pl->isSublist("Nodal Field Projection")) {
      auto&     nfp = problem_pl->sublist("Nodal Field Projection");
      int const nf  = nfp.get<int>("Number of Fields", 0);
      std::vector<NodalFieldProjector::FieldSpec> fields;
      for (int f = 0; f < nf; ++f) {
        fields.push_back({nfp.get<std::string>(Albany::strint("IP Field Name", f)), nfp.get<std::string>(Albany::strint("IP Field Layout", f))});
      }
      std::string const mass_matrix_type = nfp.get<std::string>("Mass Matrix Type", "Full");
      bool const        output_to_exodus = nfp.get<bool>("Output to File", true);
      projectors_.push_back(Teuchos::rcp(new NodalFieldProjector(app_, fields, mass_matrix_type, output_to_exodus)));
    }
  }
  for (auto const& projector : projectors_) projector->project(stamp);
}

namespace {
// Phase 0 gate: set ALBANY_TEST_REBUILD_WORKSETS=1 to exercise the
// mid-run workset rebuild path after every accepted step. Default off
// keeps existing behavior bit-identical.
bool
rebuildWorksetsEnabled()
{
  static bool const enabled = [] {
    char const* v  = std::getenv("ALBANY_TEST_REBUILD_WORKSETS");
    bool const  on = (v != nullptr) && (std::string(v) == "1");
    if (on) {
      *Teuchos::VerboseObjectBase::getDefaultOStream()
          << "[Phase 0] ALBANY_TEST_REBUILD_WORKSETS=1: "
             "rebuildWorksets() will fire after every accepted step.\n";
    }
    return on;
  }();
  return enabled;
}

}  // namespace

ObserverImpl::ObserverImpl(const Teuchos::RCP<Application>& app) : StatelessObserverImpl(app) {}

void
ObserverImpl::observeSolution(
    double                                  stamp,
    Thyra_Vector const&                     nonOverlappedSolution,
    const Teuchos::Ptr<Thyra_Vector const>& nonOverlappedSolutionDot,
    const Teuchos::Ptr<Thyra_Vector const>& nonOverlappedSolutionDotDot)
{
  app_->evaluateStateFieldManager(stamp, nonOverlappedSolution, nonOverlappedSolutionDot, nonOverlappedSolutionDotDot);

  app_->getStateMgr().updateStates();

  //! update distributed parameters in the mesh
  auto distParamLib = app_->getDistributedParameterLibrary();
  auto disc         = app_->getDiscretization();
  distParamLib->scatter();
  for (auto it : *distParamLib) {
    disc->setField(
        *it.second->overlapped_vector(),
        it.second->name(),
        /*overlapped*/ true);
  }

  // Project the just-saved quadrature-point states to nodal fields before the
  // write, so the written frame (including the first dynamic step) carries the
  // correct proj_nodal_* fields (GitHub #11).
  projectNodalFields(stamp);

  StatelessObserverImpl::observeSolution(stamp, nonOverlappedSolution, nonOverlappedSolutionDot, nonOverlappedSolutionDotDot);

  // M3a: activePart-based element death is on by default. The function
  // returns immediately if no cells died this step. It rebuilds worksets
  // internally when it does kill cells, so the Phase 0 test-only rebuild
  // is skipped in that case to avoid double work.
  bool const killed = app_->applyDeathToActivePart();
  if (!killed && rebuildWorksetsEnabled()) {
    app_->getDiscretization()->rebuildWorksets();
  }
}

void
ObserverImpl::observeSolution(double stamp, const Thyra_MultiVector& nonOverlappedSolution)
{
  app_->evaluateStateFieldManager(stamp, nonOverlappedSolution);
  app_->getStateMgr().updateStates();
  // See the vector overload: project before the write so the MultiVector
  // (e.g. Trapezoid) path, whose observer omits observeResponse entirely, still
  // gets non-zero proj_nodal_* fields (GitHub #11).
  projectNodalFields(stamp);
  StatelessObserverImpl::observeSolution(stamp, nonOverlappedSolution);

  // M3a: activePart-based element death is on by default. The function
  // returns immediately if no cells died this step. It rebuilds worksets
  // internally when it does kill cells, so the Phase 0 test-only rebuild
  // is skipped in that case to avoid double work.
  bool const killed = app_->applyDeathToActivePart();
  if (!killed && rebuildWorksetsEnabled()) {
    app_->getDiscretization()->rebuildWorksets();
  }
}

void
ObserverImpl::parameterChanged(std::string const& param)
{
  //! If a parameter has changed in value, saved/unsaved fields must be updated
  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << param << " has changed!" << std::endl;
  app_->getPhxSetup()->init_unsaved_param(param);
}

}  // namespace Albany
