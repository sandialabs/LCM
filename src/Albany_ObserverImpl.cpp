// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include "Albany_ObserverImpl.hpp"

#include <cstdlib>
#include <string>

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Teuchos_VerboseObject.hpp"

namespace Albany {

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

  StatelessObserverImpl::observeSolution(stamp, nonOverlappedSolution, nonOverlappedSolutionDot, nonOverlappedSolutionDotDot);

  if (rebuildWorksetsEnabled()) {
    app_->getDiscretization()->rebuildWorksets();
  }
}

void
ObserverImpl::observeSolution(double stamp, const Thyra_MultiVector& nonOverlappedSolution)
{
  app_->evaluateStateFieldManager(stamp, nonOverlappedSolution);
  app_->getStateMgr().updateStates();
  StatelessObserverImpl::observeSolution(stamp, nonOverlappedSolution);

  if (rebuildWorksetsEnabled()) {
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
