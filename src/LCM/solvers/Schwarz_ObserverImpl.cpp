// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include "Schwarz_ObserverImpl.hpp"

#include <cstdlib>
#include <string>

#include "Albany_AbstractDiscretization.hpp"

namespace LCM {

namespace {
// Phase 0 gate (mirrors Albany_ObserverImpl.cpp): when
// ALBANY_TEST_REBUILD_WORKSETS=1, exercise the workset-rebuild path
// on every subdomain after each accepted step.
bool
rebuildWorksetsEnabled()
{
  static bool const enabled = [] {
    char const* v = std::getenv("ALBANY_TEST_REBUILD_WORKSETS");
    return (v != nullptr) && (std::string(v) == "1");
  }();
  return enabled;
}

}  // namespace

ObserverImpl::ObserverImpl(Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>& apps) : StatelessObserverImpl(apps) { return; }

ObserverImpl::~ObserverImpl() { return; }

void
ObserverImpl::observeSolution(
    double                                           stamp,
    Teuchos::Array<Teuchos::RCP<Thyra_Vector const>> non_overlapped_solution,
    Teuchos::Array<Teuchos::RCP<Thyra_Vector const>> non_overlapped_solution_dot)
{
  for (int m = 0; m < this->n_models_; m++) {
    this->apps_[m]->evaluateStateFieldManager(stamp, *non_overlapped_solution[m], non_overlapped_solution_dot[m].ptr(), Teuchos::null);

    this->apps_[m]->getStateMgr().updateStates();
  }

  StatelessObserverImpl::observeSolution(stamp, non_overlapped_solution, non_overlapped_solution_dot);

  // M3a: activePart-based element death is on by default for each
  // subdomain. The function returns immediately if no cells died this
  // step. With M1's shared STKMeshStruct, all subdomains see the same
  // bulkData/activePart, so killing on one app naturally affects the
  // others; calling per-app is still correct because death_status_vecs_
  // is populated only by the app whose material model fires the death
  // predicate (typically the mechanical subdomain in ACE).
  bool any_killed = false;
  for (int m = 0; m < this->n_models_; m++) {
    if (this->apps_[m]->applyDeathToActivePart()) any_killed = true;
  }
  if (!any_killed && rebuildWorksetsEnabled()) {
    for (int m = 0; m < this->n_models_; m++) {
      this->apps_[m]->getDiscretization()->rebuildWorksets();
    }
  }
}

}  // namespace LCM
