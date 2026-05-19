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

  if (rebuildWorksetsEnabled()) {
    for (int m = 0; m < this->n_models_; m++) {
      this->apps_[m]->getDiscretization()->rebuildWorksets();
    }
  }
}

}  // namespace LCM
