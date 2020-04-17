// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include "Albany_StatelessObserverImpl.hpp"

#include <string>

#include "Albany_AbstractDiscretization.hpp"
#include "Teuchos_TimeMonitor.hpp"

namespace Albany {

StatelessObserverImpl::StatelessObserverImpl(
    const Teuchos::RCP<Application>& app)
    : app_(app),
      solOutTime_(Teuchos::TimeMonitor::getNewTimer("Albany: Output to File"))
{
}

RealType
StatelessObserverImpl::getTimeParamValueOrDefault(RealType defaultValue) const
{
  std::string const label("Time");
  // IKT, NOTE: solMethod == 2 corresponds to TransientTempus
  bool const use_time_param =
      (app_->getParamLib()->isParameter(label) == true) &&
      (app_->getSchwarzAlternating() == false) &&
      (app_->getSolutionMethod() != 2);

  double const this_time =
      use_time_param == true ?
          app_->getParamLib()->getRealValue<PHAL::AlbanyTraits::Residual>(
              label) :
          defaultValue;

  return this_time;
}

Teuchos::RCP<Thyra_VectorSpace const>
StatelessObserverImpl::getNonOverlappedVectorSpace() const
{
  return app_->getVectorSpace();
}

void
StatelessObserverImpl::observeSolution(
    double                                  stamp,
    Thyra_Vector const&                     nonOverlappedSolution,
    const Teuchos::Ptr<Thyra_Vector const>& nonOverlappedSolutionDot)
{
  Teuchos::TimeMonitor                   timer(*solOutTime_);
  Teuchos::RCP<Thyra_Vector const> const overlappedSolution =
      app_->getAdaptSolMgr()->updateAndReturnOverlapSolution(
          nonOverlappedSolution);
  if (nonOverlappedSolutionDot != Teuchos::null) {
    Teuchos::RCP<Thyra_Vector const> const overlappedSolutionDot =
        app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDot(
            *nonOverlappedSolutionDot);
    app_->getDiscretization()->writeSolution(
        *overlappedSolution,
        *overlappedSolutionDot,
        stamp,
        /*overlapped =*/true);
  } else {
    app_->getDiscretization()->writeSolution(
        *overlappedSolution, stamp, /*overlapped =*/true);
  }
}

void
StatelessObserverImpl::observeSolution(
    double                                  stamp,
    Thyra_Vector const&                     nonOverlappedSolution,
    const Teuchos::Ptr<Thyra_Vector const>& nonOverlappedSolutionDot,
    const Teuchos::Ptr<Thyra_Vector const>& nonOverlappedSolutionDotDot)
{
  Teuchos::TimeMonitor                   timer(*solOutTime_);
  Teuchos::RCP<Thyra_Vector const> const overlappedSolution =
      app_->getAdaptSolMgr()->updateAndReturnOverlapSolution(
          nonOverlappedSolution);
  if (nonOverlappedSolutionDot != Teuchos::null) {
    Teuchos::RCP<Thyra_Vector const> const overlappedSolutionDot =
        app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDot(
            *nonOverlappedSolutionDot);
    if (nonOverlappedSolutionDotDot != Teuchos::null) {
      Teuchos::RCP<Thyra_Vector const> const overlappedSolutionDotDot =
          app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDotDot(
              *nonOverlappedSolutionDotDot);
      app_->getDiscretization()->writeSolution(
          *overlappedSolution,
          *overlappedSolutionDot,
          *overlappedSolutionDotDot,
          stamp,
          /*overlapped =*/true);
    } else {
      app_->getDiscretization()->writeSolution(
          *overlappedSolution,
          *overlappedSolutionDot,
          stamp,
          /*overlapped =*/true);
    }
  } else {
    app_->getDiscretization()->writeSolution(
        *overlappedSolution, stamp, /*overlapped =*/true);
  }
}

void
StatelessObserverImpl::observeSolution(
    double                   stamp,
    const Thyra_MultiVector& nonOverlappedSolution)
{
  Teuchos::TimeMonitor                        timer(*solOutTime_);
  const Teuchos::RCP<const Thyra_MultiVector> overlappedSolution =
      app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionMV(
          nonOverlappedSolution);
  app_->getDiscretization()->writeSolutionMV(
      *overlappedSolution, stamp, /*overlapped =*/true);
}

}  // namespace Albany
