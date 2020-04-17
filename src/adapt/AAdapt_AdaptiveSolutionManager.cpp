// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "AAdapt_AdaptiveSolutionManager.hpp"

#include "AAdapt_CopyRemesh.hpp"
#if defined(ALBANY_BGL)
#include "AAdapt_Erosion.hpp"
#include "AAdapt_TopologyModification.hpp"
#endif
#if defined(ALBANY_STK_PERCEPT)
#include "AAdapt_STKAdaptT.hpp"
#endif
#include "AAdapt_RC_Manager.hpp"
#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_ModelEvaluator.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Thyra_ModelEvaluatorDelegatorBase.hpp"

namespace AAdapt {

AdaptiveSolutionManager::AdaptiveSolutionManager(
    Teuchos::RCP<Teuchos::ParameterList> const& appParams,
    Teuchos::RCP<Thyra_Vector const> const&     initial_guess,
    Teuchos::RCP<ParamLib> const&               param_lib,
    Albany::StateManager const&                 stateMgr,
    Teuchos::RCP<rc::Manager> const&            rc_mgr,
    Teuchos::RCP<Teuchos_Comm const> const&     comm)
    : num_time_deriv(appParams->sublist("Discretization")
                         .get<int>("Number Of Time Derivatives")),
      appParams_(appParams),
      disc_(stateMgr.getDiscretization()),
      paramLib_(param_lib),
      stateMgr_(stateMgr),
      comm_(comm),
      out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  // Create problem PL
  Teuchos::RCP<Teuchos::ParameterList> problemParams =
      Teuchos::sublist(appParams_, "Problem", true);

  // Note that piroParams_ is a member of Thyra_AdaptiveSolutionManager
  piroParams_ = Teuchos::sublist(appParams_, "Piro", true);

  if (problemParams->isSublist(
          "Adaptation")) {  // If the user has specified adaptation on input,
                            // grab the sublist
    // Note that piroParams_ and adaptiveMesh_ are members of
    // Thyra_AdaptiveSolutionManager
    adaptParams_  = Teuchos::sublist(problemParams, "Adaptation", true);
    adaptiveMesh_ = true;
    buildAdapter(rc_mgr);
  }

  // Want the initial time in the parameter library to be correct
  // if this is a restart solution
  // MJJ (12/06/16) I'll go an remove this conditional "if
  // (disc_->hasRestartSolution())" Independent of restarting analysis you want
  // to have the capability of specify an initial time. Not doing so was causing
  // that the initial time was always set to zero and the latter cause many
  // problem because for time 0, dt is set up as 1.0e-15, creating serious
  // problem when a body source is specified at time 0. For example, in a heat
  // transfer problem at time 0 and specifying a heat source the problem may
  // corresponds to a "thermal shock" problem, causing some instabilities in the
  // time integrator.
  //  if (disc_->hasRestartSolution()) {
  if (paramLib_->isParameter("Time")) {
    double initialValue = 0.0;

    if (appParams->sublist("Problem").get<std::string>(
            "Solution Method", "Steady") == "Continuation") {
      initialValue = appParams->sublist("Piro")
                         .sublist("LOCA")
                         .sublist("Stepper")
                         .get<double>("Initial Value", 0.0);
    } else if (
        appParams->sublist("Problem").get<std::string>(
            "Solution Method", "Steady") == "Transient") {
      initialValue = appParams->sublist("Piro")
                         .sublist("Trapezoid Rule")
                         .get<double>("Initial Time", 0.0);
    }
    paramLib_->setRealValue<PHAL::AlbanyTraits::Residual>("Time", initialValue);
  }
  //  }

  resizeMeshDataArrays(disc_);

  auto                           wsElNodeEqID = disc_->getWsElNodeEqID();
  auto                           coords       = disc_->getCoords();
  Teuchos::ArrayRCP<std::string> wsEBNames    = disc_->getWsEBNames();
  int const                      numDim       = disc_->getNumDim();
  int const                      neq          = disc_->getNumEq();

  Teuchos::RCP<Teuchos::ParameterList> pbParams =
      Teuchos::sublist(appParams_, "Problem", true);

  if (Teuchos::nonnull(initial_guess)) {
    current_soln->col(0)->assign(*initial_guess);
  } else {
    cas_manager->scatter(
        current_soln->col(0),
        overlapped_soln->col(0),
        Albany::CombineMode::INSERT);
    InitialConditions(
        overlapped_soln->col(0),
        wsElNodeEqID,
        wsEBNames,
        coords,
        neq,
        numDim,
        pbParams->sublist("Initial Condition"),
        disc_->hasRestartSolution());
    cas_manager->combine(
        overlapped_soln->col(0),
        current_soln->col(0),
        Albany::CombineMode::INSERT);

    if (num_time_deriv > 0) {
      cas_manager->scatter(
          current_soln->col(1),
          overlapped_soln->col(1),
          Albany::CombineMode::INSERT);
      InitialConditions(
          overlapped_soln->col(1),
          wsElNodeEqID,
          wsEBNames,
          coords,
          neq,
          numDim,
          pbParams->sublist("Initial Condition Dot"));
      cas_manager->combine(
          overlapped_soln->col(1),
          current_soln->col(1),
          Albany::CombineMode::INSERT);
    }

    if (num_time_deriv > 1) {
      cas_manager->scatter(
          current_soln->col(2),
          overlapped_soln->col(2),
          Albany::CombineMode::INSERT);
      InitialConditions(
          overlapped_soln->col(2),
          wsElNodeEqID,
          wsEBNames,
          coords,
          neq,
          numDim,
          pbParams->sublist("Initial Condition DotDot"));
      cas_manager->combine(
          overlapped_soln->col(1),
          current_soln->col(1),
          Albany::CombineMode::INSERT);
    }
  }
}

void
AdaptiveSolutionManager::buildAdapter(const Teuchos::RCP<rc::Manager>& rc_mgr)
{
  std::string& method            = adaptParams_->get("Method", "");
  std::string  first_three_chars = method.substr(0, 3);

  if (method == "Copy Remesh") {
    adapter_ =
        Teuchos::rcp(new CopyRemesh(adaptParams_, paramLib_, stateMgr_, comm_));
  } else
#if defined(ALBANY_BGL)
      if (method == "Topmod") {
    adapter_ = Teuchos::rcp(
        new TopologyMod(adaptParams_, paramLib_, stateMgr_, comm_));
  } else if (method == "Erosion") {
    adapter_ =
        Teuchos::rcp(new Erosion(adaptParams_, paramLib_, stateMgr_, comm_));
  } else
#endif

#if defined(ALBANY_STK_PERCEPT)
      if (method == "Unif Size") {
    adapter_ = Teuchos::rcp(new STKAdaptT<STKUnifRefineField>(
        adaptParams_, paramLib_, stateMgr_, comm_));
  } else
#endif
  {
    ALBANY_ABORT(
        std::endl
        << "Error! Unknown adaptivity method requested:" << method << " !"
        << std::endl
        << "Supplied parameter list is " << std::endl
        << *adaptParams_);
  }

  // Teuchos::RCP<Teuchos::ParameterList const> valid_params =
  //  adapter_->getValidAdapterParameters();
  // adaptParams_->validateParameters(*valid_params);

  *out << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << " Mesh adapter has been initialized:\n"
       << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << std::endl;
}

bool
AdaptiveSolutionManager::adaptProblem()
{
  Teuchos::RCP<Thyra::ModelEvaluator<double>> model =
      this->getState()->getModel();

  // resize problem if the mesh adapts
  if (adapter_->adaptMesh()) {
    resizeMeshDataArrays(disc_);

    Teuchos::RCP<Thyra::ModelEvaluatorDelegatorBase<ST>> base =
        Teuchos::rcp_dynamic_cast<Thyra::ModelEvaluatorDelegatorBase<ST>>(
            model);

    // If dynamic cast fails
    ALBANY_PANIC(
        base == Teuchos::null,
        std::endl
            << "Error! : Cast to Thyra::ModelEvaluatorDelegatorBase failed!"
            << std::endl);

    auto me = Teuchos::rcp_dynamic_cast<Albany::ModelEvaluator>(
        base->getNonconstUnderlyingModel());

    // If dynamic cast fails
    ALBANY_PANIC(
        me == Teuchos::null,
        std::endl
            << "Error! : Cast to Albany::ModelEvaluator failed!" << std::endl);

    // Allocate storage in the model evaluator
    me->allocateVectors();

    // Build the solution group down in Thyra_AdaptiveSolutionManager.cpp
    this->getState()->buildSolutionGroup();

    // getSolutionMV() below returns the new solution vector with the fields
    // transferred to it

    current_soln = disc_->getSolutionMV();

    adapter_->postAdapt();

    *out << "Mesh adaptation was successfully performed!" << std::endl;

    return true;
  }

  *out << "Mesh adaptation was NOT successfully performed!" << std::endl;

  *out << "Mesh adaptation machinery has returned a FAILURE error code, "
          "exiting Albany!"
       << std::endl;

  ALBANY_ABORT("Mesh adaptation failed!\n");

  return false;
}

void
AdaptiveSolutionManager::resizeMeshDataArrays(
    const Teuchos::RCP<const Albany::AbstractDiscretization>& disc)
{
  auto owned_vs      = disc->getVectorSpace();
  auto overlapped_vs = disc->getOverlapVectorSpace();

  overlapped_soln = Thyra::createMembers(overlapped_vs, num_time_deriv + 1);
  overlapped_f    = Thyra::createMember(overlapped_vs);
  overlapped_jac  = disc->createOverlapJacobianOp();

  // This call allocates the non-overlapped MV
  current_soln = disc_->getSolutionMV();

  // Create the CombineAndScatterManager for handling distributed memory linear
  // algebra communications
  cas_manager = Albany::createCombineAndScatterManager(owned_vs, overlapped_vs);
}

Teuchos::RCP<Thyra_Vector const>
AdaptiveSolutionManager::updateAndReturnOverlapSolution(
    Thyra_Vector const& solution /* not overlapped */)
{
  cas_manager->scatter(
      solution, *overlapped_soln->col(0), Albany::CombineMode::INSERT);
  return overlapped_soln->col(0);
}

Teuchos::RCP<Thyra_Vector const>
AdaptiveSolutionManager::updateAndReturnOverlapSolutionDot(
    Thyra_Vector const& solution_dot /* not overlapped */)
{
  cas_manager->scatter(
      solution_dot, *overlapped_soln->col(1), Albany::CombineMode::INSERT);
  return overlapped_soln->col(1);
}

Teuchos::RCP<Thyra_Vector const>
AdaptiveSolutionManager::updateAndReturnOverlapSolutionDotDot(
    Thyra_Vector const& solution_dotdot /* not overlapped */)
{
  cas_manager->scatter(
      solution_dotdot, *overlapped_soln->col(2), Albany::CombineMode::INSERT);
  return overlapped_soln->col(2);
}

Teuchos::RCP<const Thyra_MultiVector>
AdaptiveSolutionManager::updateAndReturnOverlapSolutionMV(
    const Thyra_MultiVector& solution /* not overlapped */)
{
  cas_manager->scatter(solution, *overlapped_soln, Albany::CombineMode::INSERT);
  return overlapped_soln;
}

void
AdaptiveSolutionManager::scatterX(
    const Thyra_MultiVector& solution) /* not overlapped */
{
  cas_manager->scatter(solution, *overlapped_soln, Albany::CombineMode::INSERT);
}

void
AdaptiveSolutionManager::scatterX(
    Thyra_Vector const&                    x,
    const Teuchos::Ptr<Thyra_Vector const> x_dot,
    const Teuchos::Ptr<Thyra_Vector const> x_dotdot)
{
  cas_manager->scatter(
      x, *overlapped_soln->col(0), Albany::CombineMode::INSERT);

  if (!x_dot.is_null()) {
    ALBANY_PANIC(
        overlapped_soln->domain()->dim() < 2,
        "AdaptiveSolutionManager error: x_dot defined but only a single "
        "solution vector is available");
    cas_manager->scatter(
        *x_dot, *overlapped_soln->col(1), Albany::CombineMode::INSERT);
  }

  if (!x_dotdot.is_null()) {
    ALBANY_PANIC(
        overlapped_soln->domain()->dim() < 3,
        "AdaptiveSolutionManager error: x_dotdot defined but only two solution "
        "vectors are available");
    cas_manager->scatter(
        *x_dotdot, *overlapped_soln->col(2), Albany::CombineMode::INSERT);
  }
}

void
AdaptiveSolutionManager::projectCurrentSolution()
{
  // grp->getNOXThyraVecRCPX() is the current solution on the old mesh
  // TO provide an example, assume that the meshes are identical and we can just
  // copy the data between them (a Copy Remesh)
  //  const Teuchos::RCP<const Tpetra_Vector> testSolution =
  //  ConverterT::getConstTpetraVector(
  //    grp_->getNOXThyraVecRCPX()->getThyraRCPVector());
  //    *initial_xT = *testSolution;
}

}  // namespace AAdapt
