// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_Application.hpp"

#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include "Teuchos_DefaultMpiComm.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Part.hpp"
#include "stk_util/parallel/ParallelReduceBool.hpp"

#include "AAdapt_RC_Manager.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_DataTypes.hpp"
#include "utility/Albany_GlobalLocalIndexer.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_DummyParameterAccessor.hpp"
#include "Albany_ElementDeath.hpp"
#include "Albany_Macros.hpp"
#include "Albany_ProblemFactory.hpp"
#include "Albany_ResponseFactory.hpp"
#include "Albany_ScalarResponseFunction.hpp"
#include "Albany_SchwarzTransfer.hpp"
#include "Albany_ThyraUtils.hpp"
#include "PHAL_Utilities.hpp"
#include "SolutionSniffer.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_VectorBase.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "stk_expreval/Evaluator.hpp"

using Teuchos::ArrayRCP;
using Teuchos::getFancyOStream;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_dynamic_cast;
using Teuchos::rcpFromRef;
using Teuchos::TimeMonitor;

int countSol;  // counter which counts instances of solution (for debug output)
int countRes;  // counter which counts instances of residual (for debug output)
int countJac;  // counter which counts instances of Jacobian (for debug output)
int countScale;

namespace {
int
calcTangentDerivDimension(const Teuchos::RCP<Teuchos::ParameterList>& problemParams)
{
  Teuchos::ParameterList& parameterParams          = problemParams->sublist("Parameters");
  int                     num_param_vecs           = parameterParams.get("Number of Parameter Vectors", 0);
  bool                    using_old_parameter_list = false;
  if (parameterParams.isType<int>("Number")) {
    int numParameters = parameterParams.get<int>("Number");
    if (numParameters > 0) {
      num_param_vecs           = 1;
      using_old_parameter_list = true;
    }
  }
  int np = 0;
  for (int i = 0; i < num_param_vecs; ++i) {
    Teuchos::ParameterList& pList = using_old_parameter_list ? parameterParams : parameterParams.sublist(Albany::strint("Parameter Vector", i));
    np += pList.get<int>("Number");
  }
  return std::max(1, np);
}
}  // namespace

namespace Albany {

Application::Application(
    const RCP<Teuchos_Comm const>&     comm_,
    const RCP<Teuchos::ParameterList>& params,
    RCP<Thyra_Vector const> const&     initial_guess,
    bool const                         schwarz)
    : is_schwarz_{schwarz},
      comm(comm_),
      out(Teuchos::VerboseObjectBase::getDefaultOStream()),
      params_(params),
      physicsBasedPreconditioner(false),
      shapeParamsHaveBeenReset(false),
      phxGraphVisDetail(0),
      stateGraphVisDetail(0),
      morphFromInit(true)
{
  initialSetUp(params);
  createMeshSpecs();
  buildProblem();
  createDiscretization();
  finalSetUp(params, initial_guess);
}

Application::Application(const RCP<Teuchos_Comm const>& comm_)
    : comm(comm_),
      out(Teuchos::VerboseObjectBase::getDefaultOStream()),
      physicsBasedPreconditioner(false),
      shapeParamsHaveBeenReset(false),
      phxGraphVisDetail(0),
      stateGraphVisDetail(0),
      morphFromInit(true)
{
  // Nothing to be done here
}

Application::Application(
    const RCP<Teuchos_Comm const>&                     comm_,
    const RCP<Teuchos::ParameterList>&                 params,
    const Teuchos::RCP<Albany::AbstractMeshStruct>&    sharedMesh,
    bool const                                         deferPostCommit,
    RCP<Thyra_Vector const> const&                     initial_guess,
    bool const                                         schwarz)
    : is_schwarz_{schwarz},
      comm(comm_),
      out(Teuchos::VerboseObjectBase::getDefaultOStream()),
      params_(params),
      physicsBasedPreconditioner(false),
      shapeParamsHaveBeenReset(false),
      phxGraphVisDetail(0),
      stateGraphVisDetail(0),
      morphFromInit(true)
{
  initialSetUp(params);

  // Shared-mesh path: inject the pre-built mesh struct so the
  // DiscretizationFactory uses it instead of constructing its own from
  // YAML. createMeshSpecs(mesh) registers the shared mesh on the factory.
  ALBANY_PANIC(sharedMesh.is_null(), "Application shared-mesh ctor requires non-null sharedMesh.\n");
  createMeshSpecs(sharedMesh);
  buildProblem();

  if (deferPostCommit) {
    // Defer disc->updateMesh() so the orchestrator can call
    // sharedMesh->commitAndPopulate() first. createDiscretization
    // still runs setFieldAndBulkData (which declares this app's fields
    // on the shared metaData but skips commit when deferCommit is set
    // on the mesh).
    // Mark the deferred state BEFORE createDiscretization so the
    // eliminateConstrainedDOFs called from createDiscretization's tail
    // takes the ACE early-exit path (overlap vector space isn't a
    // concrete Thyra type until commitAndPopulate has run).
    deferred_post_commit_pending_ = true;
    deferred_params_              = params;
    deferred_shared_mesh_         = sharedMesh;

    discFactory->deferUpdateMesh = true;
    createDiscretization();
    discFactory->deferUpdateMesh = false;  // restore for any later calls

    // initial_guess is captured by finalizePostCommit's parameter.
    (void)initial_guess;
  } else {
    createDiscretization();
    finalSetUp(params, initial_guess);
  }
}

void
Application::finalizePostCommit(RCP<Thyra_Vector const> const& initial_guess)
{
  if (!deferred_post_commit_pending_) return;

  // The shared mesh has been committed and populated by the orchestrator.
  // Run the discretization update + finalSetUp that the shared-mesh ctor
  // deferred. updateMesh lives on STKDiscretization specifically; for
  // shared-mesh ACE coupling we always have STK.
  auto stk_disc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(disc, true);
  stk_disc->updateMesh();

  // ACE-deferred elimination: now that the shared mesh is committed and the
  // discretization's nodesets + overlap vector space are concrete, parse the
  // DBCs and build descriptors. Clear the deferred flag first so the early-
  // exit guard in eliminateConstrainedDOFs lets this call through.
  deferred_post_commit_pending_ = false;
  eliminateConstrainedDOFs();

  finalSetUp(deferred_params_, initial_guess);
  deferred_params_              = Teuchos::null;
  deferred_shared_mesh_         = Teuchos::null;
}

void
Application::initialSetUp(const RCP<Teuchos::ParameterList>& params)
{
  // Create parameter libraries
  paramLib     = rcp(new ParamLib);
  distParamLib = rcp(new DistributedParameterLibrary);

  // Create problem object
  problemParams = Teuchos::sublist(params, "Problem", true);

  ProblemFactory problemFactory(params, paramLib, comm);
  rc_mgr = AAdapt::rc::Manager::create(Teuchos::rcp(&stateMgr, false), *problemParams);
  if (Teuchos::nonnull(rc_mgr)) {
    problemFactory.setReferenceConfigurationManager(rc_mgr);
  }
  problem = problemFactory.create();

  // Validate Problem parameters against list for this specific problem
  problemParams->validateParameters(*(problem->getValidProblemParameters()), 0);

  try {
    tangent_deriv_dim = calcTangentDerivDimension(problemParams);
  } catch (...) {
    tangent_deriv_dim = 1;
  }

  // Initialize Phalanx postRegistration setup
  phxSetup = Teuchos::rcp(new PHAL::Setup());
  phxSetup->init_problem_params(problemParams);

  // Pull the number of solution vectors out of the problem and send them to the
  // discretization list, if the user specifies this in the problem
  Teuchos::ParameterList& discParams = params->sublist("Discretization");

  // Set in Albany_AbstractProblem constructor or in siblings
  num_time_deriv = problemParams->get<int>("Number Of Time Derivatives");

  // Possibly set in the Discretization list in the input file - this overrides
  // the above if set
  int num_time_deriv_from_input = discParams.get<int>("Number Of Time Derivatives", -1);
  if (num_time_deriv_from_input < 0)  // Use the value from the problem by default
    discParams.set<int>("Number Of Time Derivatives", num_time_deriv);
  else
    num_time_deriv = num_time_deriv_from_input;

  if (is_schwarz_ == true) {
    // Write DTK Field to Exodus if Schwarz is used
    discParams.set<bool>("Output DTK Field to Exodus", true);
  }

  ALBANY_PANIC(num_time_deriv > 2, "Input error: number of time derivatives must be <= 2 " << "(solution, solution_dot, solution_dotdot)");

  // Save the solution method to be used
  std::string solutionMethod = problemParams->get("Solution Method", "Steady");
  if (solutionMethod == "Steady") {
    solMethod = Steady;
  } else if (solutionMethod == "Continuation") {
    solMethod            = Continuation;
    bool const have_piro = params->isSublist("Piro");
    ALBANY_ASSERT(have_piro == true, "Error! Piro sublist not found.");
    Teuchos::ParameterList& piro_params = params->sublist("Piro");
    bool const              have_nox    = piro_params.isSublist("NOX");
    if (have_nox) {
      Teuchos::ParameterList nox_params       = piro_params.sublist("NOX");
      std::string            nonlinear_solver = nox_params.get<std::string>("Nonlinear Solver");
    }
  } else if (solutionMethod == "Transient") {
    solMethod = Transient;
  } else if (solutionMethod == "Eigensolve") {
    solMethod = Eigensolve;
  } else if (solutionMethod == "Transient Tempus" || "Transient Tempus No Piro") {
    solMethod = TransientTempus;

    // Sanity check on Tempus + d-Form Newmark: it needs Line Search Based.
    Teuchos::ParameterList& piro_params           = params->sublist("Piro", true);
    Teuchos::ParameterList& tempus_params         = piro_params.sublist("Tempus", true);
    Teuchos::ParameterList& tempus_stepper_params = tempus_params.sublist("Tempus Stepper", true);
    std::string const       stepper_type          = tempus_stepper_params.get<std::string>("Stepper Type");
    if (stepper_type == "Newmark Implicit d-Form") {
      std::string const       solver_name        = tempus_stepper_params.get<std::string>("Solver Name");
      Teuchos::ParameterList& solver_name_params = tempus_stepper_params.sublist(solver_name);
      Teuchos::ParameterList& nox_params         = solver_name_params.sublist("NOX", true);
      std::string const       nonlinear_solver   = nox_params.get<std::string>("Nonlinear Solver");
      ALBANY_PANIC(
          nonlinear_solver != "Line Search Based",
          "Newmark Implicit d-Form Stepper Type will not work correctly "
          "with 'Nonlinear Solver' = "
              << nonlinear_solver
              << "!  The valid Nonlinear Solver for this scheme is 'Line "
                 "Search Based'.");
    }
  } else {
    ALBANY_ABORT(
        "Solution Method must be Steady, Transient, Transient Tempus, "
        "Transient Tempus No Piro, "
        << "Continuation, or Eigensolve, not : " << solutionMethod);
  }

  std::string stepperType;
  if (solMethod == TransientTempus) {
    // Get Piro PL
    Teuchos::RCP<Teuchos::ParameterList> piroParams = Teuchos::sublist(params, "Piro", true);
    // Check if there is Tempus Solver sublist, and get the stepper type
    if (piroParams->isSublist("Tempus")) {
      Teuchos::RCP<Teuchos::ParameterList> rythmosSolverParams = Teuchos::sublist(piroParams, "Tempus", true);
    }
  }

  determinePiroSolver(params);

  physicsBasedPreconditioner = problemParams->get("Use Physics-Based Preconditioner", false);
  if (physicsBasedPreconditioner) {
    precType = problemParams->get("Physics-Based Preconditioner", "Teko");
  }

  // Create debug output object
  auto debugParams       = Teuchos::sublist(params, "Debug Output", true);
  writeToMatrixMarketSol = debugParams->get("Write Solution to MatrixMarket", 0);
  writeToMatrixMarketRes = debugParams->get("Write Residual to MatrixMarket", 0);
  writeToMatrixMarketJac = debugParams->get("Write Jacobian to MatrixMarket", 0);
  writeToCoutSol         = debugParams->get("Write Solution to Standard Output", 0);
  writeToCoutRes         = debugParams->get("Write Residual to Standard Output", 0);
  writeToCoutJac         = debugParams->get("Write Jacobian to Standard Output", 0);
  derivatives_check_     = debugParams->get<int>("Derivative Check", 0);
  // the above parameters cannot have values < -1
  if (writeToMatrixMarketSol < -1) {
    ALBANY_ABORT(
        std::endl
        << "Invalid Parameter Write Solution to MatrixMarket.  Acceptable "
           "values are -1, 0, 1, 2, ... "
        << std::endl);
  }
  if (writeToMatrixMarketRes < -1) {
    ALBANY_ABORT(
        std::endl
        << "Invalid Parameter Write Residual to MatrixMarket.  Acceptable "
           "values are -1, 0, 1, 2, ... "
        << std::endl);
  }
  if (writeToMatrixMarketJac < -1) {
    ALBANY_ABORT(
        std::endl
        << "Invalid Parameter Write Jacobian to MatrixMarket.  Acceptable "
           "values are -1, 0, 1, 2, ... "
        << std::endl);
  }
  if (writeToCoutSol < -1) {
    ALBANY_ABORT(
        std::endl
        << "Invalid Parameter Write Solution to Standard Output.  "
           "Acceptable values are -1, 0, 1, 2, ... "
        << std::endl);
  }
  if (writeToCoutRes < -1) {
    ALBANY_ABORT(
        std::endl
        << "Invalid Parameter Write Residual to Standard Output.  "
           "Acceptable values are -1, 0, 1, 2, ... "
        << std::endl);
  }
  if (writeToCoutJac < -1) {
    ALBANY_ABORT(
        std::endl
        << "Invalid Parameter Write Jacobian to Standard Output.  "
           "Acceptable values are -1, 0, 1, 2, ... "
        << std::endl);
  }

  countSol   = 0;  // initiate counter that counts instances of solution vector
  countRes   = 0;  // initiate counter that counts instances of residual vector
  countJac   = 0;  // initiate counter that counts instances of Jacobian matrix
  countScale = 0;

  // Create discretization object
  discFactory = rcp(new Albany::DiscretizationFactory(params, comm));

  // Check for Schwarz parameters
  bool const has_app_array          = params->isParameter("Application Array");
  bool const has_app_index          = params->isParameter("Application Index");
  bool const has_app_name_index_map = params->isParameter("Application Name Index Map");

  // Only if all these are present set them in the app.
  bool const has_all = has_app_array && has_app_index && has_app_name_index_map;

  if (has_all == true) {
    Teuchos::ArrayRCP<Teuchos::RCP<Application>> aa = params->get<Teuchos::ArrayRCP<Teuchos::RCP<Application>>>("Application Array");

    int const ai = params->get<int>("Application Index");

    Teuchos::RCP<std::map<std::string, int>> anim = params->get<Teuchos::RCP<std::map<std::string, int>>>("Application Name Index Map");

    this->setApplications(aa.create_weak());
    this->setAppIndex(ai);
    this->setAppNameIndexMap(anim);
  }
}

void
Application::createMeshSpecs()
{
  // Get mesh specification object: worksetSize, cell topology, etc
  meshSpecs = discFactory->createMeshSpecs();
}

void
Application::createMeshSpecs(Teuchos::RCP<AbstractMeshStruct> mesh)
{
  // Get mesh specification object: worksetSize, cell topology, etc
  meshSpecs = discFactory->createMeshSpecs(mesh);
}

void
Application::buildProblem()
{
  // This is needed for Schwarz coupling so that when Dirichlet
  // BCs are created we know what application is doing it.
  problem->setApplication(Teuchos::rcp(this, false));

  problem->buildProblem(meshSpecs, stateMgr);

  neq               = problem->numEquations();
  spatial_dimension = problem->spatialDimension();

  // Construct responses
  // This really needs to happen after the discretization is created for
  // distributed responses, but currently it can't be moved because there
  // are responses that setup states, which has to happen before the
  // discretization is created.  We will delay setup of the distributed
  // responses to deal with this temporarily.
  Teuchos::ParameterList& responseList = problemParams->sublist("Response Functions");
  ResponseFactory         responseFactory(Teuchos::rcp(this, false), problem, meshSpecs, Teuchos::rcp(&stateMgr, false));
  responses            = responseFactory.createResponseFunctions(responseList);
  observe_responses    = responseList.get("Observe Responses", true);
  response_observ_freq = responseList.get("Responses Observation Frequency", 1);
  const Teuchos::Array<unsigned int> defaultDataUnsignedInt;
  relative_responses = responseList.get("Relative Responses Markers", defaultDataUnsignedInt);

  // Build state field manager
  if (Teuchos::nonnull(rc_mgr)) rc_mgr->beginBuildingSfm();
  sfm.resize(meshSpecs.size());
  Teuchos::RCP<PHX::DataLayout> dummy = Teuchos::rcp(new PHX::MDALayout<Dummy>(0));
  for (int ps = 0; ps < meshSpecs.size(); ps++) {
    std::string              elementBlockName              = meshSpecs[ps]->ebName;
    std::vector<std::string> responseIDs_to_require        = stateMgr.getResidResponseIDsToRequire(elementBlockName);
    sfm[ps]                                                = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>> tags = problem->buildEvaluators(*sfm[ps], *meshSpecs[ps], stateMgr, BUILD_STATE_FM, Teuchos::null);
    std::vector<std::string>::const_iterator          it;
    for (it = responseIDs_to_require.begin(); it != responseIDs_to_require.end(); it++) {
      std::string const&                              responseID = *it;
      PHX::Tag<PHAL::AlbanyTraits::Residual::ScalarT> res_response_tag(responseID, dummy);
      sfm[ps]->requireField<PHAL::AlbanyTraits::Residual>(res_response_tag);
    }
  }
  if (Teuchos::nonnull(rc_mgr)) rc_mgr->endBuildingSfm();
}

void
Application::createDiscretization()
{
  // Create the full mesh
  disc = discFactory->createDiscretization(
      neq,
      problem->getSideSetEquations(),
      stateMgr.getStateInfoStruct(),
      stateMgr.getSideSetStateInfoStruct(),
      problem->getFieldRequirements(),
      problem->getSideSetFieldRequirements(),
      problem->getNullSpace());

  eliminateConstrainedDOFs();
}

double
Application::DBCDescriptor::eval(double time) const
{
  switch (kind) {
    case Kind::Constant: return constant;

    case Kind::TimeArray: {
      auto const n = times.size();
      if (time <= times[0]) return values[0];
      if (time >= times[n - 1]) return values[n - 1];
      for (std::size_t i = 1; i < n; ++i) {
        if (time < times[i]) {
          double const slope = (values[i] - values[i - 1]) / (times[i] - times[i - 1]);
          return values[i - 1] + slope * (time - times[i - 1]);
        }
      }
      return values[n - 1];
    }

    case Kind::Expression: {
      stk::expreval::Eval ev(expr_str);
      ev.parse();
      ev.bindVariable("t", time);
      ev.bindVariable("x", x);
      ev.bindVariable("y", y);
      ev.bindVariable("z", z);
      return ev.evaluate();
    }

    case Kind::Schwarz: return schwarz_cached_value;

    case Kind::Kfield: {
      // Linearly interpolate KI(t) and KII(t) from the time series and
      // multiply by the YAML "Kfield KI" / "Kfield KII" scalars (which the
      // legacy evaluator registered as Sacado parameters so LOCA could
      // continuation-sweep them; elimination treats them as fixed scalars).
      auto const n = kfield_time_values.size();
      double     ki_t, kii_t;
      if (n == 0) {
        ki_t = 0.0; kii_t = 0.0;
      } else if (time <= kfield_time_values[0]) {
        ki_t = kfield_ki_values[0]; kii_t = kfield_kii_values[0];
      } else if (time >= kfield_time_values[n - 1]) {
        ki_t = kfield_ki_values[n - 1]; kii_t = kfield_kii_values[n - 1];
      } else {
        std::size_t i = 1;
        while (i < n && kfield_time_values[i] < time) ++i;
        double const dt    = kfield_time_values[i] - kfield_time_values[i - 1];
        double const alpha = (time - kfield_time_values[i - 1]) / dt;
        ki_t  = kfield_ki_values[i - 1]  + alpha * (kfield_ki_values[i]  - kfield_ki_values[i - 1]);
        kii_t = kfield_kii_values[i - 1] + alpha * (kfield_kii_values[i] - kfield_kii_values[i - 1]);
      }
      double const ki  = kfield_ki_scale  * ki_t;
      double const kii = kfield_kii_scale * kii_t;
      // Williams' plane-strain crack-tip displacement field, in the
      // arrangement the legacy LCM evaluator used:
      //   coeff_1 = KI / mu * sqrt(R / 2π)
      //   coeff_2 = KII / mu * sqrt(R / 2π)
      //   ux = coeff_1 (1 - 2 nu + sin^2(θ/2)) cos(θ/2)
      //      + coeff_2 (2 - 2 nu + cos^2(θ/2)) sin(θ/2)
      //   uy = coeff_1 (2 - 2 nu - cos^2(θ/2)) sin(θ/2)
      //      + coeff_2 (-1 + 2 nu + sin^2(θ/2)) cos(θ/2)
      double const tau     = 2.0 * M_PI;
      double const half_th = kfield_theta * 0.5;
      double const c_half  = std::cos(half_th);
      double const s_half  = std::sin(half_th);
      double const coeff_1 = (ki  / kfield_mu) * std::sqrt(kfield_r / tau);
      double const coeff_2 = (kii / kfield_mu) * std::sqrt(kfield_r / tau);
      double const ki_x  = coeff_1 * (1.0 - 2.0 * kfield_nu + s_half * s_half) * c_half;
      double const ki_y  = coeff_1 * (2.0 - 2.0 * kfield_nu - c_half * c_half) * s_half;
      double const kii_x = coeff_2 * (2.0 - 2.0 * kfield_nu + c_half * c_half) * s_half;
      double const kii_y = coeff_2 * (-1.0 + 2.0 * kfield_nu + s_half * s_half) * c_half;
      if (kfield_component == 0) return ki_x + kii_x;
      return ki_y + kii_y;
    }

    case Kind::EquilibriumConcentration:
      // Coupled-injection BC: the value depends on the overlap pressure DOF
      // at the same node. The injection loop reads the pressure each fill
      // and caches the resulting c_bc here. eval() returns the cached value
      // so callers like expandToFullSolution / response evaluation see the
      // same value the linear solve used.
      return eqconc_cached_value;
  }
  return 0.0;
}

Application::DBCDescriptor::Derivs
Application::DBCDescriptor::derivs_at(double time) const
{
  switch (kind) {
    case Kind::Constant: return {0.0, 0.0};

    case Kind::TimeArray: {
      auto const n = times.size();
      // Endpoints and degenerate cases: hold the last segment's slope flat at
      // the endpoints; outside the table, slope is 0 (value clamped by eval).
      if (n < 2 || time <= times[0] || time >= times[n - 1]) {
        return {0.0, 0.0};
      }
      for (std::size_t i = 1; i < n; ++i) {
        if (time < times[i]) {
          double const slope = (values[i] - values[i - 1]) / (times[i] - times[i - 1]);
          // Interior of a piecewise-linear segment: a = 0; the knot deltas
          // are treated as zero for the purposes of the BC acceleration.
          return {slope, 0.0};
        }
      }
      return {0.0, 0.0};
    }

    case Kind::Expression: {
      // Central FD over eval(t ± h). h clipped so t-h stays nonnegative when
      // the expression is only valid for t >= 0. 1e-6 keeps relative truncation
      // and roundoff balanced for typical polynomial/transcendental BCs.
      double const h_max = 1.0e-6;
      double const h     = (time > h_max) ? h_max : (time > 0.0 ? time : h_max);
      double const f0    = eval(time);
      double const fp    = eval(time + h);
      double const fm    = eval(time - h);
      double const v     = (fp - fm) / (2.0 * h);
      double const a     = (fp - 2.0 * f0 + fm) / (h * h);
      return {v, a};
    }

    case Kind::Schwarz: return {schwarz_cached_velocity, schwarz_cached_acceleration};

    case Kind::Kfield: {
      double const h_max = 1.0e-6;
      double const h     = (time > h_max) ? h_max : (time > 0.0 ? time : h_max);
      double const f0    = eval(time);
      double const fp    = eval(time + h);
      double const fm    = eval(time - h);
      return {(fp - fm) / (2.0 * h), (fp - 2.0 * f0 + fm) / (h * h)};
    }

    case Kind::EquilibriumConcentration:
      // Quasistatic concentration BC — no inherent time derivative.
      return {0.0, 0.0};
  }
  return {0.0, 0.0};
}

void
Application::eliminateConstrainedDOFs()
{

  auto const& node_set_ids = problem->getNodeSetIDs();
  auto const& bc_names     = problem->getDirichletBCNames();

  if (node_set_ids.empty()) return;
  if (bc_names.empty()) return;

  auto* stk_disc = dynamic_cast<Albany::STKDiscretization*>(disc.get());
  if (stk_disc == nullptr) return;

  // ACE Sequential Thermo-Mechanical: this method runs twice for ACE sub-
  // apps — once early from the shared-mesh ctor (when the discretization's
  // nodesets and overlap vector space are NOT yet ready, so parsing returns
  // nothing useful) and once later from finalizePostCommit (when they are).
  // The early-call exit below avoids wasting a parse pass; the late call
  // does the real work but takes the injection-only path: dbc_state_.descriptors
  // are populated for overlap-slot writes but the disc's vector space stays
  // full-size so element death can keep mutating nodeset membership without
  // invalidating a frozen reduced space.
  bool const is_ace = problemParams->isParameter("ACE Sequential Thermomechanical") &&
                      problemParams->get<bool>("ACE Sequential Thermomechanical");
  if (is_ace && deferred_post_commit_pending_) {
    dbc_state_.mode = DBCEliminationState::Mode::DeferredAce;
    return;
  }

  // Use overlap-scope nodesets (includes ghosted nodes) so each rank has
  // descriptors for every constrained DOF in its overlap — not just those it
  // owns. Without this, a rank that ghosts a constrained DOF would skip
  // injecting the BC value into its overlap slot, corrupting assembly.
  auto const& ns_gids_map   = stk_disc->getNodeSetOverlapGIDs();
  auto const& ns_coords_map = stk_disc->getNodeSetOverlapCoords();

  bool const has_dbc_params = problemParams->isSublist("Dirichlet BCs");
  if (!has_dbc_params) return;
  auto& bc_params = problemParams->sublist("Dirichlet BCs");

  // Step 1: Build per-(node_set, eq) descriptors for every constrained DOF.
  // The canonical (and only) key form is
  //
  //   "DBC on NS <nodeset> for DOF <dof>": <value>
  //
  // dispatched on the VALUE type:
  //   double  → Constant
  //   string  → Expression in x, y, z, t (STK expreval)
  //   sublist → named function, selected by the required "BC Function" entry:
  //     "Array"                     → piecewise-linear in time ("Time Values"/
  //                                   "BC Values" inline, or "Time File"/"BC File")
  //     "Schwarz"                   → value from the coupled subdomain; only on
  //                                   the pseudo-DOF "all" (covers every equation)
  //     "Kfield"                    → Williams crack-tip solution; only on the
  //                                   pseudo-DOF "K" (writes X and Y)
  //     "Equilibrium Concentration" → concentration from the local pressure DOF
  //
  // Every entry in the "Dirichlet BCs" list must match this form. Anything
  // else — including the retired spellings "SDBC on NS", "Time Dependent
  // DBC/SDBC on NS", "ExpressionEvaluated SDBC on NS", "Pressure Dependent
  // DBC on NS", and the pseudo-DOFs "StrongSchwarz" and "twist" — is
  // rejected here, before parsing.
  {
    std::set<std::string> valid_keys;
    for (auto const& ns : node_set_ids) {
      for (auto const& dof : bc_names) valid_keys.insert("DBC on NS " + ns + " for DOF " + dof);
      valid_keys.insert("DBC on NS " + ns + " for DOF K");
      valid_keys.insert("DBC on NS " + ns + " for DOF all");
    }
    std::string bad;
    for (auto it = bc_params.begin(); it != bc_params.end(); ++it) {
      std::string const& key = bc_params.name(it);
      if (valid_keys.count(key) == 0) bad += "\n  " + key;
    }
    ALBANY_ASSERT(
        bad.empty(),
        "Unrecognized Dirichlet BC key(s):" + bad +
            "\nThe only accepted form is \"DBC on NS <nodeset> for DOF <dof>\", where "
            "<nodeset> is a mesh node set and <dof> is one of the problem's DBC names "
            "(or the pseudo-DOFs \"K\" for Kfield and \"all\" for Schwarz), with a "
            "double (constant), string (expression in x, y, z, t), or sublist "
            "(BC Function: Array | Schwarz | Kfield | Equilibrium Concentration) value.");
  }
  std::map<GO, DBCDescriptor> local_gid_desc_map;

  for (std::size_t i = 0; i < node_set_ids.size(); ++i) {
    auto gid_it = ns_gids_map.find(node_set_ids[i]);
    if (gid_it == ns_gids_map.end()) continue;
    auto const& node_gids = gid_it->second;

    // Node-set coordinates parallel to node_gids (may be absent for this ns).
    std::vector<double*> const* ns_coords = nullptr;
    auto coord_it = ns_coords_map.find(node_set_ids[i]);
    if (coord_it != ns_coords_map.end()) ns_coords = &coord_it->second;

    // Iterate every equation in bc_names and check each (ns, eq) pair
    // against the standard DBC YAML key forms.
    for (int eq = 0; eq < static_cast<int>(bc_names.size()); ++eq) {
      std::string const& dof = bc_names[eq];
      std::string const& ns  = node_set_ids[i];

      // Build a descriptor prototype for this (ns, dof) pair.
      DBCDescriptor proto;

      std::string const dbc_key = "DBC on NS " + ns + " for DOF " + dof;

      if (bc_params.isType<double>(dbc_key)) {
        proto.kind     = DBCDescriptor::Kind::Constant;
        proto.constant = bc_params.get<double>(dbc_key);
      } else if (bc_params.isType<std::string>(dbc_key)) {
        proto.kind     = DBCDescriptor::Kind::Expression;
        proto.expr_str = bc_params.get<std::string>(dbc_key);
        // Coordinates are per-node and filled in the loop below.
      } else if (bc_params.isSublist(dbc_key)) {
        auto& sub = bc_params.sublist(dbc_key);
        ALBANY_ASSERT(
            sub.isType<std::string>("BC Function"),
            "Dirichlet BC \"" + dbc_key + "\": a sublist value requires a \"BC Function\" string entry "
            "(Array | Schwarz | Kfield | Equilibrium Concentration).");
        std::string const bc_fn = sub.get<std::string>("BC Function");
        if (bc_fn == "Array") {
          proto.kind = DBCDescriptor::Kind::TimeArray;
          // Accept either inline "Time Values"/"BC Values" arrays or file-backed
          // "Time File"/"BC File" (whitespace-separated array text).
          auto read_array = [](Teuchos::ParameterList& list, char const* values_key, char const* file_key) {
            if (list.isParameter(file_key)) {
              auto const filename = list.get<std::string>(file_key);
              std::ifstream file(filename);
              ALBANY_ASSERT(file.good(), "Error opening " + std::string(file_key) + ": " + filename);
              std::stringstream buffer;
              buffer << file.rdbuf();
              std::istringstream iss(buffer.str());
              Teuchos::Array<double> a;
              iss >> a;
              return a;
            }
            return list.get<Teuchos::Array<double>>(values_key);
          };
          auto tv      = read_array(sub, "Time Values", "Time File");
          auto bv      = read_array(sub, "BC Values",   "BC File");
          proto.times  = tv.toVector();
          proto.values = bv.toVector();
        } else if (bc_fn == "Equilibrium Concentration") {
          // Coupled BC: the concentration at each node depends on the local
          // pressure DOF ("TAU") through c_bc = applied * exp(pressure_factor * P).
          // Injection reads the pressure from the overlap solution.
          int p_eq = -1;
          for (std::size_t e = 0; e < bc_names.size(); ++e) {
            if (bc_names[e] == "TAU") p_eq = static_cast<int>(e);
          }
          ALBANY_ASSERT(
              p_eq >= 0,
              "Dirichlet BC \"" + dbc_key + "\": BC Function \"Equilibrium Concentration\" requires "
              "a pressure DOF named \"TAU\" in this problem.");
          proto.kind                   = DBCDescriptor::Kind::EquilibriumConcentration;
          proto.eqconc_applied         = sub.get<double>("Applied Concentration");
          proto.eqconc_pressure_factor = sub.get<double>("Pressure Factor");
          // Overlap LID of the pressure DOF at this node — resolved at
          // descriptor finalization by adding (p_eq - eq) to the
          // concentration overlap LID. Stride is +/- per DOF here:
          proto.eqconc_pressure_overlap_lid = p_eq - eq;
        } else {
          ALBANY_ABORT(
              "Dirichlet BC \"" + dbc_key + "\": unrecognized BC Function \"" + bc_fn +
              "\". Valid on a real DOF: Array | Equilibrium Concentration. "
              "Schwarz goes on the pseudo-DOF \"all\"; Kfield on the pseudo-DOF \"K\".");
        }
      } else if (bc_params.isParameter(dbc_key)) {
        ALBANY_ABORT(
            "Dirichlet BC \"" + dbc_key + "\": value must be a double (constant), a string "
            "(expression in x, y, z, t), or a sublist with a \"BC Function\" entry.");
      } else {
        // No matching BC entry — this node set / DOF has no DBC.
        continue;
      }

      for (std::size_t ni = 0; ni < node_gids.size(); ++ni) {
        GO const dof_gid = stk_disc->getGlobalDOF(node_gids[ni], eq);
        DBCDescriptor desc = proto;
        if (proto.kind == DBCDescriptor::Kind::Expression && ns_coords != nullptr) {
          double* c = (*ns_coords)[ni];
          desc.x = c[0];
          desc.y = c[1];
          desc.z = (spatial_dimension > 2) ? c[2] : 0.0;
        }
        local_gid_desc_map[dof_gid] = desc;
      }
    }

    std::string const& ns_id = node_set_ids[i];

    // Kfield BC on the pseudo-DOF "K": plane-strain crack-tip Williams'
    // solution on a single nodeset; writes both x and y displacement DOFs.
    // YAML key form:
    //   "DBC on NS <ns> for DOF K":
    //     BC Function: Kfield
    //     Time Values: [...]
    //     KI Values:   [...]
    //     KII Values:  [...]
    //     Kfield KI:      <double>
    //     Kfield KII:     <double>
    //     Shear Modulus:  <double>
    //     Poissons Ratio: <double>
    std::string const kfield_key = "DBC on NS " + ns_id + " for DOF K";
    if (bc_params.isSublist(kfield_key) && ns_coords != nullptr) {
      auto const& k_sub = bc_params.sublist(kfield_key);
      ALBANY_ASSERT(
          k_sub.isType<std::string>("BC Function") && k_sub.get<std::string>("BC Function") == "Kfield",
          "Dirichlet BC \"" + kfield_key + "\": the pseudo-DOF \"K\" requires \"BC Function: Kfield\".");
      auto const& ttv   = k_sub.get<Teuchos::Array<double>>("Time Values");
      auto const& kiv   = k_sub.get<Teuchos::Array<double>>("KI Values");
      auto const& kiiv  = k_sub.get<Teuchos::Array<double>>("KII Values");
      double const mu        = k_sub.get<double>("Shear Modulus");
      double const nu        = k_sub.get<double>("Poissons Ratio");
      double const ki_scale  = k_sub.get<double>("Kfield KI");
      double const kii_scale = k_sub.get<double>("Kfield KII");
      for (std::size_t ni = 0; ni < node_gids.size(); ++ni) {
        double* c = (*ns_coords)[ni];
        double const r     = std::sqrt(c[0] * c[0] + c[1] * c[1]);
        double const th    = std::atan2(c[1], c[0]);
        for (int comp = 0; comp < 2; ++comp) {
          GO const      dof_gid = stk_disc->getGlobalDOF(node_gids[ni], comp);
          DBCDescriptor desc;
          desc.kind               = DBCDescriptor::Kind::Kfield;
          desc.kfield_time_values = ttv.toVector();
          desc.kfield_ki_values   = kiv.toVector();
          desc.kfield_kii_values  = kiiv.toVector();
          desc.kfield_mu          = mu;
          desc.kfield_nu          = nu;
          desc.kfield_ki_scale    = ki_scale;
          desc.kfield_kii_scale   = kii_scale;
          desc.kfield_r           = r;
          desc.kfield_theta       = th;
          desc.kfield_component   = comp;
          local_gid_desc_map[dof_gid] = desc;
        }
      }
    }

    // Schwarz BC on the pseudo-DOF "all": one sublist entry covers every
    // equation on the nodeset; values come from the coupled subdomain's
    // overlap solution via DTK transfer. YAML key form:
    //   "DBC on NS <ns> for DOF all":
    //     BC Function: Schwarz
    //     Coupled Application: <name>
    std::string const schwarz_key = "DBC on NS " + ns_id + " for DOF all";
    if (bc_params.isSublist(schwarz_key)) {
      auto const& schwarz_sub = bc_params.sublist(schwarz_key);
      ALBANY_ASSERT(
          schwarz_sub.isType<std::string>("BC Function") && schwarz_sub.get<std::string>("BC Function") == "Schwarz",
          "Dirichlet BC \"" + schwarz_key + "\": the pseudo-DOF \"all\" requires \"BC Function: Schwarz\".");
      std::string const coupled_name = schwarz_sub.get<std::string>("Coupled Application");
      // Record the coupling for the collective DTK transfer driver. Every
      // rank reads the same YAML, so every rank registers the same coupling
      // here — even ranks whose local mesh slice ghosts no Schwarz DOFs. This
      // keeps Albany::computeSchwarzTransferDTK called consistently across
      // ranks (DTK MapOperator::setup/apply is collective on the MPI comm).
      dbc_state_.schwarz_couplings.emplace_back(coupled_name, ns_id);
      int const neq_schwarz = static_cast<int>(bc_names.size());
      for (int eq = 0; eq < neq_schwarz; ++eq) {
        for (std::size_t ni = 0; ni < node_gids.size(); ++ni) {
          GO const      dof_gid = stk_disc->getGlobalDOF(node_gids[ni], eq);
          DBCDescriptor desc;
          desc.kind                     = DBCDescriptor::Kind::Schwarz;
          desc.schwarz_coupled_app_name = coupled_name;
          desc.schwarz_nodeset_id       = ns_id;
          desc.schwarz_ns_node_idx      = static_cast<int>(ni);
          desc.schwarz_eq               = eq;
          local_gid_desc_map[dof_gid]   = desc;
        }
      }
    }
  }

  // Step 2: All-gather GIDs (and a sentinel initial value) so every process
  // knows the full constrained set for graph filtering.
  // Descriptor data stays local — each rank owns its overlap slice.
  // NOTE: every rank must participate in the all-gather + the subsequent
  // setConstrainedDOFs collective, even ranks with an empty local map. A rank
  // may own zero constrained DOFs in its overlap (e.g. nodeset is entirely on
  // another rank) while other ranks have some; an early return here would
  // deadlock the MPI collectives below.
  std::map<GO, DBCDescriptor> global_gid_desc_map = local_gid_desc_map;
  int const num_procs = comm->getSize();
  if (num_procs > 1) {
    // Gather just the GIDs; each rank will fill its own descriptors from the
    // local map for overlap LIDs it owns.
    int const local_count = static_cast<int>(local_gid_desc_map.size());
    std::vector<GO> local_gids;
    local_gids.reserve(local_count);
    for (auto const& [gid, desc] : local_gid_desc_map) local_gids.push_back(gid);

    std::vector<int> counts(num_procs);
    Teuchos::gatherAll(*comm, 1, &local_count, num_procs, counts.data());
    std::vector<int> displs(num_procs, 0);
    for (int i = 1; i < num_procs; ++i) displs[i] = displs[i - 1] + counts[i - 1];
    int const total = displs[num_procs - 1] + counts[num_procs - 1];

    std::vector<GO> all_gids(total);
    auto const*     mpi_comm = dynamic_cast<Teuchos::MpiComm<int> const*>(comm.get());
    MPI_Allgatherv(
        local_gids.data(), local_count, MPI_LONG_LONG,
        all_gids.data(), counts.data(), displs.data(), MPI_LONG_LONG,
        *mpi_comm->getRawMpiComm());

    for (GO gid : all_gids) {
      if (global_gid_desc_map.find(gid) == global_gid_desc_map.end())
        global_gid_desc_map[gid] = DBCDescriptor{};  // placeholder — not on this rank
    }
  }

  // ACE: defer the entire parse + injection-LID build to finalizePostCommit,
  // which fires after disc->updateMesh has populated the nodesets and made
  // the overlap vector space concrete. The early entry into this function
  // (from the shared-mesh ctor) sees empty nodesets, so nothing useful would
  // be parsed here anyway.

  // Pre-elimination owned vector space. Captured here so the Reduced branch
  // below can hand it to expandToFullSolution; in FullyConstrained mode the
  // disc isn't reduced and we clear it back to null before returning.
  auto const pre_elim_owned_vs = disc->getVectorSpace();
  dbc_state_.full_owned_vs     = pre_elim_owned_vs;

  // Fully-constrained mesh: 0 free DOFs. The linear solver can't run with an
  // empty block size, but the test still has work to do — the residual fill
  // must update internal state from the BC-prescribed u, and the output
  // expansion must read u_bc into the constrained slots. Take the injection-
  // only path: build descriptors, leave the disc's vector space intact (so
  // NOX still has owned space = full space and converges on the first iter
  // with u = u_bc), and skip the disc-level reduction.
  {
    GO const n_constrained_global = static_cast<GO>(global_gid_desc_map.size());
    GO const n_owned_global       = pre_elim_owned_vs->dim();
    if (n_constrained_global >= n_owned_global) {
      // ACE doesn't get the fully-constrained injection-only path:
      // sub-apps already have their own injection-only handling, and ACE's
      // transient dynamics make the SDirichlet residual rewrite (which
      // assumes the residual is purely (u - u_bc)) wrong for the thermal
      // sub-app's dx/dt-coupled rows.
      if (is_ace) {
        dbc_state_.full_owned_vs = Teuchos::null;
        return;
      }
      auto const overlap_vs         = disc->getOverlapVectorSpace();
      auto const overlap_vs_indexer = Albany::createGlobalLocalIndexer(overlap_vs);
      auto const owned_vs_indexer   = Albany::createGlobalLocalIndexer(pre_elim_owned_vs);
      for (auto const& [gid, desc] : global_gid_desc_map) {
        LO const overlap_lid = overlap_vs_indexer->getLocalElement(gid);
        if (overlap_lid >= 0 && local_gid_desc_map.count(gid)) {
          DBCDescriptor d  = local_gid_desc_map.at(gid);
          d.overlap_lid    = overlap_lid;
          d.full_owned_lid = owned_vs_indexer->getLocalElement(gid);
          dbc_state_.descriptors.push_back(d);
        }
      }
      // FullyConstrained injection-only path: NOX sees the full disc-owned
      // space (no reduction), and after each fill we overwrite f and the
      // Jacobian rows at constrained owned LIDs so NOX converges in one
      // iteration with u = u_bc. injectConstrainedDOFValues still writes
      // u_bc into the overlap each fill so the material model sees the
      // prescribed state.
      dbc_state_.mode          = DBCEliminationState::Mode::FullyConstrained;
      dbc_state_.full_owned_vs = Teuchos::null;
      return;
    }
  }

  // Step 3: Build the constrained GID set for the discretization and
  // populate dbc_state_.descriptors for overlap-slot injection.
  std::set<GO> constrained_gids;
  // For setConstrainedDOFs we still need a GID→initial_value map (used to
  // initialize the reduced solution on first scatter); use t=0 evaluation.
  std::map<GO, double> gid_value_map;
  auto const   overlap_vs           = disc->getOverlapVectorSpace();
  auto const   overlap_vs_indexer   = Albany::createGlobalLocalIndexer(overlap_vs);
  auto const   full_owned_vs_indexer = Albany::createGlobalLocalIndexer(dbc_state_.full_owned_vs);

  for (auto const& [gid, desc] : global_gid_desc_map) {
    constrained_gids.insert(gid);
    gid_value_map[gid] = desc.eval(0.0);
    LO const overlap_lid = overlap_vs_indexer->getLocalElement(gid);
    if (overlap_lid >= 0 && local_gid_desc_map.count(gid)) {
      DBCDescriptor d   = local_gid_desc_map.at(gid);
      d.overlap_lid     = overlap_lid;
      d.full_owned_lid  = full_owned_vs_indexer->getLocalElement(gid);
      // For Equilibrium Concentration BCs the parser stored a stride
      // (delta-eq from the concentration DOF to the pressure DOF); now that
      // we know overlap_lid we can resolve it to the pressure DOF's overlap
      // LID. Overlap LIDs are laid out so DOFs at the same node are stride
      // 1 apart per equation (LID = node_lid * neq + eq).
      if (d.kind == DBCDescriptor::Kind::EquilibriumConcentration) {
        d.eqconc_pressure_overlap_lid = overlap_lid + d.eqconc_pressure_overlap_lid;
      }
      dbc_state_.descriptors.push_back(d);
    }
  }

  disc->setConstrainedDOFs(constrained_gids, gid_value_map);
  dbc_state_.mode = DBCEliminationState::Mode::Reduced;

  // Retranslate nodeSets LIDs to the new (post-elimination) owned vector space.
  // Eliminated DOFs get LID = -1; free DOFs get their new contiguous LID.
  // This keeps workset.nodeSets consistent for all BC evaluators.
  {
    auto const new_vs          = disc->getVectorSpace();
    auto const new_vs_indexer  = Albany::createGlobalLocalIndexer(new_vs);
    auto const full_vs_indexer = Albany::createGlobalLocalIndexer(dbc_state_.full_owned_vs);
    auto&      ns_map          = disc->getNodeSets();
    for (auto& [ns_id, ns_nodes] : ns_map) {
      for (auto& dofs : ns_nodes) {
        for (auto& lid : dofs) {
          GO const gid     = full_vs_indexer->getGlobalElement(lid);
          LO const new_lid = new_vs_indexer->getLocalElement(gid);
          lid = (new_lid >= 0) ? static_cast<int>(new_lid) : -1;
        }
      }
    }
  }
}

void
Application::injectConstrainedDOFValues(double time, bool fill_has_xdot, bool fill_has_xdotdot)
{
  // Gate on the GLOBAL elimination state, not the per-rank descriptor count.
  // A rank may legitimately have an empty dbc_state_.descriptors in parallel (e.g.
  // when all constrained DOFs are owned/ghosted by other ranks) while other
  // ranks need it to participate in the Schwarz DTK collectives below. An
  // early return here on emptiness alone deadlocks the parallel solve.
  if (dbc_state_.full_owned_vs.is_null() && dbc_state_.schwarz_couplings.empty()) return;

  last_transient_time_ = time;

  // Lazy-init Schwarz descriptors: resolve coupled-app name → app index.
  // No coordinate matching is needed because the DTK transfer below does
  // spatial interpolation — the Schwarz interface meshes can be non-aligned.
  if (!apps_.is_null() && !app_name_index_map_.is_null()) {
    for (auto& desc : dbc_state_.descriptors) {
      if (desc.kind != DBCDescriptor::Kind::Schwarz || desc.schwarz_initialized) continue;
      auto const name_it = app_name_index_map_->find(desc.schwarz_coupled_app_name);
      if (name_it == app_name_index_map_->end()) continue;
      int const coupled_idx = name_it->second;
      if (apps_[coupled_idx] == Teuchos::null) continue;
      desc.schwarz_coupled_app_idx = coupled_idx;
      desc.schwarz_initialized     = true;
    }
  }

  // Refresh Schwarz caches per (coupled_app, nodeset) pair via DTK transfer.
  // The returned MV array has one entry per time-derivative slot (length =
  // num_time_deriv + 1). Each MV is backed by this app's solution_field_dtk
  // STK field, so getData(eq)[overlap_node_lid] reads the interpolated value
  // at this nodeset's local node identified by its overlap node LID.
  // Recompute every call: the coupled-app state changes both across Schwarz
  // iterations (subdomain solves alternate) and across time steps.
  int const neq = getNumEquations();
  std::map<std::pair<int, std::string>,
           Teuchos::Array<Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>>>
      schwarz_xfer;
  // Drive DTK calls from the rank-invariant dbc_state_.schwarz_couplings set so that
  // every MPI rank participates in computeSchwarzTransferDTK for every
  // (coupled_app, nodeset) pair, even ranks with no local descriptor on that
  // pair. DTK::MapOperator::setup is a collective on the MPI comm; asymmetric
  // participation across ranks deadlocks the parallel solve.
  if (!apps_.is_null() && !app_name_index_map_.is_null()) {
    for (auto const& [coupled_name, ns_id] : dbc_state_.schwarz_couplings) {
      auto const name_it = app_name_index_map_->find(coupled_name);
      if (name_it == app_name_index_map_->end()) continue;
      int const coupled_idx = name_it->second;
      if (apps_[coupled_idx] == Teuchos::null) continue;
      auto const& coupled_app   = *apps_[coupled_idx];
      auto const  key           = std::make_pair(coupled_idx, ns_id);
      schwarz_xfer[key]         = Albany::computeSchwarzTransferDTK(*this, coupled_app, ns_id);
    }
  }
  for (auto& desc : dbc_state_.descriptors) {
    if (desc.kind != DBCDescriptor::Kind::Schwarz || !desc.schwarz_initialized) continue;
    auto const key = std::make_pair(desc.schwarz_coupled_app_idx, desc.schwarz_nodeset_id);
    auto       it  = schwarz_xfer.find(key);
    if (it == schwarz_xfer.end()) continue;
    auto const& xfer = it->second;
    // The DTK target MV is created from this app's solution_field_dtk STK
    // field via createFieldMultiVector. Its row LID space matches the overlap
    // node LID space (StrongSchwarzBC's weak-path consumer relies on the same
    // x_dof / neq indexing, which empirically works).
    LO const node_lid         = desc.overlap_lid / neq;
    desc.schwarz_cached_value = xfer[0]->getData(desc.schwarz_eq)[node_lid];
    if (xfer.size() > 1) {
      desc.schwarz_cached_velocity = xfer[1]->getData(desc.schwarz_eq)[node_lid];
    }
    if (xfer.size() > 2) {
      desc.schwarz_cached_acceleration = xfer[2]->getData(desc.schwarz_eq)[node_lid];
    }
  }

  // Inject u, v, a into the constrained overlap slots. The reduced-space
  // scatter cannot reach these slots (their GIDs don't exist in the reduced
  // source); without this they would hold stale values and corrupt the
  // mass/inertia contributions in element residuals that touch constrained
  // nodes. Idempotent during Newton iterations within a step (eval and
  // derivs_at depend only on `time`; Schwarz returns the cached values set
  // above on the most recent step boundary).
  // Keep the column RCPs alive for the duration of the inject loop —
  // ArrayRCPs returned by getNonconstLocalData are valid only as long as
  // the underlying Thyra_Vector temporary lives.
  auto       overlapped_MV   = solMgr->getOverlappedSolution();
  auto       x_overlap       = overlapped_MV->col(0);
  auto const x_data          = Albany::getNonconstLocalData(x_overlap);
  Teuchos::RCP<Thyra_Vector> xdot_overlap;
  Teuchos::RCP<Thyra_Vector> xdotdot_overlap;
  Teuchos::ArrayRCP<ST>      xdot_data;
  Teuchos::ArrayRCP<ST>      xdotdot_data;
  if (num_time_deriv >= 1) {
    xdot_overlap = overlapped_MV->col(1);
    xdot_data    = Albany::getNonconstLocalData(xdot_overlap);
  }
  if (num_time_deriv >= 2) {
    xdotdot_overlap = overlapped_MV->col(2);
    xdotdot_data    = Albany::getNonconstLocalData(xdotdot_overlap);
  }

  // Build per-constrained-GID u, v, a maps for the discretization to inject
  // when expanding the reduced owned solution back out to the overlap STK
  // fields at Exodus output time. EVERY local (overlap-scope) descriptor
  // participates, not just owned ones: the disc-side expand-and-inject
  // loops patch the OVERLAP vector, and a rank's SHARED copy of a
  // constrained node otherwise stays at the import's zero (decomposed
  // meshes carry node-set membership only on some ranks, so the owner
  // cannot be assumed to be the only rank writing the node to Exodus --
  // epu may take any piece's copy of a shared node).
  std::map<GO, double> u_map;
  std::map<GO, double> v_map;
  std::map<GO, double> a_map;
  auto const overlap_inject_indexer = Albany::createGlobalLocalIndexer(disc->getOverlapVectorSpace());
  auto const full_owned_vs_indexer = dbc_state_.full_owned_vs.is_null() ? Teuchos::null : Albany::createGlobalLocalIndexer(dbc_state_.full_owned_vs);

  for (auto const& desc : dbc_state_.descriptors) {
    auto const derivs = desc.derivs_at(time);
    double     u      = desc.eval(time);
    // EquilibriumConcentration is the one BC kind whose value depends on
    // another DOF at the same node: c_bc = applied * exp(pressure_factor *
    // P_overlap). Read the pressure from the overlap solution here (after
    // scatterX has populated it for free DOFs) and compute c_bc inline.
    if (desc.kind == DBCDescriptor::Kind::EquilibriumConcentration) {
      double const pressure   = x_data[desc.eqconc_pressure_overlap_lid];
      u                       = desc.eqconc_applied * std::exp(desc.eqconc_pressure_factor * pressure);
      desc.eqconc_cached_value = u;
    }
    x_data[desc.overlap_lid] = u;
    // BC rates belong in the overlap xdot only when the fill itself
    // carries xdot (implicit dynamic fills, where the residual's
    // consistent-mass term M_fc * xdot_bc is correct). For fills without
    // xdot -- in particular the explicit lumped-mass right-hand-side
    // evaluation, which Albany::ModelEvaluator routes here with x_dot
    // null -- write zero, both for consistency with the fill's xdot = 0
    // and to clear any stale values in the overlap slot.
    if (num_time_deriv >= 1) {
      xdot_data[desc.overlap_lid] = fill_has_xdot ? derivs.v : 0.0;
    }
    if (num_time_deriv >= 2) {
      xdotdot_data[desc.overlap_lid] = fill_has_xdotdot ? derivs.a : 0.0;
    }
    if (desc.overlap_lid >= 0) {
      GO const gid = overlap_inject_indexer->getGlobalElement(desc.overlap_lid);
      u_map[gid]   = u;
      if (num_time_deriv >= 1) v_map[gid] = derivs.v;
      if (num_time_deriv >= 2) a_map[gid] = derivs.a;
    }
  }

  if (!dbc_state_.full_owned_vs.is_null()) {
    disc->setConstrainedDOFValues(u_map, v_map, a_map);
  }
}

Teuchos::RCP<Thyra_Vector const>
Application::expandToFullSolution(Teuchos::RCP<Thyra_Vector const> const& x, double time)
{
  // Check on dbc_state_.full_owned_vs (not dbc_state_.descriptors) — elimination is a global
  // property of the application. A rank that owns zero constrained DOFs in its
  // overlap still has an empty dbc_state_.descriptors, but it must still return the
  // FULL (pre-elimination) owned vector like other ranks, because response
  // functions invoke collectives (e.g. Allreduce in one->dot(x)) that deadlock
  // if ranks use different-sized vector spaces.
  if (dbc_state_.full_owned_vs.is_null()) return x;

  // Build the full (pre-elimination) owned vector directly without going through
  // the overlap space. dbc_state_.full_owned_vs is 1-to-1 (same as disc->getVectorSpace()
  // pre-elimination), so Tpetra Import from reduced→full_owned routes each free
  // GID to exactly one target rank. Constrained slots stay at 0 after the import
  // (their GIDs don't exist in the reduced source) and are then filled in per
  // rank from each descriptor's cached full_owned_lid.
  auto full_x = Thyra::createMember(dbc_state_.full_owned_vs);
  full_x->assign(0.0);
  auto cas_red_to_full =
      Albany::createCombineAndScatterManager(disc->getVectorSpace(), dbc_state_.full_owned_vs);
  cas_red_to_full->scatter(*x, *full_x, Albany::CombineMode::INSERT);

  auto full_data = Albany::getNonconstLocalData(full_x);
  for (auto const& desc : dbc_state_.descriptors) {
    if (desc.full_owned_lid >= 0) {
      full_data[desc.full_owned_lid] = desc.eval(time);
    }
  }
  return full_x;
}

void
Application::setScaling(const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  // get info from Scaling parameter list (for scaling Jacobian/residual)
  RCP<Teuchos::ParameterList> scalingParams = Teuchos::sublist(params, "Scaling", true);
  scale                                     = scalingParams->get<double>("Scale", 0.0);
  scaleBCdofs                               = scalingParams->get<bool>("Scale BC Dofs", false);
  std::string scaleType                     = scalingParams->get<std::string>("Type", "Constant");

  if (scale == 0.0) {
    scale = 1.0;
  }

  if (scaleType == "Constant") {
    scale_type = CONSTANT;
  } else if (scaleType == "Diagonal") {
    scale_type = DIAG;
    scale      = 1.0e1;
  } else if (scaleType == "Abs Row Sum") {
    scale_type = ABSROWSUM;
    scale      = 1.0e1;
  } else {
    ALBANY_ABORT(
        "The scaling Type you selected " << scaleType << " is not supported!"
                                         << "Supported scaling Types are currently: Constant" << std::endl);
  }

  if (scale == 1.0) scaleBCdofs = false;
}

void
Application::finalSetUp(const Teuchos::RCP<Teuchos::ParameterList>& params, Teuchos::RCP<Thyra_Vector const> const& initial_guess)
{
  setScaling(params);

  // Now that space is allocated in STK for state fields, initialize states.
  // If the states have been already allocated, skip this.
  if (!stateMgr.areStateVarsAllocated()) stateMgr.setupStateArrays(disc);

  solMgr = rcp(new AAdapt::AdaptiveSolutionManager(
      params,
      initial_guess,
      paramLib,
      stateMgr,
      // Prevent a circular dependency.
      Teuchos::rcp(rc_mgr.get(), false),
      comm));
  if (Teuchos::nonnull(rc_mgr)) rc_mgr->setSolutionManager(solMgr);

  // Create Distributed parameters and initialize them with data stored in the
  // mesh.
  const StateInfoStruct& distParamSIS = disc->getNodalParameterSIS();
  for (size_t is = 0; is < distParamSIS.size(); is++) {
    // Get name of distributed parameter
    std::string const& param_name = distParamSIS[is]->name;

    // Get parameter vector spaces and build parameter vector
    // Create distributed parameter and set workset_elem_dofs
    Teuchos::RCP<DistributedParameter> parameter(
        new DistributedParameter(param_name, disc->getVectorSpace(param_name), disc->getOverlapVectorSpace(param_name)));
    parameter->set_workset_elem_dofs(Teuchos::rcpFromRef(disc->getElNodeEqID(param_name)));

    // Get the vector and lower/upper bounds, and fill them with available
    // data
    Teuchos::RCP<Thyra_Vector> dist_param            = parameter->vector();
    Teuchos::RCP<Thyra_Vector> dist_param_lowerbound = parameter->lower_bounds_vector();
    Teuchos::RCP<Thyra_Vector> dist_param_upperbound = parameter->upper_bounds_vector();

    std::stringstream lowerbound_name, upperbound_name;
    lowerbound_name << param_name << "_lowerbound";
    upperbound_name << param_name << "_upperbound";

    // Initialize parameter with data stored in the mesh
    disc->getField(*dist_param, param_name);
    const auto& nodal_param_states = disc->getNodalParameterSIS();
    bool        has_lowerbound(false), has_upperbound(false);
    for (int ist = 0; ist < static_cast<int>(nodal_param_states.size()); ist++) {
      has_lowerbound = has_lowerbound || (nodal_param_states[ist]->name == lowerbound_name.str());
      has_upperbound = has_upperbound || (nodal_param_states[ist]->name == upperbound_name.str());
    }
    if (has_lowerbound) {
      disc->getField(*dist_param_lowerbound, lowerbound_name.str());
    } else {
      dist_param_lowerbound->assign(std::numeric_limits<ST>::lowest());
    }
    if (has_upperbound) {
      disc->getField(*dist_param_upperbound, upperbound_name.str());
    } else {
      dist_param_upperbound->assign(std::numeric_limits<ST>::max());
    }
    // JR: for now, initialize to constant value from user input if requested.
    // This needs to be generalized.
    if (params->sublist("Problem").isType<Teuchos::ParameterList>("Topology Parameters")) {
      Teuchos::ParameterList& topoParams = params->sublist("Problem").sublist("Topology Parameters");
      if (topoParams.isType<std::string>("Entity Type") && topoParams.isType<double>("Initial Value")) {
        if (topoParams.get<std::string>("Entity Type") == "Distributed Parameter" && topoParams.get<std::string>("Topology Name") == param_name) {
          double initVal = topoParams.get<double>("Initial Value");
          dist_param->assign(initVal);
        }
      }
    }

    // Add parameter to the distributed parameter library
    distParamLib->add(parameter->name(), parameter);
  }

  // Now setup response functions (see note above)
  for (int i = 0; i < responses.size(); i++) {
    responses[i]->setup();
  }

  // Set up memory for workset
  fm = problem->getFieldManager();
  ALBANY_PANIC(fm == Teuchos::null, "getFieldManager not implemented!!!");

  offsets_    = problem->getOffsets();
  nodeSetIDs_ = problem->getNodeSetIDs();

  nfm = problem->getNeumannFieldManager();

  if (comm->getRank() == 0) {
    phxGraphVisDetail   = problemParams->get("Phalanx Graph Visualization Detail", 0);
    stateGraphVisDetail = phxGraphVisDetail;
  }

  *out << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << " Sacado ParameterLibrary has been initialized:\n " << *paramLib << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << std::endl;

  // Allow Problem to add custom NOX status test
  problem->applyProblemSpecificSolverSettings(params);

  ignore_residual_in_jacobian = problemParams->get("Ignore Residual In Jacobian", false);


  is_adjoint = problemParams->get("Solve Adjoint", false);

  // For backward compatibility, use any value at the old location of the
  // "Compute Sensitivity" flag as a default value for the new flag location
  // when the latter has been left undefined
  std::string const              sensitivityToken = "Compute Sensitivities";
  const Teuchos::Ptr<bool const> oldSensitivityFlag(problemParams->getPtr<bool>(sensitivityToken));
  if (Teuchos::nonnull(oldSensitivityFlag)) {
    Teuchos::ParameterList& solveParams = params->sublist("Piro").sublist("Analysis").sublist("Solve");
    solveParams.get(sensitivityToken, *oldSensitivityFlag);
  }

  // MPerego: Preforming post registration setup here to make sure that the
  // discretization is already created, so that  derivative dimensions are
  // known.
  for (int i = 0; i < responses.size(); ++i) {
    responses[i]->postRegSetup();
  }

}

RCP<AbstractDiscretization>
Application::getDiscretization() const
{
  return disc;
}

RCP<AbstractProblem>
Application::getProblem() const
{
  return problem;
}

RCP<Teuchos_Comm const>
Application::getComm() const
{
  return comm;
}


Teuchos::RCP<Thyra_VectorSpace const>
Application::getVectorSpace() const
{
  return disc->getVectorSpace();
}

Teuchos::RCP<Thyra_VectorSpace const>
Application::getFullVectorSpace() const
{
  return dbc_state_.full_owned_vs.is_null() ? disc->getVectorSpace() : dbc_state_.full_owned_vs;
}

RCP<Thyra_LinearOp>
Application::createJacobianOp() const
{
  return disc->createJacobianOp();
}

RCP<Thyra_LinearOp>
Application::getPreconditioner()
{
  return Teuchos::null;
}

RCP<ParamLib>
Application::getParamLib() const
{
  return paramLib;
}

RCP<DistributedParameterLibrary>
Application::getDistributedParameterLibrary() const
{
  return distParamLib;
}

int
Application::getNumResponses() const
{
  return responses.size();
}

Teuchos::RCP<AbstractResponseFunction>
Application::getResponse(int i) const
{
  return responses[i];
}

bool
Application::suppliesPreconditioner() const
{
  return physicsBasedPreconditioner;
}

namespace {
// nfm is an ArrayRCP indexed by physics set in some problems and length-1
// in others. This wrapper hides the difference: most problems hit the
// length-1 case and the wsPhysIndex lookup is the fallback.
inline Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>&
deref_nfm(Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>>& nfm, const WorksetArray<int>::type& wsPhysIndex, int ws)
{
  return nfm.size() == 1 ?          // Currently, all problems seem to have one nfm ...
             nfm[0] :               // ... hence this is the intended behavior ...
             nfm[wsPhysIndex[ws]];  // ... and this is not, but may one day be
                                    // again.
}

// For the perturbation xd,
//     f_i(x + xd) = f_i(x) + J_i(x) xd + O(xd' H_i(x) xd),
// where J_i is the i'th row of the Jacobian matrix and H_i is the Hessian of
// f_i at x. We don't have the Hessian, however, so approximate the last term by
// norm(f) O(xd' xd). We use the inf-norm throughout.
//   For check_lvl >= 1, check that f(x + xd) - f(x) is approximately equal to
// J(x) xd by computing
//     reldif(f(x + dx) - f(x), J(x) dx)
//        = norm(f(x + dx) - f(x) - J(x) dx) /
//          max(norm(f(x + dx) - f(x)), norm(J(x) dx)).
// This norm should be on the order of norm(xd).
//   For check_lvl >= 2, output a multivector in matrix market format having
// columns
//     [x, dx, f(x), f(x + dx) - f(x), f(x + dx) - f(x) - J(x) dx].
//   The purpose of this derivative checker is to help find programming errors
// in the Jacobian. Automatic differentiation largely or entirely prevents math
// errors, but other kinds of programming errors (uninitialized memory,
// accidental omission of a FadType, etc.) can cause errors. The common symptom
// of such an error is that the residual is correct, and therefore so is the
// solution, but convergence to the solution is not quadratic.
//   A complementary method to check for errors in the Jacobian is to use
//     Piro -> Jacobian Operator = Matrix-Free,
//   Enable this check using the debug block:
//     <ParameterList>
//       <ParameterList name="Debug Output">
//         <Parameter name="Derivative Check" type="int" value="1"/>
void
checkDerivatives(
    Application&                              app,
    double const                              time,
    Teuchos::RCP<Thyra_Vector const> const&   x,
    Teuchos::RCP<Thyra_Vector const> const&   xdot,
    Teuchos::RCP<Thyra_Vector const> const&   xdotdot,
    const Teuchos::Array<ParamVec>&           p,
    Teuchos::RCP<Thyra_Vector const> const&   fi,
    const Teuchos::RCP<const Thyra_LinearOp>& jacobian,
    int const                                 check_lvl)
{
  if (check_lvl <= 0) {
    return;
  }

  // Work vectors. x's map is compatible with f's, so don't distinguish among
  // maps in this function.
  Teuchos::RCP<Thyra_Vector> w1 = Thyra::createMember(x->space());
  Teuchos::RCP<Thyra_Vector> w2 = Thyra::createMember(x->space());
  Teuchos::RCP<Thyra_Vector> w3 = Thyra::createMember(x->space());

  Teuchos::RCP<Thyra_MultiVector> mv;
  if (check_lvl > 1) {
    mv = Thyra::createMembers(x->space(), 5);
  }

  // Construct a perturbation.
  double const               delta = 1e-7;
  Teuchos::RCP<Thyra_Vector> xd    = w1;
  xd->randomize(-Teuchos::ScalarTraits<ST>::rmax(), Teuchos::ScalarTraits<ST>::rmax());
  Teuchos::RCP<Thyra_Vector> xpd = w2;
  {
    const Teuchos::ArrayRCP<const RealType> x_d   = getLocalData(x);
    const Teuchos::ArrayRCP<RealType>       xd_d  = getNonconstLocalData(xd);
    const Teuchos::ArrayRCP<RealType>       xpd_d = getNonconstLocalData(xpd);
    for (int i = 0; i < x_d.size(); ++i) {
      xd_d[i] = 2 * xd_d[i] - 1;
      if (x_d[i] == 0) {
        // No scalar-level way to get the magnitude of x_i, so just go with
        // something:
        xd_d[i] = xpd_d[i] = delta * xd_d[i];
      } else {
        // Make the perturbation meaningful relative to the magnitude of x_i.
        xpd_d[i] = (1 + delta * xd_d[i]) * x_d[i];  // mult line
        // Sanitize xd_d.
        xd_d[i] = xpd_d[i] - x_d[i];
        if (xd_d[i] == 0) {
          // Underflow in "mult line" occurred because x_d[i] is something like
          // 1e-314. That's a possible sign of uninitialized memory. However,
          // carry on here to get a good perturbation by reverting to the
          // no-magnitude case:
          xd_d[i] = xpd_d[i] = delta * xd_d[i];
        }
      }
    }
  }
  if (Teuchos::nonnull(mv)) {
    scale_and_update(mv->col(0), 0.0, x, 1.0);
    scale_and_update(mv->col(1), 0.0, xd, 1.0);
  }

  // If necessary, compute f(x).
  Teuchos::RCP<Thyra_Vector const> f;
  if (fi.is_null()) {
    Teuchos::RCP<Thyra_Vector> tmp = Thyra::createMember(x->space());
    app.computeGlobalResidual(time, x, xdot, xdotdot, p, tmp);
    f = tmp;
  } else {
    f = fi;
  }
  if (Teuchos::nonnull(mv)) {
    mv->col(2)->assign(0);
    mv->col(2)->update(1.0, *f);
  }

  // fpd = f(xpd).
  Teuchos::RCP<Thyra_Vector> fpd = w3;
  app.computeGlobalResidual(time, xpd, xdot, xdotdot, p, fpd);

  // fd = fpd - f.
  Teuchos::RCP<Thyra_Vector> fd = fpd;
  scale_and_update(fpd, 1.0, f, -1.0);
  if (Teuchos::nonnull(mv)) {
    scale_and_update(mv->col(3), 0.0, fd, 1.0);
  }

  // Jxd = J xd.
  Teuchos::RCP<Thyra_Vector> Jxd = w2;
  jacobian->apply(Thyra::NOTRANS, *xd, Jxd.ptr(), 1.0, 0.0);

  // Norms.
  const ST fdn  = fd->norm_inf();
  const ST Jxdn = Jxd->norm_inf();
  const ST xdn  = xd->norm_inf();
  // d = norm(fd - Jxd).
  Teuchos::RCP<Thyra_Vector> d = fd;
  scale_and_update(d, 1.0, Jxd, -1.0);
  if (Teuchos::nonnull(mv)) {
    scale_and_update(mv->col(4), 0.0, d, 1.0);
  }
  double const dn = d->norm_inf();

  // Assess.
  double const den = std::max(fdn, Jxdn), e = dn / den;
  *Teuchos::VerboseObjectBase::getDefaultOStream() << "Albany::Application Check Derivatives level " << check_lvl << ":\n"
                                                   << "   reldif(f(x + dx) - f(x), J(x) dx) = " << e << ",\n which should be on the order of " << xdn << "\n";

  if (Teuchos::nonnull(mv)) {
    static int        ctr = 0;
    std::stringstream ss;
    ss << "dc" << ctr << ".mm";
    writeMatrixMarket(mv.getConst(), "dc", ctr);
    ++ctr;
  }
}
}  // namespace

template <>
void
Application::postRegSetup<PHAL::AlbanyTraits::Residual>()
{
  using EvalT = PHAL::AlbanyTraits::Residual;

  std::string evalName = PHAL::evalName<EvalT>("FM", 0);
  if (phxSetup->contain_eval(evalName)) return;

  for (int ps = 0; ps < fm.size(); ps++) {
    evalName = PHAL::evalName<EvalT>("FM", ps);
    phxSetup->insert_eval(evalName);

    fm[ps]->postRegistrationSetupForType<EvalT>(*phxSetup);

    // Update phalanx saved/unsaved fields based on field dependencies
    phxSetup->check_fields(fm[ps]->getFieldTagsForSizing<EvalT>());
    phxSetup->update_fields();

    writePhalanxGraph<EvalT>(fm[ps], evalName, phxGraphVisDetail);
  }
  if (nfm != Teuchos::null)
    for (int ps = 0; ps < nfm.size(); ps++) {
      evalName = PHAL::evalName<EvalT>("NFM", ps);
      phxSetup->insert_eval(evalName);

      nfm[ps]->postRegistrationSetupForType<EvalT>(*phxSetup);

      // Update phalanx saved/unsaved fields based on field dependencies
      phxSetup->check_fields(nfm[ps]->getFieldTagsForSizing<EvalT>());
      phxSetup->update_fields();

      writePhalanxGraph<EvalT>(nfm[ps], evalName, phxGraphVisDetail);
    }
}

template <>
void
Application::postRegSetup<PHAL::AlbanyTraits::Jacobian>()
{
  postRegSetupDImpl<PHAL::AlbanyTraits::Jacobian>();
}

template <typename EvalT>
void
Application::postRegSetupDImpl()
{
  std::string evalName = PHAL::evalName<EvalT>("FM", 0);
  if (phxSetup->contain_eval(evalName)) return;

  for (int ps = 0; ps < fm.size(); ps++) {
    evalName = PHAL::evalName<EvalT>("FM", ps);
    phxSetup->insert_eval(evalName);

    std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(PHAL::getDerivativeDimensions<EvalT>(this, ps));
    fm[ps]->setKokkosExtendedDataTypeDimensions<EvalT>(derivative_dimensions);
    fm[ps]->postRegistrationSetupForType<EvalT>(*phxSetup);

    // Update phalanx saved/unsaved fields based on field dependencies
    phxSetup->check_fields(fm[ps]->getFieldTagsForSizing<EvalT>());
    phxSetup->update_fields();

    writePhalanxGraph<EvalT>(fm[ps], evalName, phxGraphVisDetail);

    if (nfm != Teuchos::null && ps < nfm.size()) {
      evalName = PHAL::evalName<EvalT>("NFM", ps);
      phxSetup->insert_eval(evalName);

      nfm[ps]->setKokkosExtendedDataTypeDimensions<EvalT>(derivative_dimensions);
      nfm[ps]->postRegistrationSetupForType<EvalT>(*phxSetup);

      // Update phalanx saved/unsaved fields based on field dependencies
      phxSetup->check_fields(nfm[ps]->getFieldTagsForSizing<EvalT>());
      phxSetup->update_fields();

      writePhalanxGraph<EvalT>(nfm[ps], evalName, phxGraphVisDetail);
    }
  }
}

template <typename EvalT>
void
Application::writePhalanxGraph(Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>> fm, std::string const& evalName, int const& phxGraphVisDetail)
{
  if (phxGraphVisDetail > 0) {
    bool const detail = (phxGraphVisDetail > 1) ? true : false;
    *out << "Phalanx writing graphviz file for graph of " << evalName << " (detail = " << phxGraphVisDetail << ")" << std::endl;
    std::string const graphName = "phalanxGraph" + evalName;
    *out << "Process using 'dot -Tpng -O " << graphName << std::endl;
    fm->writeGraphvizFile<EvalT>(graphName, detail, detail);

    // Print phalanx setup info
    phxSetup->print(*out);
  }
}

// Element death: fix orphan nodes after global assembly.
// Element death leaves orphan nodes -- nodes connected only to dead
// (deactivated) elements. No active element contributes to an orphan
// DOF, so its row in any assembled operator is entirely zero. A zero
// row makes the operator singular, which breaks the linear solves and,
// for the mass matrix that Piro::InvertMassMatrixDecorator lumps for
// the explicit thermal solve, causes a divide-by-zero.
//
// This regularizes every zero-diagonal row: it zeroes the row and sets
// the diagonal to a representative magnitude, decoupling the orphan DOF
// (and giving a finite 1/m for a lumped mass matrix). Detecting orphans
// straight from the assembled operator -- rather than from a per-app
// cell-death snapshot -- keeps the fix correct no matter how erosion
// has rebuilt the worksets, and needs no death bookkeeping that could
// go stale across a mid-solve workset rebuild.
//
// The residual needs no companion fix: an orphan node belongs to no
// active-element workset, so nothing scatters into its residual entry
// and it is already zero.
void
Application::fixOrphanNodesForElementDeath(Teuchos::RCP<Thyra_LinearOp> jac)
{
  if (Teuchos::is_null(jac)) return;

  // The operator arrives fill-complete (just assembled and combined).
  // Read the diagonal and the orphan rows' column layouts in that
  // state -- getDiagonalCopy / getLocalRowValues are only valid on a
  // fill-complete matrix -- then re-open it for value modification.
  // The caller fill-completes the operator afterward, so this routine
  // leaves it fill-active, matching the Dirichlet-BC code path.
  Teuchos::RCP<Thyra_Vector> diag;
  Albany::getDiagonalCopy(jac, diag);
  Teuchos::ArrayRCP<const ST> const diag_data = Albany::getLocalData(diag.getConst());
  LO const                          num_rows  = static_cast<LO>(diag_data.size());

  // Representative diagonal magnitude from the non-orphan rows.
  ST rep_sum   = 0.0;
  LO rep_count = 0;
  for (LO i = 0; i < num_rows; ++i) {
    ST const d = std::abs(diag_data[i]);
    if (d > 0.0) {
      rep_sum += d;
      ++rep_count;
    }
  }
  ST const diag_scale = (rep_count > 0) ? rep_sum / rep_count : 1.0;

  // Rows to regularize:
  //  (1) genuine zero-diagonal orphans (a node whose only incident cells died
  //      in place -- nothing scatters to its row), and
  //  (2) every fully-dead node by CONNECTIVITY (getDeadNodeDOFGids: all incident
  //      cells dead). A calved/disconnected node carries a tiny NONZERO diagonal
  //      (leftover mass on the cloned node + parallel-combine roundoff), so the
  //      exact-zero test in (1) misses it and -- with that vanishing diagonal
  //      dividing the residual -- it free-falls under gravity to ~1e8 m. Keying
  //      (2) off the connectivity-dead set (not a diagonal tolerance) catches
  //      these without snaring gradual-death decaying-but-alive cells. Paired
  //      with zeroResidualAtDeadNodes (the residual companion), this is a true
  //      hold-in-place Dirichlet condition (row = identity, r = 0 -> du = 0).
  std::set<LO> pin_row_set;
  for (LO i = 0; i < num_rows; ++i) {
    if (diag_data[i] == 0.0) pin_row_set.insert(i);
  }
  if (!frozen_dead_dof_gids_.empty()) {
    auto indexer = Albany::createGlobalLocalIndexer(diag->space());
    for (GO const g : frozen_dead_dof_gids_) {
      LO const lid = indexer->getLocalElement(g);
      if (lid >= 0 && lid < num_rows) pin_row_set.insert(lid);
    }
  }

  // Collect the pin rows' column layouts while the operator is fill-complete.
  std::vector<LO>                 orphan_rows;
  std::vector<Teuchos::Array<LO>> orphan_cols;
  for (LO const i : pin_row_set) {
    Teuchos::Array<LO> indices;
    Teuchos::Array<ST> values;
    Albany::getLocalRowValues(jac, i, indices, values);
    if (indices.size() == 0) continue;
    orphan_rows.push_back(i);
    orphan_cols.push_back(indices);
  }

  // Re-open for value modification; the caller fill-completes later.
  resumeFill(jac);
  for (size_t r = 0; r < orphan_rows.size(); ++r) {
    LO const                  row     = orphan_rows[r];
    Teuchos::Array<LO> const& indices = orphan_cols[r];
    Teuchos::Array<ST>        values(indices.size(), 0.0);
    for (int k = 0; k < indices.size(); ++k) {
      if (indices[k] == row) values[k] = diag_scale;
    }
    Albany::setLocalRowValues(jac, row, indices(), values());
  }
}

void
Application::zeroResidualAtDeadNodes(Teuchos::RCP<Thyra_Vector> const& f) const
{
  if (Teuchos::is_null(f)) return;
  auto const& dead_dof_gids = frozen_dead_dof_gids_;
  if (dead_dof_gids.empty()) return;

  // Hold-in-place Dirichlet residual: r = 0 at every fully-dead node DOF, so
  // the Newton update du = -r/diag = 0 leaves the dead node where it was when
  // it became fully dead. This is the companion the orphan-Jacobian pin was
  // missing: without it, a calved node's nonzero residual stayed in |F| and
  // (with the pinned row) could not be reduced -> the solve stalled. Map the
  // global dead-DOF ids through the owned vector's own indexer, bounds-safe and
  // immune to DBC elimination (which keeps original gids in a reduced space).
  auto       indexer = Albany::createGlobalLocalIndexer(f->space());
  auto       f_data  = Albany::getNonconstLocalData(f);
  auto const n_local = static_cast<LO>(f_data.size());
  for (GO const g : dead_dof_gids) {
    LO const lid = indexer->getLocalElement(g);
    if (lid >= 0 && lid < n_local) f_data[lid] = 0.0;
  }
}

bool
Application::applyDeathToActivePart()
{
  // Phase 1 of the activePart-based element-death port.
  //
  // Pre: death_status_vecs_ has been populated by the material model
  //   (J2Erosion writes (*death_status_vec_)[cell] = 1.0 at the last-pt
  //   propagation point in J2Erosion_Def.hpp:770). Called once per
  //   accepted time step from the observer hooks.
  //
  // Within the step that a cell dies, the existing scatter-skip in
  // PHAL_ScatterResidual_Def.hpp still gates its assembly contribution.
  // This routine handles the BETWEEN-steps housekeeping: the dead cell
  // physically leaves activePart and the workset buckets, so subsequent
  // steps never see it again.

  if (death_status_vecs_.empty()) return false;

  auto stk_disc = dynamic_cast<Albany::STKDiscretization*>(disc.get());
  if (stk_disc == nullptr) return false;

  auto& stk_mesh = *stk_disc->getSTKMeshStruct();
  auto* activePart    = stk_mesh.getActivePart();
  auto* deadCellsPart = stk_mesh.getDeadCellsPart();
  if (activePart == nullptr || deadCellsPart == nullptr) return false;

  auto& bulkData = *stk_mesh.bulkData;
  auto& elemGIDws = stk_disc->getElemGIDws();

  // Gather killed cells: those flagged dead in death_status_vecs_ that
  // are still members of activePart. Mirrors the workset/cell scan in
  // fixOrphanNodesForElementDeath.
  stk::mesh::EntityVector killed;
  int const num_worksets = static_cast<int>(death_status_vecs_.size());
  for (auto const& [gid, lid] : elemGIDws) {
    if (lid.ws >= num_worksets) continue;
    if (death_status_vecs_[lid.ws] == Teuchos::null) continue;
    auto const& ds = *death_status_vecs_[lid.ws];
    if (lid.LID >= static_cast<int>(ds.size())) continue;
    if (ds[lid.LID] <= 0.0) continue;

    stk::mesh::Entity cell = bulkData.get_entity(stk::topology::ELEMENT_RANK, gid);
    if (!bulkData.is_valid(cell)) continue;
    if (!bulkData.bucket(cell).member(*activePart)) continue;
    // Killed cells stay in activePart (Step B1 adds to deadCellsPart but
    // does not remove from activePart), so dedup against deadCellsPart.
    if (bulkData.bucket(cell).member(*deadCellsPart)) continue;

    killed.push_back(cell);
  }

  // applyElementDeath -- and the collective calls inside it
  // (modification cycles, the parallel field-sum exchanges) and
  // rebuildWorksets below -- must be entered by *every* rank, even
  // ranks with no locally killed cells: they pass an empty list. Make
  // the early-out a global decision so the ranks never diverge and
  // deadlock.
  if (!stk::is_true_on_any_proc(bulkData.parallel(), !killed.empty())) {
    return false;
  }

  // Calving: a death this step may have severed a block from the
  // kinematic ground. Such a block free falls and, at a coarse coupling
  // step, teleports an absurd distance before the kinematic criteria can
  // fire. If the deck names anchor node sets, find every live cell no
  // longer connected (through live cells) to them and kill it now. This
  // is collective (findDetachedCells exchanges reachability across
  // ranks), so every rank must call it -- guaranteed because the global
  // early-out above already passed on all ranks. findDetachedCells sets
  // cell_death on the detached cells; we append them to the kill list so
  // the same surgery removes them. Anchor nodes themselves are excluded
  // from being killed since they seed the reachable set.
  auto const anchor_node_sets =
      problemParams->get<Teuchos::Array<std::string>>("Anchor Node Sets", Teuchos::Array<std::string>());
  if (!anchor_node_sets.empty()) {
    std::vector<std::string> anchors(anchor_node_sets.begin(), anchor_node_sets.end());
    auto const detached = stk_disc->findDetachedCells(anchors, killed);
    killed.insert(killed.end(), detached.begin(), detached.end());
  }

  // Build the part vectors that the death machinery needs.
  //
  // side_parts: new faces are painted into {activePart, deadCellsPart}
  // (so they are IO-visible and tagged on-the-dead-boundary) plus every
  // "-erodible" side-set, so the eroding surface tracks the receding
  // bluff -- STKDiscretization::computeNodeSets clips an "-erodible"
  // node set to that side-set's nodes, keeping a Dirichlet BC on the
  // live exposed surface (see doc/element-death.md section 8).
  //
  // bc_mesh_parts (boundary_mesh_parts): every declared side-set. STK
  // inherits a dying cell's side-set membership onto the newly-exposed
  // faces, so a Neumann-style BC extends onto the new interface. With
  // the shared mesh both apps' side-sets live on one metaData; each
  // app's BC evaluator iterates only its own parts, so the union is
  // harmless.
  stk::mesh::PartVector side_parts{activePart, deadCellsPart};
  stk::mesh::PartVector bc_mesh_parts;
  bc_mesh_parts.reserve(stk_mesh.ssPartVec.size());
  for (auto& kv : stk_mesh.ssPartVec) {
    if (kv.second == nullptr) continue;
    bc_mesh_parts.push_back(kv.second);
    if (kv.first.find("erodible") != std::string::npos) {
      side_parts.push_back(kv.second);
    }
  }

  // Drive the active/dead interface update. applyElementDeath uses the
  // clone-before-disconnect algorithm (Adagio-style), which sidesteps
  // an STK multi-rank harmonization bug in
  // make_mesh_parallel_consistent_after_element_death. See
  // doc/element-death.md (section "Implementation reference") for the
  // algorithm and ~/LCM/stk_findings_draft.txt for the STK-bug
  // diagnosis.
  applyElementDeath(
      bulkData, *activePart, *deadCellsPart,
      killed, side_parts, bc_mesh_parts);

  // Step B3: rebuild worksets so the discretization picks up the new
  // faces and computeNodeSets re-clips the "-erodible" node sets to the
  // grown side-set. This keeps the mesh usable for the rest of THIS step
  // (e.g. the immediate output write), but refreshes worksets ONLY.
  stk_disc->rebuildWorksets();

  // The clone-before-disconnect surgery above also added nodes (the clones
  // carry new global ids, so the owned/overlap DOF maps grow) and, in
  // parallel, modification_end may have migrated ownership of boundary
  // nodes across ranks. rebuildWorksets() does not refresh the owned/overlap
  // vector spaces, DOF maps, Jacobian graph, or the solution manager's
  // CombineAndScatter manager, so the parallel partition is now stale.
  // Rebuilding those here would invalidate the model evaluator's x_space
  // mid-evalModel (this runs inside the solver's observer), so instead flag
  // the change and let the driving solver do the full rebuild + solution
  // migration at a clean between-step point.
  topology_changed_ = true;

  return true;
}

void
Application::computeGlobalResidualImpl(
    double const                           current_time,
    Teuchos::RCP<Thyra_Vector const> const x,
    Teuchos::RCP<Thyra_Vector const> const x_dot,
    Teuchos::RCP<Thyra_Vector const> const x_dotdot,
    Teuchos::Array<ParamVec> const&        p,
    Teuchos::RCP<Thyra_Vector> const&      f,
    double                                 dt,
    bool const                             suppress_constrained_rates)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany Fill: Residual");
  using EvalT = PHAL::AlbanyTraits::Residual;
  postRegSetup<EvalT>();

  // Load connectivity map and coordinates
  const auto& wsElNodeEqID = disc->getWsElNodeEqID();
  const auto& wsPhysIndex  = disc->getWsPhysIndex();

  int const                        numWorksets  = wsElNodeEqID.size();
  Teuchos::RCP<Thyra_Vector> const overlapped_f = solMgr->get_overlapped_f();

  Teuchos::RCP<const CombineAndScatterManager> cas_manager = solMgr->get_cas_manager();

  // Scatter x and xdot to the overlapped distrbution
  solMgr->scatterX(*x, x_dot.ptr(), x_dotdot.ptr());
  injectConstrainedDOFValues(
      fixTime(current_time),
      Teuchos::nonnull(x_dot) && !suppress_constrained_rates,
      Teuchos::nonnull(x_dotdot) && !suppress_constrained_rates);

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i = 0; i < p.size(); i++) {
    for (unsigned int j = 0; j < p[i].size(); j++) {
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);
    }
  }

  // Store pointers to solution and time derivatives.
  // Needed for Schwarz coupling.
  if (x != Teuchos::null)
    x_ = x;
  else
    x_ = Teuchos::null;
  if (x_dot != Teuchos::null)
    xdot_ = x_dot;
  else
    xdot_ = Teuchos::null;
  if (x_dotdot != Teuchos::null)
    xdotdot_ = x_dotdot;
  else
    xdotdot_ = Teuchos::null;

  // Zero out overlapped residual
  overlapped_f->assign(0.0);
  f->assign(0.0);

  // Set data in Workset struct, and perform fill via field manager
  {
    TEUCHOS_FUNC_TIME_MONITOR("Albany Residual Fill: Evaluate");
    if (Teuchos::nonnull(rc_mgr)) {
      rc_mgr->init_x_if_not(x->space());
    }

    PHAL::Workset workset;

    double const this_time = fixTime(current_time);

    loadBasicWorksetInfo(workset, this_time);

    workset.time_step = dt;

    workset.f = overlapped_f;

    workset.num_worksets = numWorksets;

    for (int ws = 0; ws < numWorksets; ws++) {
      std::string const evalName = PHAL::evalName<EvalT>("FM", wsPhysIndex[ws]);
      loadWorksetBucketInfo<EvalT>(workset, ws, evalName);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<EvalT>(workset);

      if (nfm != Teuchos::null) {
        workset.workset_num = ws;
        deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<EvalT>(workset);
      }
    }
  }

  // Assemble the residual into a non-overlapping vector
  {
    TEUCHOS_FUNC_TIME_MONITOR("Albany Residual Fill: Export");
    cas_manager->combine(overlapped_f, f, CombineMode::ADD);
  }

  // Element death: hold-in-place Dirichlet residual at fully-dead node DOFs
  // (companion to the Jacobian row pin in fixOrphanNodesForElementDeath).
  zeroResidualAtDeadNodes(f);

  // Allocate scaleVec_
  if (scale != 1.0) {
    if (scaleVec_ == Teuchos::null) {
      scaleVec_ = Thyra::createMember(f->space());
      scaleVec_->assign(0.0);
      setScale();
    } else if (Teuchos::nonnull(f)) {
      if (scaleVec_->space()->dim() != f->space()->dim()) {
        scaleVec_ = Thyra::createMember(f->space());
        scaleVec_->assign(0.0);
        setScale();
      }
    }
  }

  if (scaleBCdofs == false && scale != 1.0) {
    Thyra::ele_wise_scale<ST>(*scaleVec_, f.ptr());
  }

  // Push the assembled residual values back into the overlap vector
  cas_manager->scatter(f, overlapped_f, CombineMode::INSERT);

  // Under DBC elimination, constrained-DOF rows are not in the reduced owned
  // map, so the combine above never cross-rank-sums their overlap values. Each
  // rank's overlap slot holds only its local element assembly, so serial and
  // parallel diverge at shared constrained nodes (parallel shows a fraction of
  // the full reaction). Zero those slots to restore the pre-elimination
  // behavior (residual ≈ 0 at constrained DOFs after BC enforcement) and
  // preserve serial/parallel reproducibility in exodus output. Skip for the
  // fully-constrained injection-only path: there, the residual at constrained
  // overlap slots IS the reaction force (= K_internal * u - f_external), and
  // it's the only output the test sees for those nodes — zeroing erases it.
  if (!dbc_state_.descriptors.empty() && dbc_state_.mode != DBCEliminationState::Mode::FullyConstrained) {
    auto overlap_data = Albany::getNonconstLocalData(overlapped_f);
    for (auto const& desc : dbc_state_.descriptors) {
      overlap_data[desc.overlap_lid] = 0.0;
    }
  }

  // Write the residual to the discretization, which will later (optionally)
  // be written to the output file
  disc->setResidualField(*overlapped_f);

  // Fully-constrained injection-only path (every owned DOF is BC-prescribed,
  // e.g. single-element CrystalPlasticity / SurfaceElement / cohesive tests):
  // NOX runs against the full disc-owned space, but the actual residual at
  // constrained rows is K*u_bc - f_ext and can be arbitrarily large. Force
  // the owned residual to (u - u_bc) = 0 at every constrained owned LID so
  // NOX sees |F| = 0 and exits in iter 1 with u = u_bc, while the residual
  // fill above still updates the per-quadrature-point internal state from
  // the prescribed displacement.
  if (dbc_state_.mode == DBCEliminationState::Mode::FullyConstrained) {
    auto owned_f = Albany::getNonconstLocalData(f);
    auto owned_x = Albany::getLocalData(x);
    double const this_time = fixTime(current_time);
    for (auto const& desc : dbc_state_.descriptors) {
      if (desc.full_owned_lid >= 0) {
        // SDirichlet residual rewrite: residual at constrained rows is the
        // BC mismatch (x - u_bc), so NOX's J·Δ = -r solve drives x → u_bc
        // in one Newton step paired with the identity-row Jacobian below.
        owned_f[desc.full_owned_lid] = owned_x[desc.full_owned_lid] - desc.eval(this_time);
      }
    }
  }

  // scale residual by scaleVec_ if scaleBCdofs is on
  if (scaleBCdofs == true) {
    Thyra::ele_wise_scale<ST>(*scaleVec_, f.ptr());
  }
}

void
Application::computeGlobalResidual(
    double const                            current_time,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& x_dot,
    Teuchos::RCP<Thyra_Vector const> const& x_dotdot,
    const Teuchos::Array<ParamVec>&         p,
    Teuchos::RCP<Thyra_Vector> const&       f,
    double const                            dt,
    bool const                              suppress_constrained_rates)
{
  this->computeGlobalResidualImpl(current_time, x, x_dot, x_dotdot, p, f, dt, suppress_constrained_rates);

  // Debut output write residual or solution to MatrixMarket
  // every time it arises or at requested count#
  auto const write_sol_mm = writeToMatrixMarketSol != 0 && (writeToMatrixMarketSol == -1 || countSol == writeToMatrixMarketSol);

  if (write_sol_mm == true) {
    *out << "Writing global solution #" << countSol << " to MatrixMarket at time t = " << current_time << ".\n";
    writeMatrixMarket(x, "disp", countSol);
    writeMatrixMarket(x_dot, "velo", countSol);
    if (x_dotdot != Teuchos::null) {
      writeMatrixMarket(x_dotdot, "acce", countSol);
    }
  }
  auto const write_sol_co = writeToCoutSol != 0 && (writeToCoutSol == -1 || countSol == writeToCoutSol);
  if (write_sol_co == true) {
    *out << "Global solution #" << countSol << " corresponding to time t = " << current_time << ":\n";
    describe(x.getConst(), *out, Teuchos::VERB_EXTREME);
  }
  if (writeToMatrixMarketSol != 0 || writeToCoutSol != 0) {
    countSol++;
  }

  auto const write_res_mm = writeToMatrixMarketRes != 0 && (writeToMatrixMarketRes == -1 || countRes == writeToMatrixMarketRes);

  if (write_res_mm == true) {
    *out << "Writing global residual #" << countRes << " to MatrixMarket at time t = " << current_time << ".\n";
    writeMatrixMarket(f, "resi", countRes);
  }
  auto const write_res_co = writeToCoutRes != 0 && (writeToCoutRes == -1 || countRes == writeToCoutRes);
  if (write_res_co == true) {
    *out << "Global residual #" << countRes << " corresponding to time t = " << current_time << ":\n";
    describe(f.getConst(), *out, Teuchos::VERB_EXTREME);
  }
  if (writeToMatrixMarketRes != 0 || writeToCoutRes != 0) {
    countRes++;
  }
}

void
Application::computeGlobalJacobianImpl(
    double const                            alpha,
    double const                            beta,
    double const                            omega,
    double const                            current_time,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& xdot,
    Teuchos::RCP<Thyra_Vector const> const& xdotdot,
    const Teuchos::Array<ParamVec>&         p,
    Teuchos::RCP<Thyra_Vector> const&       f,
    const Teuchos::RCP<Thyra_LinearOp>&     jac,
    double const                            dt)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany Fill: Jacobian");
  using EvalT = PHAL::AlbanyTraits::Jacobian;
  postRegSetup<EvalT>();

  // Load connectivity map and coordinates
  const auto& wsElNodeEqID = disc->getWsElNodeEqID();
  const auto& wsPhysIndex  = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Thyra_Vector> overlapped_f;
  if (Teuchos::nonnull(f)) {
    overlapped_f = solMgr->get_overlapped_f();
  }

  Teuchos::RCP<Thyra_LinearOp> overlapped_jac = solMgr->get_overlapped_jac();
  auto                         cas_manager    = solMgr->get_cas_manager();

  // Scatter x and xdot to the overlapped distribution
  solMgr->scatterX(*x, xdot.ptr(), xdotdot.ptr());
  injectConstrainedDOFValues(fixTime(current_time), Teuchos::nonnull(xdot), Teuchos::nonnull(xdotdot));

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i = 0; i < p.size(); i++) {
    for (unsigned int j = 0; j < p[i].size(); j++) {
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);
    }
  }

  // Zero out overlapped residual
  if (Teuchos::nonnull(f)) {
    overlapped_f->assign(0.0);
    f->assign(0.0);
  }

  // Zero out Jacobian
  resumeFill(jac);
  assign(jac, 0.0);

  if (!isFillActive(overlapped_jac)) {
    resumeFill(overlapped_jac);
  }
  assign(overlapped_jac, 0.0);
  if (isFillActive(overlapped_jac)) {
    fillComplete(overlapped_jac);
  }
  if (!isFillActive(overlapped_jac)) {
    resumeFill(overlapped_jac);
  }

  // Set data in Workset struct, and perform fill via field manager
  {
    TEUCHOS_FUNC_TIME_MONITOR("Albany Jacobian Fill: Evaluate");
    PHAL::Workset workset;

    double const this_time = fixTime(current_time);

    loadBasicWorksetInfo(workset, this_time);

    workset.time_step = dt;

    workset.f   = overlapped_f;
    workset.Jac = overlapped_jac;
    loadWorksetJacobianInfo(workset, alpha, beta, omega);

    // fill Jacobian derivative dimensions:
    for (int ps = 0; ps < fm.size(); ps++) {
      (workset.Jacobian_deriv_dims).push_back(PHAL::getDerivativeDimensions<EvalT>(this, ps));
    }

    if (!workset.f.is_null()) {
      workset.f_kokkos = getNonconstDeviceData(workset.f);
    }
    if (!workset.Jac.is_null()) {
      workset.Jac_kokkos = getNonconstDeviceData(workset.Jac);
    }
    workset.num_worksets = numWorksets;
    for (int ws = 0; ws < numWorksets; ws++) {
      std::string const evalName = PHAL::evalName<EvalT>("FM", wsPhysIndex[ws]);
      loadWorksetBucketInfo<EvalT>(workset, ws, evalName);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<EvalT>(workset);
      workset.workset_num = ws;
      if (Teuchos::nonnull(nfm)) deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<EvalT>(workset);
    }
  }

  // Allocate and populate scaleVec_
  if (scale != 1.0) {
    if (scaleVec_ == Teuchos::null || scaleVec_->space()->dim() != jac->domain()->dim()) {
      scaleVec_ = Thyra::createMember(jac->range());
      scaleVec_->assign(0.0);
      setScale();
    }
  }

  {
    TEUCHOS_FUNC_TIME_MONITOR("Albany Jacobian Fill: Export");
    // Assemble global residual
    if (Teuchos::nonnull(f)) {
      cas_manager->combine(overlapped_f, f, CombineMode::ADD);
    }
    // Assemble global Jacobian
    cas_manager->combine(overlapped_jac, jac, CombineMode::ADD);
  }

  // Element death: hold-in-place Dirichlet on fully-dead node DOFs. The
  // Jacobian row pin and the residual zeroing must use the SAME dead set so
  // the pinned rows (du = -r/diag) drive du = 0; apply both here.
  zeroResidualAtDeadNodes(f);

  // Element death: regularize zero-diagonal (orphan) rows + pin fully-dead rows
  fixOrphanNodesForElementDeath(jac);

  // scale Jacobian
  if (scaleBCdofs == false && scale != 1.0) {
    fillComplete(jac);
    // set the scaling
    setScale(jac);

    // scale Jacobian
    // We MUST be able to cast jac to ScaledLinearOpBase in order to left
    // scale it.
    auto jac_scaled_lop = Teuchos::rcp_dynamic_cast<Thyra::ScaledLinearOpBase<ST>>(jac, true);
    jac_scaled_lop->scaleLeft(*scaleVec_);
    resumeFill(jac);
    countScale++;
  }

  fillComplete(jac);

  // Fully-constrained injection-only path: after the fill, zero the rows
  // of the Jacobian at constrained owned LIDs and put 1 on the diagonal so
  // the linear solve becomes J·Δ = -r → Δ = u_bc - x. NOX exits in one
  // Newton step with x = u_bc — the SDirichlet residual rewrite above
  // turns this into the BC mismatch and the identity row turns the solve
  // into a pure substitution.
  if (dbc_state_.mode == DBCEliminationState::Mode::FullyConstrained) {
    resumeFill(jac);
    for (auto const& desc : dbc_state_.descriptors) {
      LO const row = desc.full_owned_lid;
      if (row < 0) continue;
      Teuchos::Array<LO> indices;
      Teuchos::Array<ST> values;
      Albany::getLocalRowValues(jac, row, indices, values);
      Teuchos::Array<ST> new_values(indices.size(), 0.0);
      for (int k = 0; k < indices.size(); ++k) {
        if (indices[k] == row) new_values[k] = 1.0;
      }
      Albany::setLocalRowValues(jac, row, indices(), new_values());
    }
    fillComplete(jac);
  }

  // Apply scaling to residual and Jacobian
  if (scaleBCdofs == true) {
    if (Teuchos::nonnull(f)) {
      Thyra::ele_wise_scale<ST>(*scaleVec_, f.ptr());
    }
    // We MUST be able to cast jac to ScaledLinearOpBase in order to left scale
    // it.
    auto jac_scaled_lop = Teuchos::rcp_dynamic_cast<Thyra::ScaledLinearOpBase<ST>>(jac, true);
    jac_scaled_lop->scaleLeft(*scaleVec_);
  }

  if (isFillActive(overlapped_jac)) {
    // Makes getLocalMatrix() valid.
    fillComplete(overlapped_jac);
  }
  if (derivatives_check_ > 0) {
    checkDerivatives(*this, current_time, x, xdot, xdotdot, p, f, jac, derivatives_check_);
  }
}  // namespace Albany

void
Application::computeGlobalJacobian(
    double const                            alpha,
    double const                            beta,
    double const                            omega,
    double const                            current_time,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& xdot,
    Teuchos::RCP<Thyra_Vector const> const& xdotdot,
    const Teuchos::Array<ParamVec>&         p,
    Teuchos::RCP<Thyra_Vector> const&       f,
    const Teuchos::RCP<Thyra_LinearOp>&     jac,
    double const                            dt)
{
  this->computeGlobalJacobianImpl(alpha, beta, omega, current_time, x, xdot, xdotdot, p, f, jac, dt);
  // Debut output
  if (writeToMatrixMarketJac != 0) {
    // If requesting writing to MatrixMarket of Jacobian...
    if (writeToMatrixMarketJac == -1) {
      // write jacobian to MatrixMarket every time it arises

      *out << "Writing global Jacobian #" << countJac << " to MatrixMarket at time t = " << current_time << ".\n";
      writeMatrixMarket(jac.getConst(), "jac", countJac);
    } else if (countJac == writeToMatrixMarketJac) {
      // write jacobian only at requested count#
      *out << "Writing global Jacobian #" << countJac << " to MatrixMarket at time t = " << current_time << ".\n";
      writeMatrixMarket(jac.getConst(), "jac", countJac);
    }
  }
  if (writeToCoutJac != 0) {
    // If requesting writing Jacobian to standard output (cout)...
    if (writeToCoutJac == -1) {  // cout jacobian every time it arises
      *out << "Global Jacobian #" << countJac << " corresponding to time t = " << current_time << ":\n";
      describe(jac.getConst(), *out, Teuchos::VERB_EXTREME);
    } else if (countJac == writeToCoutJac) {
      // cout jacobian only at requested count#
      *out << "Global Jacobian #" << countJac << " corresponding to time t = " << current_time << ":\n";
      describe(jac.getConst(), *out, Teuchos::VERB_EXTREME);
    }
  }
  if (writeToMatrixMarketJac != 0 || writeToCoutJac != 0) {
    countJac++;  // increment Jacobian counter
  }
}

void
Application::evaluateResponse(
    int                                     response_index,
    double const                            current_time,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& xdot,
    Teuchos::RCP<Thyra_Vector const> const& xdotdot,
    const Teuchos::Array<ParamVec>&         p,
    Teuchos::RCP<Thyra_Vector> const&       g)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany Fill: Response");
  double const this_time = fixTime(current_time);
  // For DBC elimination, the expansion needs the correct transient time.
  // During the post-integration response pass, x_dot is null so is_dynamic=false
  // and curr_time=getCurrentTime()=0; use last_transient_time_ as fallback.
  double const expand_time = (dbc_state_.descriptors.empty() || this_time != 0.0) ? this_time : last_transient_time_;
  auto const   x_full      = expandToFullSolution(x, expand_time);
  responses[response_index]->evaluateResponse(this_time, x_full, xdot, xdotdot, p, g);
}

void
Application::evaluateStateFieldManager(double const current_time, const Thyra_MultiVector& x)
{
  int num_vecs = x.domain()->dim();

  if (num_vecs == 1) {
    this->evaluateStateFieldManager(current_time, *x.col(0), Teuchos::null, Teuchos::null);
  } else if (num_vecs == 2) {
    this->evaluateStateFieldManager(current_time, *x.col(0), x.col(1).ptr(), Teuchos::null);
  } else {
    this->evaluateStateFieldManager(current_time, *x.col(0), x.col(1).ptr(), x.col(2).ptr());
  }
}

void
Application::evaluateStateFieldManager(
    double const                     current_time,
    Thyra_Vector const&              x,
    Teuchos::Ptr<Thyra_Vector const> xdot,
    Teuchos::Ptr<Thyra_Vector const> xdotdot)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany Fill: State Residual");
  {
    std::string evalName = PHAL::evalName<PHAL::AlbanyTraits::Residual>("SFM", 0);
    if (!phxSetup->contain_eval(evalName)) {
      for (int ps = 0; ps < sfm.size(); ++ps) {
        evalName = PHAL::evalName<PHAL::AlbanyTraits::Residual>("SFM", ps);
        phxSetup->insert_eval(evalName);

        std::vector<PHX::index_size_type> derivative_dimensions;
        derivative_dimensions.push_back(PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(this, ps));
        sfm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(derivative_dimensions);
        sfm[ps]->postRegistrationSetup(*phxSetup);

        // Update phalanx saved/unsaved fields based on field dependencies
        phxSetup->check_fields(sfm[ps]->getFieldTagsForSizing<PHAL::AlbanyTraits::Residual>());
        phxSetup->update_fields();

        writePhalanxGraph<PHAL::AlbanyTraits::Residual>(sfm[ps], evalName, stateGraphVisDetail);
      }
    }
  }

  Teuchos::RCP<Thyra_Vector> overlapped_f = solMgr->get_overlapped_f();

  // Load connectivity map and coordinates
  const auto& wsElNodeEqID = disc->getWsElNodeEqID();
  const auto& wsPhysIndex  = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  // Scatter to the overlapped distrbution
  solMgr->scatterX(x, xdot, xdotdot);
  injectConstrainedDOFValues(fixTime(current_time), !xdot.is_null(), !xdotdot.is_null());

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set data in Workset struct
  PHAL::Workset workset;
  loadBasicWorksetInfo(workset, current_time);
  workset.f = overlapped_f;

  // Perform fill via field manager
  if (Teuchos::nonnull(rc_mgr)) rc_mgr->beginEvaluatingSfm();
  for (int ws = 0; ws < numWorksets; ws++) {
    std::string const evalName = PHAL::evalName<PHAL::AlbanyTraits::Residual>("SFM", wsPhysIndex[ws]);
    loadWorksetBucketInfo<PHAL::AlbanyTraits::Residual>(workset, ws, evalName);
    sfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
  }
  if (Teuchos::nonnull(rc_mgr)) rc_mgr->endEvaluatingSfm();
}

void
Application::registerShapeParameters()
{
  int numShParams = shapeParams.size();
  if (shapeParamNames.size() == 0) {
    shapeParamNames.resize(numShParams);
    for (int i = 0; i < numShParams; i++) shapeParamNames[i] = strint("ShapeParam", i);
  }
  DummyParameterAccessor<PHAL::AlbanyTraits::Jacobian, SPL_Traits>* dJ = new DummyParameterAccessor<PHAL::AlbanyTraits::Jacobian, SPL_Traits>();

  // Register Parameter for Residual fill using "this->getValue" but
  // create dummy ones for other type that will not be used.
  for (int i = 0; i < numShParams; i++) {
    *out << "Registering Shape Param " << shapeParamNames[i] << std::endl;
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Residual, SPL_Traits>(shapeParamNames[i], this, paramLib);
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Jacobian, SPL_Traits>(shapeParamNames[i], dJ, paramLib);
  }
}

PHAL::AlbanyTraits::Residual::ScalarT&
Application::getValue(std::string const& name)
{
  int index = -1;
  for (unsigned int i = 0; i < shapeParamNames.size(); i++) {
    if (name == shapeParamNames[i]) index = i;
  }
  ALBANY_PANIC(index == -1, "Error in GatherCoordinateVector::getValue, \n" << "   Unrecognized param name: " << name << std::endl);

  shapeParamsHaveBeenReset = true;

  return shapeParams[index];
}

void
Application::determinePiroSolver(const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams)
{
  const Teuchos::RCP<Teuchos::ParameterList>& localProblemParams = Teuchos::sublist(topLevelParams, "Problem", true);

  const Teuchos::RCP<Teuchos::ParameterList>& piroParams = Teuchos::sublist(topLevelParams, "Piro");

  // If not explicitly specified, determine which Piro solver to use from the
  // problem parameters
  if (!piroParams->getPtr<std::string>("Solver Type")) {
    std::string const secondOrder = localProblemParams->get("Second Order", "No");

    ALBANY_PANIC(
        secondOrder != "No" && secondOrder != "Velocity Verlet" && secondOrder != "Newmark" && secondOrder != "Trapezoid Rule",
        "Invalid value for Second Order: (No, Velocity Verlet, Newmark, "
        "Trapezoid Rule): "
            << secondOrder << "\n");

    // Populate the Piro parameter list accordingly to inform the Piro solver
    // factory
    std::string piroSolverToken;
    if (solMethod == Steady) {
      piroSolverToken = "NOX";
    } else if (solMethod == Continuation) {
      piroSolverToken = "LOCA";
    } else if (solMethod == Transient) {
      ALBANY_PANIC(
          secondOrder == "No",
          "Transient solution method only works for second order in time problem!\n"
              << "For first order in time problems, please use 'Transient Tempus'.\n");
      if (secondOrder != "No") piroSolverToken = secondOrder;
    } else if (solMethod == TransientTempus) {
      piroSolverToken = (secondOrder == "No") ? "Tempus" : secondOrder;
    } else {
      // Piro cannot handle the corresponding problem
      piroSolverToken = "Unsupported";
    }
    piroParams->set("Solver Type", piroSolverToken);

    const bool  compute_sens = problemParams->get<bool>("Compute Sensitivities", false);
    std::string sens_method  = "None";
    if (compute_sens == true) {
      sens_method = piroParams->get<std::string>("Sensitivity Method", "Forward");
    }
    if ((sens_method == "Adjoint") && (piroSolverToken == "Tempus")) {
      ALBANY_ABORT("LCM does not have support for adjoint transient sensitivities!\n");
    }
  }
}

void
Application::loadBasicWorksetInfo(PHAL::Workset& workset, double current_time)
{
  auto overlapped_MV = solMgr->getOverlappedSolution();
  auto numVectors    = overlapped_MV->domain()->dim();

  workset.x       = overlapped_MV->col(0);
  workset.xdot    = numVectors > 1 ? overlapped_MV->col(1) : Teuchos::null;
  workset.xdotdot = numVectors > 2 ? overlapped_MV->col(2) : Teuchos::null;

  workset.numEqs            = neq;
  workset.current_time      = current_time + time_shift_;
  workset.distParamLib      = distParamLib;
  workset.disc              = disc;
  // suppress_dynamics_ (ACE preload) forces a quasi-static fill: drop the
  // rate/inertia terms even though the time integrator supplies xdot/xdotdot.
  workset.transientTerms    = Teuchos::nonnull(workset.xdot) && !suppress_dynamics_;
  workset.accelerationTerms = Teuchos::nonnull(workset.xdotdot) && !suppress_dynamics_;
}

void
Application::loadWorksetJacobianInfo(PHAL::Workset& workset, double const alpha, double const beta, double const omega)
{
  workset.m_coeff         = alpha;
  workset.n_coeff         = omega;
  workset.j_coeff         = beta;
  workset.ignore_residual = ignore_residual_in_jacobian;
  workset.is_adjoint      = is_adjoint;
}

void
Application::setScale(Teuchos::RCP<const Thyra_LinearOp> jac)
{
  if (scaleBCdofs == true) {
    if (scaleVec_->norm_2() == 0.0) {
      scaleVec_->assign(1.0);
    }
    return;
  }

  if (scale_type == CONSTANT) {  // constant scaling
    scaleVec_->assign(1.0 / scale);
  } else if (scale_type == DIAG) {  // diagonal scaling
    if (jac == Teuchos::null) {
      scaleVec_->assign(1.0);
    } else {
      getDiagonalCopy(jac, scaleVec_);
      Thyra::reciprocal<ST>(*scaleVec_, scaleVec_.ptr());
    }
  } else if (scale_type == ABSROWSUM) {  // absolute value of row sum scaling
    if (jac == Teuchos::null) {
      scaleVec_->assign(1.0);
    } else {
      scaleVec_->assign(0.0);
      // We MUST be able to cast the linear op to RowStatLinearOpBase, in order
      // to get row informations
      auto jac_row_stat = Teuchos::rcp_dynamic_cast<const Thyra::RowStatLinearOpBase<ST>>(jac, true);

      // Compute the inverse of the absolute row sum
      jac_row_stat->getRowStat(Thyra::RowStatLinearOpBaseUtils::ROW_STAT_INV_ROW_SUM, scaleVec_.ptr());
    }
  }
}

void
Application::loadWorksetSidesetInfo(PHAL::Workset& workset, int const ws)
{
  workset.sideSets = Teuchos::rcpFromRef(disc->getSideSets(ws));
}

void
Application::setupBasicWorksetInfo(
    PHAL::Workset&                          workset,
    double                                  current_time,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& xdot,
    Teuchos::RCP<Thyra_Vector const> const& xdotdot,
    const Teuchos::Array<ParamVec>&         p)
{
  Teuchos::RCP<const Thyra_MultiVector> overlapped_MV = solMgr->getOverlappedSolution();
  auto                                  numVectors    = overlapped_MV->domain()->dim();

  Teuchos::RCP<Thyra_Vector const> overlapped_x       = overlapped_MV->col(0);
  Teuchos::RCP<Thyra_Vector const> overlapped_xdot    = numVectors > 1 ? overlapped_MV->col(1) : Teuchos::null;
  Teuchos::RCP<Thyra_Vector const> overlapped_xdotdot = numVectors > 2 ? overlapped_MV->col(2) : Teuchos::null;

  // Scatter xT and xdotT to the overlapped distrbution
  solMgr->scatterX(*x, xdot.ptr(), xdotdot.ptr());
  injectConstrainedDOFValues(fixTime(current_time), Teuchos::nonnull(xdot), Teuchos::nonnull(xdotdot));

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i = 0; i < p.size(); i++) {
    for (unsigned int j = 0; j < p[i].size(); j++) {
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);
    }
  }

  double const this_time    = fixTime(current_time);
  workset.current_time      = this_time + time_shift_;
  workset.x                 = overlapped_x;
  workset.xdot              = overlapped_xdot;
  workset.xdotdot           = overlapped_xdotdot;
  workset.distParamLib      = distParamLib;
  workset.disc              = disc;
  // suppress_dynamics_ (ACE preload) forces a quasi-static fill: drop the
  // rate/inertia terms even though the time integrator supplies xdot/xdotdot.
  workset.transientTerms    = Teuchos::nonnull(workset.xdot) && !suppress_dynamics_;
  workset.accelerationTerms = Teuchos::nonnull(workset.xdotdot) && !suppress_dynamics_;
  workset.comm              = comm;
  workset.x_cas_manager     = solMgr->get_cas_manager();
}

void
Application::setCoupledAppBlockNodeset(std::string const& app_name, std::string const& block_name, std::string const& nodeset_name)
{
  // Check for valid application name
  auto it = app_name_index_map_->find(app_name);

  ALBANY_PANIC(it == app_name_index_map_->end(), "Trying to couple to an unknown Application: " << app_name << '\n');

  int const app_index             = it->second;
  auto      block_nodeset_names   = std::make_pair(block_name, nodeset_name);
  auto      app_index_block_names = std::make_pair(app_index, block_nodeset_names);
  coupled_app_index_block_nodeset_names_map_.insert(app_index_block_names);
}

}  // namespace Albany
