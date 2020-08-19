// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_Application.hpp"

#include <string>

#include "AAdapt_RC_Manager.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_DummyParameterAccessor.hpp"
#include "Albany_Macros.hpp"
#include "Albany_ProblemFactory.hpp"
#include "Albany_ResponseFactory.hpp"
#include "Albany_ScalarResponseFunction.hpp"
#include "Albany_ThyraUtils.hpp"
#include "PHAL_Utilities.hpp"
#include "SolutionSniffer.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_VectorBase.hpp"
#include "Thyra_VectorStdOps.hpp"

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
    Teuchos::ParameterList& pList =
        using_old_parameter_list ? parameterParams : parameterParams.sublist(Albany::strint("Parameter Vector", i));
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
      no_dir_bcs_(false),
      requires_sdbcs_(false),
      requires_orig_dbcs_(false),
      comm(comm_),
      out(Teuchos::VerboseObjectBase::getDefaultOStream()),
      params_(params),
      physicsBasedPreconditioner(false),
      shapeParamsHaveBeenReset(false),
      phxGraphVisDetail(0),
      stateGraphVisDetail(0),
      morphFromInit(true),
      perturbBetaForDirichlets(0.0)
{
  initialSetUp(params);
  createMeshSpecs();
  buildProblem();
  createDiscretization();
  finalSetUp(params, initial_guess);
}

Application::Application(const RCP<Teuchos_Comm const>& comm_)
    : no_dir_bcs_(false),
      requires_sdbcs_(false),
      requires_orig_dbcs_(false),
      comm(comm_),
      out(Teuchos::VerboseObjectBase::getDefaultOStream()),
      physicsBasedPreconditioner(false),
      shapeParamsHaveBeenReset(false),
      phxGraphVisDetail(0),
      stateGraphVisDetail(0),
      morphFromInit(true),
      perturbBetaForDirichlets(0.0)
{
  // Nothing to be done here
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

#if defined(ALBANY_DTK)
  if (is_schwarz_ == true) {
    // Write DTK Field to Exodus if Schwarz is used
    discParams.set<bool>("Output DTK Field to Exodus", true);
  }
#endif

  ALBANY_PANIC(
      num_time_deriv > 2,
      "Input error: number of time derivatives must be <= 2 "
          << "(solution, solution_dot, solution_dotdot)");

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

    // Add NOX pre-post-operator for debugging.
    bool const have_piro = params->isSublist("Piro");
    ALBANY_ASSERT(have_piro == true, "Error! Piro sublist not found.\n");
    Teuchos::ParameterList& piro_params = params->sublist("Piro");
    bool const              have_dbcs   = problemParams->isSublist("Dirichlet BCs");
    if (have_dbcs == false) no_dir_bcs_ = true;
    bool const have_tempus = piro_params.isSublist("Tempus");
    ALBANY_ASSERT(have_tempus == true, "Error! Tempus sublist not found.\n");
    Teuchos::ParameterList& tempus_params       = piro_params.sublist("Tempus");
    bool const              have_tempus_stepper = tempus_params.isSublist("Tempus Stepper");

    ALBANY_ASSERT(have_tempus_stepper == true, "Error! Tempus stepper sublist not found.\n");

    Teuchos::ParameterList& tempus_stepper_params = tempus_params.sublist("Tempus Stepper");

    std::string stepper_type = tempus_stepper_params.get<std::string>("Stepper Type");

    Teuchos::ParameterList nox_params;

    //The following code checks if we are using an Explicit stepper in Tempus, so as 
    //to do appropriate error checking (e.g., disallow DBCs, which do not work with explicit steppers). 
    //IKT, 8/13/2020: warning - the logic here may not encompass all explicit steppers
    //in Tempus! 
    std::string const expl_str = "Explicit"; 
    std::string const forward_eul = "Forward Euler"; 
    bool is_explicit_scheme = false; 
    std::size_t found = stepper_type.find(expl_str); 
    std::size_t found2 = stepper_type.find(forward_eul); 
    if ((found != std::string::npos) || (found2 != std::string::npos)) {
      is_explicit_scheme = true; 
    }
    if ((stepper_type == "General ERK") || (stepper_type == "RK1")) {
      is_explicit_scheme = true;
    } 
    
    if ((stepper_type == "Newmark Implicit d-Form") || (stepper_type == "Newmark Implicit a-Form")) {
      bool const have_solver_name = tempus_stepper_params.isType<std::string>("Solver Name");

      ALBANY_ASSERT(have_solver_name == true, "Error! Implicit solver sublist not found.\n");

      std::string const solver_name = tempus_stepper_params.get<std::string>("Solver Name");

      Teuchos::ParameterList& solver_name_params = tempus_stepper_params.sublist(solver_name);

      bool const have_nox = solver_name_params.isSublist("NOX");
      ALBANY_ASSERT(have_nox == true, "Error! Nox sublist not found.\n");
      nox_params                   = solver_name_params.sublist("NOX");
      std::string nonlinear_solver = nox_params.get<std::string>("Nonlinear Solver");

      // Set flag marking that we are running with Tempus + d-Form Newmark +
      // SDBCs.
      if (stepper_type == "Newmark Implicit d-Form") {
        if (nonlinear_solver != "Line Search Based") {
          ALBANY_ABORT(
              "Newmark Implicit d-Form Stepper Type will not work correctly "
              "with 'Nonlinear Solver' = "
              << nonlinear_solver
              << "!  The valid Nonlinear Solver for this scheme is 'Line "
                 "Search Based'.");
        }
      }
      if (stepper_type == "Newmark Implicit a-Form") { requires_orig_dbcs_ = true; }
    } 
    //Explicit steppers require SDBCs
    if (is_explicit_scheme == true) {
      requires_sdbcs_ = true;
    }

  } else {
    ALBANY_ABORT(
        "Solution Method must be Steady, Transient, Transient Tempus, "
        "Transient Tempus No Piro, "
        << "Continuation, or Eigensolve, not : " << solutionMethod);
  }

  std::string stepperType;
  if (solMethod == Transient) {
    // Get Piro PL
    Teuchos::RCP<Teuchos::ParameterList> piroParams = Teuchos::sublist(params, "Piro", true);
    // Check if there is Rythmos Solver sublist, and get the stepper type
    if (piroParams->isSublist("Rythmos Solver")) {
      Teuchos::RCP<Teuchos::ParameterList> rythmosSolverParams = Teuchos::sublist(piroParams, "Rythmos Solver", true);
      if (rythmosSolverParams->isSublist("Rythmos")) {
        Teuchos::RCP<Teuchos::ParameterList> rythmosParams = Teuchos::sublist(rythmosSolverParams, "Rythmos", true);
        if (rythmosParams->isSublist("Stepper Settings")) {
          Teuchos::RCP<Teuchos::ParameterList> stepperSettingsParams =
              Teuchos::sublist(rythmosParams, "Stepper Settings", true);
          if (stepperSettingsParams->isSublist("Stepper Selection")) {
            Teuchos::RCP<Teuchos::ParameterList> stepperSelectionParams =
                Teuchos::sublist(stepperSettingsParams, "Stepper Selection", true);
            stepperType = stepperSelectionParams->get("Stepper Type", "Backward Euler");
          }
        }
      }
    }
    // Check if there is Rythmos sublist, and get the stepper type
    else if (piroParams->isSublist("Rythmos")) {
      Teuchos::RCP<Teuchos::ParameterList> rythmosParams = Teuchos::sublist(piroParams, "Rythmos", true);
      stepperType                                        = rythmosParams->get("Stepper Type", "Backward Euler");
    }
  } else if (solMethod == TransientTempus) {
    // Get Piro PL
    Teuchos::RCP<Teuchos::ParameterList> piroParams = Teuchos::sublist(params, "Piro", true);
    // Check if there is Rythmos Solver sublist, and get the stepper type
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
    Teuchos::ArrayRCP<Teuchos::RCP<Application>> aa =
        params->get<Teuchos::ArrayRCP<Teuchos::RCP<Application>>>("Application Array");

    int const ai = params->get<int>("Application Index");

    Teuchos::RCP<std::map<std::string, int>> anim =
        params->get<Teuchos::RCP<std::map<std::string, int>>>("Application Name Index Map");

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

  if ((requires_sdbcs_ == true) && (problem->useSDBCs() == false) && (no_dir_bcs_ == false)) {
    ALBANY_ABORT(
        "Error in Albany::Application: you are using a "
        "solver that requires SDBCs yet you are not "
        "using SDBCs!  Explicit time-steppers require SDBCs.\n");
  }

  if ((requires_orig_dbcs_ == true) && (problem->useSDBCs() == true)) {
    ALBANY_ABORT(
        "Error in Albany::Application: you are using a "
        "solver with SDBCs that does not work correctly "
        "with them!\n");
  }

  if ((no_dir_bcs_ == true) && (scaleBCdofs == true)) {
    ALBANY_ABORT(
        "Error in Albany::Application: you are attempting "
        "to set 'Scale DOF BCs = true' for a problem with no  "
        "Dirichlet BCs!  Scaling will do nothing.  Re-run "
        "with 'Scale DOF BCs = false'\n");
  }

  neq               = problem->numEquations();
  spatial_dimension = problem->spatialDimension();

  // Construct responses
  // This really needs to happen after the discretization is created for
  // distributed responses, but currently it can't be moved because there
  // are responses that setup states, which has to happen before the
  // discretization is created.  We will delay setup of the distributed
  // responses to deal with this temporarily.
  Teuchos::ParameterList& responseList = problemParams->sublist("Response Functions");
  ResponseFactory responseFactory(Teuchos::rcp(this, false), problem, meshSpecs, Teuchos::rcp(&stateMgr, false));
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
    std::string              elementBlockName       = meshSpecs[ps]->ebName;
    std::vector<std::string> responseIDs_to_require = stateMgr.getResidResponseIDsToRequire(elementBlockName);
    sfm[ps]                                         = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>> tags =
        problem->buildEvaluators(*sfm[ps], *meshSpecs[ps], stateMgr, BUILD_STATE_FM, Teuchos::null);
    std::vector<std::string>::const_iterator it;
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

  if ((scale != 1.0) && (problem->useSDBCs() == true)) {
    ALBANY_ABORT("'Scaling' sublist not recognized when using SDBCs.");
  }
}

void
Application::finalSetUp(
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    Teuchos::RCP<Thyra_Vector const> const&     initial_guess)
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
    Teuchos::RCP<DistributedParameter> parameter(new DistributedParameter(
        param_name, disc->getVectorSpace(param_name), disc->getOverlapVectorSpace(param_name)));
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
        if (topoParams.get<std::string>("Entity Type") == "Distributed Parameter" &&
            topoParams.get<std::string>("Topology Name") == param_name) {
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
  dfm = problem->getDirichletFieldManager();

  offsets_    = problem->getOffsets();
  nodeSetIDs_ = problem->getNodeSetIDs();

  nfm = problem->getNeumannFieldManager();

  if (comm->getRank() == 0) {
    phxGraphVisDetail   = problemParams->get("Phalanx Graph Visualization Detail", 0);
    stateGraphVisDetail = phxGraphVisDetail;
  }

  *out << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << " Sacado ParameterLibrary has been initialized:\n " << *paramLib
       << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << std::endl;

  // Allow Problem to add custom NOX status test
  problem->applyProblemSpecificSolverSettings(params);

  ignore_residual_in_jacobian = problemParams->get("Ignore Residual In Jacobian", false);

  perturbBetaForDirichlets = problemParams->get("Perturb Dirichlet", 0.0);

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
// amb-nfm I think right now there is some confusion about nfm. Long ago, nfm
// was
// like dfm, just a single field manager. Then it became an array like fm. At
// that time, it may have been true that nfm was indexed just like fm, using
// wsPhysIndex. However, it is clear at present (7 Nov 2014) that nfm is
// definitely not indexed like fm. As an example, compare nfm in
// Albany::MechanicsProblem::constructNeumannEvaluators and fm in
// Albany::MechanicsProblem::buildProblem. For now, I'm going to keep nfm as an
// array, but this this new function is a wrapper around the unclear intended
// behavior.
inline Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>&
deref_nfm(
    Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>>& nfm,
    const WorksetArray<int>::type&                                          wsPhysIndex,
    int                                                                     ws)
{
  return nfm.size() == 1 ?          // Currently, all problems seem to have one nfm ...
             nfm[0] :               // ... hence this is the intended behavior ...
             nfm[wsPhysIndex[ws]];  // ... and this is not, but may one day be
                                    // again.
}

// Convenience routine for setting dfm workset data. Cut down on redundant code.
void
dfm_set(
    PHAL::Workset&                          workset,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& xd,
    Teuchos::RCP<Thyra_Vector const> const& xdd,
    Teuchos::RCP<AAdapt::rc::Manager>&      rc_mgr)
{
  workset.x                 = Teuchos::nonnull(rc_mgr) ? rc_mgr->add_x(x) : x;
  workset.xdot              = Teuchos::nonnull(rc_mgr) ? rc_mgr->add_x(xd) : xd;
  workset.xdotdot           = Teuchos::nonnull(rc_mgr) ? rc_mgr->add_x(xdd) : xdd;
  workset.transientTerms    = Teuchos::nonnull(xd);
  workset.accelerationTerms = Teuchos::nonnull(xdd);
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
  *Teuchos::VerboseObjectBase::getDefaultOStream()
      << "Albany::Application Check Derivatives level " << check_lvl << ":\n"
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

PHAL::Workset
Application::set_dfm_workset(
    double const                            current_time,
    Teuchos::RCP<Thyra_Vector const> const  x,
    Teuchos::RCP<Thyra_Vector const> const  x_dot,
    Teuchos::RCP<Thyra_Vector const> const  x_dotdot,
    Teuchos::RCP<Thyra_Vector> const&       f,
    Teuchos::RCP<Thyra_Vector const> const& x_post_SDBCs)
{
  PHAL::Workset workset;

  workset.f = f;

  loadWorksetNodesetInfo(workset);

  if (scaleBCdofs == true) {
    setScaleBCDofs(workset);
    countScale++;
  }

  if (x_post_SDBCs == Teuchos::null) {
    dfm_set(workset, x, x_dot, x_dotdot, rc_mgr);
  } else {
    dfm_set(workset, x_post_SDBCs, x_dot, x_dotdot, rc_mgr);
  }

  double const this_time = fixTime(current_time);

  workset.current_time       = this_time;
  workset.apps_              = apps_;
  workset.current_app_       = Teuchos::rcp(this, false);
  workset.distParamLib       = distParamLib;
  workset.disc               = disc;
  workset.spatial_dimension_ = getSpatialDimension();
  workset.numEqs             = neq;

  return workset;
}

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
  if (dfm != Teuchos::null) {
    evalName = PHAL::evalName<EvalT>("DFM", 0);
    phxSetup->insert_eval(evalName);

    dfm->postRegistrationSetupForType<EvalT>(*phxSetup);

    // Update phalanx saved/unsaved fields based on field dependencies
    phxSetup->check_fields(dfm->getFieldTagsForSizing<EvalT>());
    phxSetup->update_fields();

    writePhalanxGraph<EvalT>(dfm, evalName, phxGraphVisDetail);
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
  if (dfm != Teuchos::null) {
    evalName = PHAL::evalName<EvalT>("DFM", 0);
    phxSetup->insert_eval(evalName);

    // amb Need to look into this. What happens with DBCs in meshes having
    // different element types?
    std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(PHAL::getDerivativeDimensions<EvalT>(this, 0));
    dfm->setKokkosExtendedDataTypeDimensions<EvalT>(derivative_dimensions);
    dfm->postRegistrationSetupForType<EvalT>(*phxSetup);

    // Update phalanx saved/unsaved fields based on field dependencies
    phxSetup->check_fields(dfm->getFieldTagsForSizing<EvalT>());
    phxSetup->update_fields();

    writePhalanxGraph<EvalT>(dfm, evalName, phxGraphVisDetail);
  }
}

template <typename EvalT>
void
Application::writePhalanxGraph(
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>> fm,
    std::string const&                                  evalName,
    int const&                                          phxGraphVisDetail)
{
  if (phxGraphVisDetail > 0) {
    bool const detail = (phxGraphVisDetail > 1) ? true : false;
    *out << "Phalanx writing graphviz file for graph of " << evalName << " (detail = " << phxGraphVisDetail << ")"
         << std::endl;
    std::string const graphName = "phalanxGraph" + evalName;
    *out << "Process using 'dot -Tpng -O " << graphName << std::endl;
    fm->writeGraphvizFile<EvalT>(graphName, detail, detail);

    // Print phalanx setup info
    phxSetup->print(*out);
  }
}

void
Application::computeGlobalResidualImpl(
    double const                           current_time,
    Teuchos::RCP<Thyra_Vector const> const x,
    Teuchos::RCP<Thyra_Vector const> const x_dot,
    Teuchos::RCP<Thyra_Vector const> const x_dotdot,
    Teuchos::Array<ParamVec> const&        p,
    Teuchos::RCP<Thyra_Vector> const&      f,
    double                                 dt)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany Fill: Residual");
  using EvalT = PHAL::AlbanyTraits::Residual;
  postRegSetup<EvalT>();

  // Load connectivity map and coordinates
  const auto& wsElNodeEqID = disc->getWsElNodeEqID();
  const auto& wsPhysIndex  = disc->getWsPhysIndex();

  int const numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Thyra_Vector> const overlapped_f = solMgr->get_overlapped_f();

  Teuchos::RCP<const CombineAndScatterManager> cas_manager = solMgr->get_cas_manager();

  // Scatter x and xdot to the overlapped distrbution
  solMgr->scatterX(*x, x_dot.ptr(), x_dotdot.ptr());

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

    Teuchos::RCP<Thyra_Vector> x_post_SDBCs;
    if ((dfm != Teuchos::null) && (problem->useSDBCs() == true)) {
      workset = set_dfm_workset(current_time, x, x_dot, x_dotdot, f);

      // FillType template argument used to specialize Sacado
      dfm->preEvaluate<EvalT>(workset);
      x_post_SDBCs = workset.x->clone_v();
      loadBasicWorksetInfoSDBCs(workset, x_post_SDBCs, this_time);
    }

    workset.time_step = dt;

    workset.f = overlapped_f;

    for (int ws = 0; ws < numWorksets; ws++) {
      std::string const evalName = PHAL::evalName<EvalT>("FM", wsPhysIndex[ws]);
      loadWorksetBucketInfo<EvalT>(workset, ws, evalName);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<EvalT>(workset);

      if (nfm != Teuchos::null) {
        deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<EvalT>(workset);
      }
    }
  }

  // Assemble the residual into a non-overlapping vector
  {
    TEUCHOS_FUNC_TIME_MONITOR("Albany Residual Fill: Export");
    cas_manager->combine(overlapped_f, f, CombineMode::ADD);
  }

  // Allocate scaleVec_
#if defined(ALBANY_MPI)
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
#else
  ALBANY_ASSERT(scale == 1.0, "non-unity scale implementation requires MPI!");
#endif

  if (scaleBCdofs == false && scale != 1.0) {
    Thyra::ele_wise_scale<ST>(*scaleVec_, f.ptr());
  }

  // Push the assembled residual values back into the overlap vector
  cas_manager->scatter(f, overlapped_f, CombineMode::INSERT);
  // Write the residual to the discretization, which will later (optionally)
  // be written to the output file
  disc->setResidualField(*overlapped_f);

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)

  if (dfm != Teuchos::null) {
    PHAL::Workset workset = set_dfm_workset(current_time, x, x_dot, x_dotdot, f);

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<EvalT>(workset);
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
    double const                            dt)
{
  this->computeGlobalResidualImpl(current_time, x, x_dot, x_dotdot, p, f, dt);

  // Debut output write residual or solution to MatrixMarket
  // every time it arises or at requested count#
  auto const write_sol_mm =
      writeToMatrixMarketSol != 0 && (writeToMatrixMarketSol == -1 || countSol == writeToMatrixMarketSol);

  if (write_sol_mm == true) {
    *out << "Writing global solution #" << countSol << " to MatrixMarket at time t = " << current_time << ".\n";
    writeMatrixMarket(x, "sol", countSol);
  }
  auto const write_sol_co = writeToCoutSol != 0 && (writeToCoutSol == -1 || countSol == writeToCoutSol);
  if (write_sol_co == true) {
    *out << "Global solution #" << countSol << " corresponding to time t = " << current_time << ":\n";
    describe(x.getConst(), *out, Teuchos::VERB_EXTREME);
  }
  if (writeToMatrixMarketSol != 0 || writeToCoutSol != 0) {
    countSol++;
  }

  auto const write_res_mm =
      writeToMatrixMarketRes != 0 && (writeToMatrixMarketRes == -1 || countRes == writeToMatrixMarketRes);

  if (write_res_mm == true) {
    *out << "Writing global residual #" << countRes << " to MatrixMarket at time t = " << current_time << ".\n";
    writeMatrixMarket(f, "rhs", countRes);
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
    for (int ws = 0; ws < numWorksets; ws++) {
      std::string const evalName = PHAL::evalName<EvalT>("FM", wsPhysIndex[ws]);
      loadWorksetBucketInfo<EvalT>(workset, ws, evalName);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<EvalT>(workset);
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

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (Teuchos::nonnull(dfm)) {
    PHAL::Workset workset;

    workset.f       = f;
    workset.Jac     = jac;
    workset.m_coeff = alpha;
    workset.n_coeff = omega;
    workset.j_coeff = beta;

    double const this_time = fixTime(current_time);

    workset.current_time = this_time;

    if (beta == 0.0 && perturbBetaForDirichlets > 0.0) workset.j_coeff = perturbBetaForDirichlets;

    dfm_set(workset, x, xdot, xdotdot, rc_mgr);

    loadWorksetNodesetInfo(workset);

    if (scaleBCdofs == true) {
      setScaleBCDofs(workset, jac);
      countScale++;
    }

    workset.distParamLib = distParamLib;
    workset.disc         = disc;

    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_        = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<EvalT>(workset);
  }
  fillComplete(jac);

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
  responses[response_index]->evaluateResponse(this_time, x, xdot, xdotdot, p, g);
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
  DummyParameterAccessor<PHAL::AlbanyTraits::Jacobian, SPL_Traits>* dJ =
      new DummyParameterAccessor<PHAL::AlbanyTraits::Jacobian, SPL_Traits>();

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
  ALBANY_PANIC(
      index == -1,
      "Error in GatherCoordinateVector::getValue, \n"
          << "   Unrecognized param name: " << name << std::endl);

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
        secondOrder != "No" && secondOrder != "Velocity Verlet" && secondOrder != "Newmark" &&
            secondOrder != "Trapezoid Rule",
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
      piroSolverToken = (secondOrder == "No") ? "Rythmos" : secondOrder;
    } else if (solMethod == TransientTempus) {
      piroSolverToken = (secondOrder == "No") ? "Tempus" : secondOrder;
    } else {
      // Piro cannot handle the corresponding problem
      piroSolverToken = "Unsupported";
    }
    piroParams->set("Solver Type", piroSolverToken);
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
  workset.current_time      = current_time;
  workset.distParamLib      = distParamLib;
  workset.disc              = disc;
  workset.transientTerms    = Teuchos::nonnull(workset.xdot);
  workset.accelerationTerms = Teuchos::nonnull(workset.xdotdot);
}

void
Application::loadBasicWorksetInfoSDBCs(
    PHAL::Workset&                          workset,
    Teuchos::RCP<Thyra_Vector const> const& owned_sol,
    double const                            current_time)
{
  // Scatter owned solution into the overlapped one
  auto overlapped_MV  = solMgr->getOverlappedSolution();
  auto overlapped_sol = Thyra::createMember(overlapped_MV->range());
  overlapped_sol->assign(0.0);
  solMgr->get_cas_manager()->scatter(owned_sol, overlapped_sol, CombineMode::INSERT);

  auto numVectors = overlapped_MV->domain()->dim();
  workset.x       = overlapped_sol;
  workset.xdot    = numVectors > 1 ? overlapped_MV->col(1) : Teuchos::null;
  workset.xdotdot = numVectors > 2 ? overlapped_MV->col(2) : Teuchos::null;

  workset.numEqs            = neq;
  workset.current_time      = current_time;
  workset.distParamLib      = distParamLib;
  workset.disc              = disc;
  workset.transientTerms    = Teuchos::nonnull(workset.xdot);
  workset.accelerationTerms = Teuchos::nonnull(workset.xdotdot);
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
Application::loadWorksetNodesetInfo(PHAL::Workset& workset)
{
  workset.nodeSets      = Teuchos::rcpFromRef(disc->getNodeSets());
  workset.nodeSetCoords = Teuchos::rcpFromRef(disc->getNodeSetCoords());
  workset.nodeSetGIDs   = Teuchos::rcpFromRef(disc->getNodeSetGIDs());
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
Application::setScaleBCDofs(PHAL::Workset& workset, Teuchos::RCP<const Thyra_LinearOp> jac)
{
  // First step: set scaleVec_ to all 1.0s if it is all 0s
  if (scaleVec_->norm_2() == 0) {
    scaleVec_->assign(1.0);
  }

  // If calling setScaleBCDofs with null Jacobian, don't recompute the scaling
  if (jac == Teuchos::null) {
    return;
  }

  // For diagonal or abs row sum scaling, set the scale equal to the maximum
  // magnitude value of the diagonal / abs row sum (inf-norm).  This way,
  // scaling adjusts throughout the simulation based on the Jacobian.
  Teuchos::RCP<Thyra_Vector> tmp = Thyra::createMember(scaleVec_->space());
  if (scale_type == DIAG) {
    getDiagonalCopy(jac, tmp);
    scale = tmp->norm_inf();
  } else if (scale_type == ABSROWSUM) {
    // We MUST be able to cast the linear op to RowStatLinearOpBase, in order to
    // get row informations
    auto jac_row_stat = Teuchos::rcp_dynamic_cast<const Thyra::RowStatLinearOpBase<ST>>(jac, true);

    // Compute the absolute row sum
    jac_row_stat->getRowStat(Thyra::RowStatLinearOpBaseUtils::ROW_STAT_ROW_SUM, tmp.ptr());
    scale = tmp->norm_inf();
  }

  if (scale == 0.0) {
    scale = 1.0;
  }

  auto scaleVecLocalData = getNonconstLocalData(scaleVec_);
  for (size_t ns = 0; ns < nodeSetIDs_.size(); ns++) {
    std::string key = nodeSetIDs_[ns];

    std::vector<std::vector<int>> const& nsNodes = workset.nodeSets->find(key)->second;
    for (unsigned int i = 0; i < nsNodes.size(); i++) {
      for (unsigned j = 0; j < offsets_[ns].size(); j++) {
        int lunk                = nsNodes[i][offsets_[ns][j]];
        scaleVecLocalData[lunk] = scale;
      }
    }
  }

  if (problem->getSideSetEquations().size() > 0) {
    ALBANY_ABORT(
        "Application::setScaleBCDofs is not yet implemented for"
        << " sideset equations!\n");
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

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i = 0; i < p.size(); i++) {
    for (unsigned int j = 0; j < p[i].size(); j++) {
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);
    }
  }

  double const this_time    = fixTime(current_time);
  workset.current_time      = this_time;
  workset.x                 = overlapped_x;
  workset.xdot              = overlapped_xdot;
  workset.xdotdot           = overlapped_xdotdot;
  workset.distParamLib      = distParamLib;
  workset.disc              = disc;
  workset.transientTerms    = Teuchos::nonnull(workset.xdot);
  workset.accelerationTerms = Teuchos::nonnull(workset.xdotdot);
  workset.comm              = comm;
  workset.x_cas_manager     = solMgr->get_cas_manager();
}

void
Application::setCoupledAppBlockNodeset(
    std::string const& app_name,
    std::string const& block_name,
    std::string const& nodeset_name)
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
