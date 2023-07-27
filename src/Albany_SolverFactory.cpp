// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_SolverFactory.hpp"

#include "ACE_ThermoMechanical.hpp"
#include "Albany_Application.hpp"
#include "Albany_Macros.hpp"
#include "Albany_ModelEvaluator.hpp"
#include "Albany_PiroObserver.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Utils.hpp"
#include "Piro_NOXSolver.hpp"
#include "Piro_ProviderBase.hpp"
#include "Piro_SolverFactory.hpp"
#include "Piro_StratimikosUtils.hpp"
#include "Schwarz_Alternating.hpp"
#include "Schwarz_Coupled.hpp"
#include "Schwarz_PiroObserver.hpp"
#include "Stratimikos_DefaultLinearSolverBuilder.hpp"
#include "Teuchos_AbstractFactoryStd.hpp"
#include "Thyra_Ifpack2PreconditionerFactory.hpp"

#if defined(ALBANY_MUELU)
#include "Stratimikos_MueLuHelpers.hpp"
#endif /* ALBANY_MUELU */

#if defined(ALBANY_FROSCH)
#include "Stratimikos_FROSch_decl.hpp"
#include "Stratimikos_FROSch_def.hpp"
#endif /* ALBANY_FROSCH */

#include "Albany_Macros.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListHelpers.hpp"
#include "Thyra_DefaultModelEvaluatorWithSolveFactory.hpp"
#include "Thyra_DetachedVectorView.hpp"

namespace {

void
enableIfpack2(Stratimikos::DefaultLinearSolverBuilder& linearSolverBuilder)
{
  typedef Thyra::PreconditionerFactoryBase<ST>                  Base;
  typedef Thyra::Ifpack2PreconditionerFactory<Tpetra_CrsMatrix> Impl;
  linearSolverBuilder.setPreconditioningStrategyFactory(Teuchos::abstractFactoryStd<Base, Impl>(), "Ifpack2");
}

void
enableMueLu(Stratimikos::DefaultLinearSolverBuilder& linearSolverBuilder)
{
#if defined(ALBANY_MUELU)
  Stratimikos::enableMueLu<LO, Tpetra_GO, KokkosNode>(linearSolverBuilder);
#endif
}

void
enableFROSch(Stratimikos::DefaultLinearSolverBuilder& linearSolverBuilder)
{
#if defined(ALBANY_FROSCH)
  Stratimikos::enableFROSch<LO, Tpetra_GO, KokkosNode>(linearSolverBuilder);
#endif
}
}  // namespace

namespace Albany {

SolverFactory::SolverFactory(std::string const& inputFile, const Teuchos::RCP<Teuchos_Comm const>& comm) : out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  // Set up application parameters: read and broadcast input file, and set
  // defaults
  // Teuchos::RCP<Teuchos::ParameterList> input_
  appParams = Teuchos::createParameterList("Albany Parameters");

  std::string const input_extension = getFileExtension(inputFile);

  if (input_extension == "yaml" || input_extension == "yml") {
    Teuchos::updateParametersFromYamlFileAndBroadcast(inputFile, appParams.ptr(), *comm);
  } else {
    Teuchos::updateParametersFromXmlFileAndBroadcast(inputFile, appParams.ptr(), *comm);
  }

  // do not set default solver parameters for ATO::Solver problems,
  // ... as they handle this themselves
  std::string solution_method = appParams->sublist("Problem").get("Solution Method", "Steady");
  if (solution_method != "ATO Problem") {
    Teuchos::RCP<Teuchos::ParameterList> defaultSolverParams = rcp(new Teuchos::ParameterList());
    setSolverParamDefaults(defaultSolverParams.get(), comm->getRank());
    appParams->setParametersNotAlreadySet(*defaultSolverParams);
  }
  appParams->validateParametersAndSetDefaults(*getValidAppParameters(), 0);
  if (appParams->isSublist("Debug Output")) {
    Teuchos::RCP<Teuchos::ParameterList> debugPL = Teuchos::rcpFromRef(appParams->sublist("Debug Output", false));
    debugPL->validateParametersAndSetDefaults(*getValidDebugParameters(), 0);
  }
  if (appParams->isSublist("Scaling")) {
    Teuchos::RCP<Teuchos::ParameterList> scalingPL = Teuchos::rcpFromRef(appParams->sublist("Scaling", false));
    scalingPL->validateParametersAndSetDefaults(*getValidScalingParameters(), 0);
  }
}

SolverFactory::SolverFactory(const Teuchos::RCP<Teuchos::ParameterList>& input_appParams, const Teuchos::RCP<Teuchos_Comm const>& comm)
    : appParams(input_appParams), out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  // do not set default solver parameters for ATO::Solver
  // problems,
  // ... as they handle this themselves
  std::string solution_method = appParams->sublist("Problem").get("Solution Method", "Steady");
  if (solution_method != "ATO Problem") {
    Teuchos::RCP<Teuchos::ParameterList> defaultSolverParams = rcp(new Teuchos::ParameterList());
    setSolverParamDefaults(defaultSolverParams.get(), comm->getRank());
    appParams->setParametersNotAlreadySet(*defaultSolverParams);
  }
  appParams->validateParametersAndSetDefaults(*getValidAppParameters(), 0);
  if (appParams->isSublist("Debug Output")) {
    Teuchos::RCP<Teuchos::ParameterList> debugPL = Teuchos::rcpFromRef(appParams->sublist("Debug Output", false));
    debugPL->validateParametersAndSetDefaults(*getValidDebugParameters(), 0);
  }
}

Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>
SolverFactory::create(
    const Teuchos::RCP<Teuchos_Comm const>& appComm,
    const Teuchos::RCP<Teuchos_Comm const>& solverComm,
    Teuchos::RCP<Thyra_Vector const> const& initial_guess)
{
  Teuchos::RCP<Application> dummyAlbanyApp;
  return createAndGetAlbanyApp(dummyAlbanyApp, appComm, solverComm, initial_guess);
}

Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>
SolverFactory::createAndGetAlbanyApp(
    Teuchos::RCP<Application>&              albanyApp,
    const Teuchos::RCP<Teuchos_Comm const>& appComm,
    const Teuchos::RCP<Teuchos_Comm const>& solverComm,
    Teuchos::RCP<Thyra_Vector const> const& initial_guess,
    bool                                    createAlbanyApp,
    double const                            init_time)
{
  const Teuchos::RCP<Teuchos::ParameterList> problemParams  = Teuchos::sublist(appParams, "Problem");
  std::string const                          solutionMethod = problemParams->get("Solution Method", "Steady");

  bool const is_schwarz = solutionMethod == "Coupled Schwarz" || solutionMethod == "Schwarz Alternating";

  bool const is_ace_thermo_mech = solutionMethod == "ACE Sequential Thermo-Mechanical";

  if (is_schwarz == true) {
#if !defined(ALBANY_DTK)
    ALBANY_ASSERT(appComm->getSize() == 1, "Parallel Schwarz requires DTK");
#endif  // ALBANY_DTK
  }
  if (solutionMethod == "Coupled Schwarz") {
    // IKT: We are assuming the "Piro" list will come from the main coupled
    // Schwarz input file (not the sub-input
    // files for each model).
    const Teuchos::RCP<Teuchos::ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");

    const Teuchos::RCP<Teuchos::ParameterList> stratList = Piro::extractStratimikosParams(piroParams);
    // Create and setup the Piro solver factory
    Piro::SolverFactory piroFactory;
    // Setup linear solver
    Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
    enableIfpack2(linearSolverBuilder);
    enableMueLu(linearSolverBuilder);

    linearSolverBuilder.setParameterList(stratList);

    const Teuchos::RCP<Thyra_LOWS_Factory> lowsFactory = createLinearSolveStrategy(linearSolverBuilder);

    const Teuchos::RCP<LCM::SchwarzCoupled> coupled_model_with_solve = Teuchos::rcp(new LCM::SchwarzCoupled(appParams, solverComm, initial_guess, lowsFactory));

    observer_ = Teuchos::rcp(new LCM::Schwarz_PiroObserver(coupled_model_with_solve));

    // WARNING: Coupled Schwarz does not contain a primary Application
    // instance and so albanyApp is null.
    return piroFactory.createSolver<ST>(piroParams, coupled_model_with_solve, Teuchos::null, observer_);
  }

  if (solutionMethod == "Schwarz Alternating") {
    return Teuchos::rcp(new LCM::SchwarzAlternating(appParams, solverComm));
  }

  if (solutionMethod == "ACE Sequential Thermo-Mechanical") {
    return Teuchos::rcp(new LCM::ACEThermoMechanical(appParams, solverComm));
  }

  model_ = createAlbanyAppAndModel(albanyApp, appComm, initial_guess, createAlbanyApp, init_time);

  const Teuchos::RCP<Teuchos::ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");
  const Teuchos::RCP<Teuchos::ParameterList> stratList  = Piro::extractStratimikosParams(piroParams);

  if (Teuchos::is_null(stratList)) {
    *out << "Error: cannot locate Stratimikos solver parameters in the input "
            "file."
         << std::endl;
    *out << "Printing the Piro parameter list:" << std::endl;
    piroParams->print(*out);
    // GAH: this is an error - should be fatal
    ALBANY_ABORT(
        "Error: cannot locate Stratimikos solver parameters in the input file."
        << "\n");
  }

  Teuchos::RCP<Thyra_ModelEvaluator> modelWithSolve;
  if (Teuchos::nonnull(model_->get_W_factory())) {
    modelWithSolve = model_;
  } else {
    // Setup linear solver
    Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
    enableIfpack2(linearSolverBuilder);
    enableMueLu(linearSolverBuilder);
    enableFROSch(linearSolverBuilder);
    linearSolverBuilder.setParameterList(stratList);

    const Teuchos::RCP<Thyra_LOWS_Factory> lowsFactory = createLinearSolveStrategy(linearSolverBuilder);

    modelWithSolve = rcp(new Thyra::DefaultModelEvaluatorWithSolveFactory<ST>(model_, lowsFactory));
  }

  const auto solMgr = albanyApp->getAdaptSolMgr();

  Piro::SolverFactory piroFactory;
  observer_ = Teuchos::rcp(new PiroObserver(albanyApp, modelWithSolve));
  if (solMgr->isAdaptive()) {
    return piroFactory.createSolverAdaptive<ST>(piroParams, modelWithSolve, Teuchos::null, solMgr, observer_);
  } else {
    return piroFactory.createSolver<ST>(piroParams, modelWithSolve, Teuchos::null, observer_);
  }
  ALBANY_ABORT(
      "Reached end of createAndGetAlbanyAppT()"
      << "\n");

  // Silence compiler warning in case it wasn't used (due to ifdef logic)
  (void)solverComm;

  return Teuchos::null;
}

Teuchos::RCP<Thyra_ModelEvaluator>
SolverFactory::createAlbanyAppAndModel(
    Teuchos::RCP<Application>&              albanyApp,
    const Teuchos::RCP<Teuchos_Comm const>& appComm,
    Teuchos::RCP<Thyra_Vector const> const& initial_guess,
    bool const                              createAlbanyApp, 
    const double                            init_time)
{
  if (createAlbanyApp) {
    std::cout << "IKT creating AlbanyApp\n"; 
    // Create application
    albanyApp = Teuchos::rcp(new Application(appComm, appParams, initial_guess, is_schwarz_, init_time));
    //  albanyApp = rcp(new ApplicationT(appComm, appParams,
    //  initial_guess));
  }

  // Validate Response list: may move inside individual Problem class
  Teuchos::RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(appParams, "Problem");
  problemParams->sublist("Response Functions").validateParameters(*getValidResponseParameters(), 0);

  // If not explicitly specified, determine which Piro solver to use from the
  // problem parameters
  const Teuchos::RCP<Teuchos::ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");
  if (!piroParams->getPtr<std::string>("Solver Type")) {
    std::string const solutionMethod = problemParams->get("Solution Method", "Steady");

    /* TODO: this should be a boolean, not a string ! */
    std::string const secondOrder = problemParams->get("Second Order", "No");
    ALBANY_PANIC(
        secondOrder != "No",
        "Second Order is not supported"
            << "\n");

    // Populate the Piro parameter list accordingly to inform the Piro solver
    // factory
    std::string piroSolverToken;
    if (solutionMethod == "Steady") {
      piroSolverToken = "NOX";
    } else if (solutionMethod == "Transient Tempus") {
      piroSolverToken = "Tempus";
    } else {
      // Piro cannot handle the corresponding problem
      piroSolverToken = "Unsupported";
    }

    ALBANY_ASSERT(piroSolverToken != "Unsupported", "Unsupported Solution Method: " << solutionMethod);

    piroParams->set("Solver Type", piroSolverToken);
  }

  // Create model evaluator
  return Teuchos::rcp(new ModelEvaluator(albanyApp, appParams));
}

int
SolverFactory::checkSolveTestResults(
    int                                          response_index,
    int                                          parameter_index,
    Teuchos::RCP<Thyra_Vector const> const&      g,
    const Teuchos::RCP<const Thyra_MultiVector>& dgdp) const
{
  Teuchos::ParameterList* testParams = getTestParameters(response_index);

  int          failures    = 0;
  int          comparisons = 0;
  double const relTol      = testParams->get<double>("Relative Tolerance");
  double const absTol      = testParams->get<double>("Absolute Tolerance");

  // Get number of responses (g) to test
  int const numResponseTests = testParams->get<int>("Number of Comparisons");
  if (numResponseTests > 0) {
    ALBANY_ASSERT(g != Teuchos::null, "There are Response Tests but the response vector is null!");
    ALBANY_ASSERT(
        numResponseTests <= g->space()->dim(),
        "Number of Response Tests (" << numResponseTests << ") greater than number of responses (" << g->space()->dim() << ") !");
    Teuchos::Array<double> testValues = testParams->get<Teuchos::Array<double>>("Test Values");

    ALBANY_ASSERT(
        numResponseTests == testValues.size(),
        "Number of Response Tests (" << numResponseTests << ") != number of Test Values (" << testValues.size() << ") !");

    Teuchos::ArrayRCP<const ST> g_view = getLocalData(g);
    for (int i = 0; i < testValues.size(); i++) {
      auto s = std::string("Response Test ") + std::to_string(i);
      failures += scaledCompare(g_view[i], testValues[i], relTol, absTol, s);
      comparisons++;
    }
  }

  // Repeat comparisons for sensitivities
  Teuchos::ParameterList* sensitivityParams        = 0;
  std::string             sensitivity_sublist_name = strint("Sensitivity Comparisons", parameter_index);
  if (parameter_index == 0 && !testParams->isSublist(sensitivity_sublist_name))
    sensitivityParams = testParams;
  else if (testParams->isSublist(sensitivity_sublist_name))
    sensitivityParams = &(testParams->sublist(sensitivity_sublist_name));
  int numSensTests = 0;
  if (sensitivityParams != 0) {
    numSensTests = sensitivityParams->get<int>("Number of Sensitivity Comparisons", 0);
  }
  if (numSensTests > 0) {
    ALBANY_ASSERT(
        dgdp != Teuchos::null, "There are Sensitivity Tests but the sensitivity vector (" << response_index << ", " << parameter_index << ") is null!");
    ALBANY_ASSERT(
        numSensTests <= dgdp->range()->dim(),
        "Number of sensitivity tests (" << numSensTests << ") != number of sensitivities [" << response_index << "][" << parameter_index << "] ("
                                        << dgdp->range()->dim() << ") !");
  }
  for (int i = 0; i < numSensTests; i++) {
    int const              numVecs        = dgdp->domain()->dim();
    Teuchos::Array<double> testSensValues = sensitivityParams->get<Teuchos::Array<double>>(strint("Sensitivity Test Values", i));
    ALBANY_ASSERT(
        numVecs == testSensValues.size(),
        "Number of Sensitivity Test Values (" << testSensValues.size() << " != number of sensitivity vectors (" << numVecs << ") !");
    auto dgdp_view = getLocalData(dgdp);
    for (int jvec = 0; jvec < numVecs; jvec++) {
      auto s = std::string("Sensitivity Test ") + std::to_string(i) + "," + std::to_string(jvec);
      failures += scaledCompare(dgdp_view[jvec][i], testSensValues[jvec], relTol, absTol, s);
      comparisons++;
    }
  }

  storeTestResults(testParams, failures, comparisons);

  return failures;
}

int
SolverFactory::checkDakotaTestResults(int response_index, const Teuchos::SerialDenseVector<int, double>* drdv) const
{
  Teuchos::ParameterList* testParams = getTestParameters(response_index);

  int          failures    = 0;
  int          comparisons = 0;
  double const relTol      = testParams->get<double>("Relative Tolerance");
  double const absTol      = testParams->get<double>("Absolute Tolerance");

  int const numDakotaTests = testParams->get<int>("Number of Dakota Comparisons");
  if (numDakotaTests > 0 && drdv != NULL) {
    ALBANY_ASSERT(numDakotaTests <= drdv->length(), "more Dakota Tests (" << numDakotaTests << ") than derivatives (" << drdv->length() << ") !\n");
    // Read accepted test results
    Teuchos::Array<double> testValues = testParams->get<Teuchos::Array<double>>("Dakota Test Values");

    ALBANY_PANIC(numDakotaTests != testValues.size());
    for (int i = 0; i < numDakotaTests; i++) {
      auto s = std::string("Dakota Test ") + std::to_string(i);
      failures += scaledCompare((*drdv)[i], testValues[i], relTol, absTol, s);
      comparisons++;
    }
  }

  storeTestResults(testParams, failures, comparisons);

  return failures;
}

int
SolverFactory::checkAnalysisTestResults(int response_index, const Teuchos::RCP<Thyra::VectorBase<double>>& tvec) const
{
  Teuchos::ParameterList* testParams = getTestParameters(response_index);

  int          failures    = 0;
  int          comparisons = 0;
  double const relTol      = testParams->get<double>("Relative Tolerance");
  double const absTol      = testParams->get<double>("Absolute Tolerance");

  int numPiroTests = testParams->get<int>("Number of Piro Analysis Comparisons");
  if (numPiroTests > 0 && tvec != Teuchos::null) {
    // Create indexable thyra vector
    ::Thyra::DetachedVectorView<double> p(tvec);

    ALBANY_ASSERT(numPiroTests <= p.subDim(), "more Piro Analysis Comparisons (" << numPiroTests << ") than values (" << p.subDim() << ") !\n");
    // Read accepted test results
    Teuchos::Array<double> testValues = testParams->get<Teuchos::Array<double>>("Piro Analysis Test Values");

    ALBANY_PANIC(numPiroTests != testValues.size());
    if (testParams->get<bool>("Piro Analysis Test Two Norm", false)) {
      const auto norm = tvec->norm_2();
      *out << "Parameter Vector Two Norm: " << norm << std::endl;
      failures += scaledCompare(norm, testValues[0], relTol, absTol, "Piro Analysis Test Two Norm");
      comparisons++;
    } else {
      for (int i = 0; i < numPiroTests; i++) {
        auto s = std::string("Piro Analysis Test ") + std::to_string(i);
        failures += scaledCompare(p[i], testValues[i], relTol, absTol, s);
        comparisons++;
      }
    }
  }

  storeTestResults(testParams, failures, comparisons);

  return failures;
}

Teuchos::ParameterList*
SolverFactory::getTestParameters(int response_index) const
{
  Teuchos::ParameterList* result;

  if (response_index == 0 && appParams->isSublist("Regression Results")) {
    result = &(appParams->sublist("Regression Results"));
  } else {
    result = &(appParams->sublist(strint("Regression Results", response_index)));
  }

  ALBANY_PANIC(result->isType<std::string>("Test Values"), "Array information in input file must now be of type Array(double)\n");
  result->validateParametersAndSetDefaults(*getValidRegressionResultsParameters(), 0);

  return result;
}

void
SolverFactory::storeTestResults(Teuchos::ParameterList* testParams, int failures, int comparisons) const
{
  // Store failures in param list (this requires mutable appParams!)
  testParams->set("Number of Failures", failures);
  testParams->set("Number of Comparisons Attempted", comparisons);
  *out << "\nCheckTestResults: Number of Comparisons Attempted = " << comparisons << std::endl;
}

bool
SolverFactory::scaledCompare(double x1, double x2, double relTol, double absTol, std::string const& name) const
{
  auto d       = fabs(x1 - x2);
  auto avg_mag = (0.5 * (fabs(x1) + fabs(x2)));
  auto rel_ok  = (d <= (avg_mag * relTol));
  auto abs_ok  = (d <= fabs(absTol));
  auto ok      = rel_ok || abs_ok;
  if (!ok) {
    *out << name << ": " << x1 << " != " << x2 << " (rel " << relTol << " abs " << absTol << ")\n";
  }
  return !ok;
}

void
SolverFactory::setSolverParamDefaults(Teuchos::ParameterList* appParams_, int myRank)
{
  // Set the nonlinear solver method
  Teuchos::ParameterList& piroParams = appParams_->sublist("Piro");
  Teuchos::ParameterList& noxParams  = piroParams.sublist("NOX");
  noxParams.set("Nonlinear Solver", "Line Search Based");

  // Set the printing parameters in the "Printing" sublist
  Teuchos::ParameterList& printParams = noxParams.sublist("Printing");
  printParams.set("MyPID", myRank);
  printParams.set("Output Precision", 3);
  printParams.set("Output Processor", 0);
  printParams.set(
      "Output Information",
      NOX::Utils::OuterIteration + NOX::Utils::OuterIterationStatusTest + NOX::Utils::InnerIteration + NOX::Utils::Parameters + NOX::Utils::Details +
          NOX::Utils::LinearSolverDetails + NOX::Utils::Warning + NOX::Utils::Error);

  // Sublist for line search
  Teuchos::ParameterList& searchParams = noxParams.sublist("Line Search");
  searchParams.set("Method", "Full Step");

  // Sublist for direction
  Teuchos::ParameterList& dirParams = noxParams.sublist("Direction");
  dirParams.set("Method", "Newton");
  Teuchos::ParameterList& newtonParams = dirParams.sublist("Newton");
  newtonParams.set("Forcing Term Method", "Constant");

  // Sublist for linear solver for the Newton method
  Teuchos::ParameterList& lsParams = newtonParams.sublist("Linear Solver");
  lsParams.set("Max Iterations", 43);
  lsParams.set("Tolerance", 1e-4);
}

Teuchos::RCP<Teuchos::ParameterList const>
SolverFactory::getValidAppParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("ValidAppParams"));

  validPL->sublist("Problem", false, "Problem sublist");
  validPL->sublist("Debug Output", false, "Debug Output sublist");
  validPL->sublist("Scaling", false, "Jacobian/Residual Scaling sublist");
  validPL->sublist("DataTransferKit", false, "DataTransferKit sublist");
  validPL->sublist("DataTransferKit", false, "DataTransferKit sublist").sublist("Consistent Interpolation", false, "DTK Consistent Interpolation sublist");
  validPL->sublist("DataTransferKit", false, "DataTransferKit sublist").sublist("Search", false, "DTK Search sublist");
  validPL->sublist("DataTransferKit", false, "DataTransferKit sublist").sublist("L2 Projection", false, "DTK L2 Projection sublist");
  validPL->sublist("DataTransferKit", false, "DataTransferKit sublist").sublist("Point Cloud", false, "DTK Point Cloud sublist");
  validPL->sublist("Discretization", false, "Discretization sublist");
  validPL->sublist("Quadrature", false, "Quadrature sublist");
  validPL->sublist("Regression Results", false, "Regression Results sublist");
  validPL->sublist("VTK", false, "DEPRECATED  VTK sublist");
  validPL->sublist("Piro", false, "Piro sublist");
  validPL->sublist("Coupled System", false, "Coupled system sublist");
  validPL->sublist("Alternating System", false, "Alternating system sublist");
  validPL->set<bool>("Enable TimeMonitor Output", false, "Flag to enable TimeMonitor output");

  // validPL->set<std::string>("Jacobian Operator", "Have Jacobian", "Flag to
  // allow Matrix-Free specification in Piro");
  // validPL->set<double>("Matrix-Free Perturbation", 3.0e-7, "delta in
  // matrix-free formula");

  return validPL;
}

Teuchos::RCP<Teuchos::ParameterList const>
SolverFactory::getValidDebugParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("ValidDebugParams"));
  validPL->set<int>("Write Jacobian to MatrixMarket", 0, "Jacobian Number to Dump to MatrixMarket");
  validPL->set<int>("Compute Jacobian Condition Number", 0, "Jacobian Condition Number to Compute");
  validPL->set<int>("Write Residual to MatrixMarket", 0, "Residual Number to Dump to MatrixMarket");
  validPL->set<int>("Write Jacobian to Standard Output", 0, "Jacobian Number to Dump to Standard Output");
  validPL->set<int>("Write Residual to Standard Output", 0, "Residual Number to Dump to Standard Output");
  validPL->set<int>("Derivative Check", 0, "Derivative check");
  validPL->set<int>("Write Solution to MatrixMarket", 0, "Solution Number to Dump to MatrixMarket");
  validPL->set<bool>("Write Distributed Solution and Map to MatrixMarket", false, "Flag to Write Distributed Solution and Map to MatrixMarket");
  validPL->set<int>("Write Solution to Standard Output", 0, "Solution Number to Dump to  Standard Output");
  validPL->set<bool>("Analyze Memory", false, "Flag to Analyze Memory");
  return validPL;
}

Teuchos::RCP<Teuchos::ParameterList const>
SolverFactory::getValidScalingParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("ValidScalingParams"));
  validPL->set<double>("Scale", 0.0, "Value of Scaling to Apply to Jacobian/Residual");
  validPL->set<bool>("Scale BC Dofs", false, "Flag to Scale Jacobian/Residual Rows Corresponding to DBC Dofs");
  validPL->set<std::string>("Type", "Constant", "Scaling Type (Constant, Diagonal, AbsRowSum)");
  return validPL;
}

Teuchos::RCP<Teuchos::ParameterList const>
SolverFactory::getValidRegressionResultsParameters() const
{
  using Teuchos::Array;
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("ValidRegressionParams"));
  ;
  Array<double> ta;
  ;  // std::string to be converted to teuchos array

  validPL->set<double>("Relative Tolerance", 1.0e-4, "Relative Tolerance used in regression testing");
  validPL->set<double>("Absolute Tolerance", 1.0e-8, "Absolute Tolerance used in regression testing");

  validPL->set<int>("Number of Comparisons", 0, "Number of responses to regress against");
  validPL->set<Array<double>>("Test Values", ta, "Array of regression values for responses");

  validPL->set<int>("Number of Sensitivity Comparisons", 0, "Number of sensitivity vectors to regress against");

  int const maxSensTests = 10;
  for (int i = 0; i < maxSensTests; i++) {
    validPL->set<Array<double>>(strint("Sensitivity Test Values", i), ta, strint("Array of regression values for Sensitivities w.r.t parameter", i));
    validPL->sublist(strint("Sensitivity Comparisons", i), false, "Sensitivity Comparisons sublist");
  }

  validPL->set<int>("Number of Dakota Comparisons", 0, "Number of parameters from Dakota runs to regress against");
  validPL->set<Array<double>>("Dakota Test Values", ta, "Array of regression values for final parameters from Dakota runs");

  validPL->set<int>("Number of Piro Analysis Comparisons", 0, "Number of parameters from Analysis to regress against");
  validPL->set<bool>("Piro Analysis Test Two Norm", false, "Test l2 norm of final parameters from Analysis runs");
  validPL->set<Array<double>>("Piro Analysis Test Values", ta, "Array of regression values for final parameters from Analysis runs");

  // Should deprecate these options, but need to remove them from all input
  // files
  validPL->set<int>("Number of Stochastic Galerkin Comparisons", 0, "Number of stochastic Galerkin expansions to regress against");

  int const maxSGTests = 10;
  for (int i = 0; i < maxSGTests; i++) {
    validPL->set<Array<double>>(
        strint("Stochastic Galerkin Expansion Test Values", i), ta, strint("Array of regression values for stochastic Galerkin expansions", i));
  }

  validPL->set<int>("Number of Stochastic Galerkin Mean Comparisons", 0, "Number of SG mean responses to regress against");
  validPL->set<Array<double>>("Stochastic Galerkin Mean Test Values", ta, "Array of regression values for SG mean responses");
  validPL->set<int>("Number of Stochastic Galerkin Standard Deviation Comparisons", 0, "Number of SG standard deviation responses to regress against");
  validPL->set<Array<double>>("Stochastic Galerkin Standard Deviation Test Values", ta, "Array of regression values for SG standard deviation responses");
  // End of deprecated Stochastic Galerkin Options

  // These two are typically not set on input, just output.
  validPL->set<int>(
      "Number of Failures",
      0,
      "Output information from regression tests reporting number of failed "
      "tests");
  validPL->set<int>(
      "Number of Comparisons Attempted",
      0,
      "Output information from regression tests reporting number of "
      "comparisons attempted");

  return validPL;
}

Teuchos::RCP<Teuchos::ParameterList const>
SolverFactory::getValidParameterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("ValidParameterParams"));
  ;

  validPL->set<int>("Number", 0);
  int const maxParameters = 100;
  for (int i = 0; i < maxParameters; i++) {
    validPL->set<std::string>(strint("Parameter", i), "");
  }
  return validPL;
}

Teuchos::RCP<Teuchos::ParameterList const>
SolverFactory::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("ValidResponseParams"));
  ;
  validPL->set<std::string>("Collection Method", "Sum Responses");
  validPL->set<int>("Number of Response Vectors", 0);
  validPL->set<bool>("Observe Responses", true);
  validPL->set<int>("Responses Observation Frequency", 1);
  Teuchos::Array<unsigned int> defaultDataUnsignedInt;
  validPL->set<Teuchos::Array<unsigned int>>(
      "Relative Responses Markers", defaultDataUnsignedInt, "Array of responses for which relative change will be obtained");

  validPL->set<int>("Number", 0);
  validPL->set<int>("Equation", 0);
  int const maxParameters = 500;
  for (int i = 0; i < maxParameters; i++) {
    validPL->set<std::string>(strint("Response", i), "");
    validPL->sublist(strint("ResponseParams", i));
    validPL->sublist(strint("Response Vector", i));
  }
  return validPL;
}

}  // namespace Albany
