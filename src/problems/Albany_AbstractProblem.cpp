// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include "Albany_AbstractProblem.hpp"

#include "NOX_StatusTest_Generic.H"

// Generic implementations that can be used by derived problems

Albany::AbstractProblem::AbstractProblem(
    const Teuchos::RCP<Teuchos::ParameterList>& params_,
    const Teuchos::RCP<ParamLib>&               paramLib_,
    // const Teuchos::RCP<DistributedParameterLibrary>& distParamLib_,
    int const neq_)
    : out(Teuchos::VerboseObjectBase::getDefaultOStream()),
      neq(neq_),
      number_of_time_deriv(-1),
      SolutionMethodName(Unknown),
      params(params_),
      paramLib(paramLib_),
      // distParamLib(distParamLib_),
      rigidBodyModes(Teuchos::rcp(new Albany::RigidBodyModes(neq_)))
{
  /*
   * Set the number of time derivatives. Semantics are to set the number of time
   * derivatives: x = 0, xdot = 1, xdotdot = 2 using the Discretization
   * parameter "Number Of Time Derivatives" if this is specified, or if not set
   * it to zero if the problem is steady, or to one if it is transient. This
   * needs to be overridden in each problem is this logic is not sufficient.
   */

  /* Override this logic by specifying the below in the Discretization PL with

  <Parameter name="Number Of Time Derivatives" type="int" value="2"/>
  */

  std::string solutionMethod = params->get("Solution Method", "Steady");
  if (solutionMethod == "Steady") {
    number_of_time_deriv = 0;
    SolutionMethodName   = Steady;
  } else if (solutionMethod == "Continuation") {
    number_of_time_deriv = 0;
    SolutionMethodName   = Continuation;
  } else if (solutionMethod == "Transient") {
    number_of_time_deriv = 1;
    SolutionMethodName   = Transient;
  } else if (solutionMethod == "Transient Tempus" || solutionMethod == "Transient Tempus No Piro") {
    number_of_time_deriv = 1;
    SolutionMethodName   = TransientTempus;
  } else if (solutionMethod == "Eigensolve") {
    number_of_time_deriv = 0;
    SolutionMethodName   = Eigensolve;
  } else if (solutionMethod == "Aeras Hyperviscosity") {
    number_of_time_deriv = 1;
    SolutionMethodName   = AerasHyperviscosity;
  } else
    ALBANY_ABORT(
        "Solution Method must be Steady, Transient, Transient Tempus, " << "Continuation, Eigensolve, or Aeras Hyperviscosity, not : " << solutionMethod);

  // Set the number in the Problem PL
  params->set<int>("Number Of Time Derivatives", number_of_time_deriv);
}

unsigned int
Albany::AbstractProblem::numEquations() const
{
  ALBANY_PANIC(neq <= 0, "A Problem must have at least 1 equation: " << neq);
  return neq;
}

std::map<int, std::vector<std::string>> const&
Albany::AbstractProblem::getSideSetEquations() const
{
  return sideSetEquations;
}

void
Albany::AbstractProblem::setNumEquations(int const neq_)
{
  neq = neq_;
  rigidBodyModes->setNumPDEs(neq_);
}

// Get the solution method type name
Albany::SolutionMethodType
Albany::AbstractProblem::getSolutionMethod()
{
  return SolutionMethodName;
}

Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>>
Albany::AbstractProblem::getFieldManager()
{
  return fm;
}

Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
Albany::AbstractProblem::getDirichletFieldManager()
{
  return dfm;
}

Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>>
Albany::AbstractProblem::getNeumannFieldManager()
{
  return nfm;
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::AbstractProblem::getGenericProblemParams(std::string listname) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::rcp(new Teuchos::ParameterList(listname));
  ;
  validPL->set<std::string>("Name", "", "String to designate Problem Class");
  // The following is for LandIce problems.
  validPL->set<int>("Number RBMs for ML", 0, "Number of RBMs provided to ML");
  validPL->set<int>("Number of Spatial Processors", -1, "Number of spatial processors in multi-level parallelism");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Flag to select outpuy of Phalanx Graph and level of detail");
  validPL->set<bool>(
      "Use Physics-Based Preconditioner",
      false,
      "Flag to create signal that this problem will creat its own "
      "preconditioner");
  validPL->set<std::string>("Physics-Based Preconditioner", "None", "Type of preconditioner that problem will create");
  Teuchos::RCP<Albany::Application> dummy_app;
  validPL->set<Teuchos::RCP<Albany::Application>>("Application", dummy_app, "Application to couple to");

  validPL->set<Teuchos::Array<std::string>>("Required Fields", Teuchos::Array<std::string>(), "List of field requirements");
  validPL->sublist("Initial Condition", false, "");
  validPL->sublist("ACE Thermal Parameters", false, "");
  validPL->sublist("Initial Condition Dot", false, "");
  validPL->sublist("Initial Condition DotDot", false, "");
  validPL->sublist("Source Functions", false, "");
  validPL->sublist("Absorption", false, "");
  validPL->sublist("Response Functions", false, "");
  validPL->sublist("Parameters", false, "");
  validPL->sublist("Distributed Parameters", false, "");
  validPL->sublist("Teko", false, "");
  validPL->sublist("XFEM", false, "");
  validPL->sublist("Dirichlet BCs", false, "");
  validPL->sublist("Neumann BCs", false, "");
  validPL->sublist("Adaptation", false, "");
  validPL->sublist("Catalyst", false, "");
  validPL->set<bool>("Solve Adjoint", false, "");
  validPL->set<bool>(
      "Overwrite Nominal Values With Final Point",
      false,
      "Whether 'reportFinalPoint' should be allowed to overwrite nominal "
      "values");
  validPL->set<int>("Number Of Time Derivatives", 1, "Number of time derivatives in use in the problem");

  validPL->set<bool>("Use MDField Memoization", false, "Use memoization to avoid recomputing MDFields");
  validPL->set<bool>("Use MDField Memoization For Parameters", false, "Use memoization to avoid recomputing MDFields dependent on parameters");
  validPL->set<bool>(
      "Ignore Residual In Jacobian",
      false,
      "Ignore residual calculations while computing the Jacobian (only "
      "generally appropriate for linear problems)");
  validPL->set<double>(
      "Perturb Dirichlet",
      0.0,
      "Add this (small) perturbation to the diagonal to prevent Mass Matrices "
      "from being singular for Dirichlets)");

  validPL->sublist("Model Order Reduction", false, "Specify the options relative to model order reduction");

  // Contact PL
  validPL->sublist("Contact", false, "");

  // Candidates for deprecation. Pertain to the solution rather than the problem
  // definition.
  validPL->set<std::string>("Solution Method", "Steady", "Flag for Steady, Transient, or Continuation");
  validPL->set<double>("Homotopy Restart Step", 1., "Flag for LandIce Homotopy Restart Step");
  validPL->set<std::string>("Second Order", "No", "Flag to indicate that a transient problem has two time derivs");
  validPL->set<bool>("Print Response Expansion", true, "");

  // Deprecated parameters, kept solely for backward compatibility
  validPL->set<bool>(
      "Compute Sensitivities",
      true,
      "Deprecated; Use parameter located under \"Piro\"/\"Analysis\"/\"Solve\" "
      "instead.");

  // NOX status test that allows constutive models to cut the global time step
  // needed at the Problem scope when running Schwarz coupling
  validPL->set<Teuchos::RCP<NOX::StatusTest::Generic>>(
      "Constitutive Model NOX Status Test",
      Teuchos::RCP<NOX::StatusTest::Generic>(),
      "NOX status test that facilitates communication between a ModelEvaluator "
      "and a NOX solver");

  validPL->set<bool>("ACE Sequential Thermomechanical", false, "ACE Sequential Thermomechanical Problem");
  validPL->set<double>("ACE Thermomechanical Problem Current Time", 0.0, "Current Time in ACE Sequential Thermomechanical Problem");

  return validPL;
}
