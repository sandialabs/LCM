// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include "MechanicsProblem.hpp"

#include <algorithm>

#include "MechanicsProblem_Def.hpp"

namespace Albany {

MechanicsProblem::MechanicsProblem(
    Teuchos::RCP<Teuchos::ParameterList> const& params,
    Teuchos::RCP<ParamLib> const&               param_lib,
    int const                                   num_dims,
    Teuchos::RCP<AAdapt::rc::Manager> const&    rc_mgr,
    Teuchos::RCP<Teuchos::Comm<int> const>&     commT)
    : AbstractProblem(params, param_lib),
      have_source_(false),
      use_sdbcs_(false),
      thermal_source_(SOURCE_TYPE_NONE),
      thermal_source_evaluated_(false),
      num_dims_(num_dims),
      params_(params),
      // Note: have_mech_eq_ defaults to true in the header, but we override
      // here. getVariableType() below sets it to the value implied by the
      // "Displacement" sublist.
      have_mech_eq_(false),
      rc_mgr_(rc_mgr)
{
  std::string& method = params->get("Name", "Mechanics ");
  *out << "Problem Name = " << method << '\n';

  std::string& sol_method = params->get("Solution Method", "Steady");
  *out << "Solution Method = " << sol_method << '\n';

  dynamic_tempus_ = (sol_method == "Transient Tempus");

  have_source_ = params->isSublist("Source Functions");

  getVariableType(params->sublist("Displacement"), "DOF", mech_type_, have_mech_, have_mech_eq_);
  getVariableType(params->sublist("Temperature"), "None", temperature_type_, have_temperature_, have_temperature_eq_);
  getVariableType(params->sublist("ACE Temperature"), "None", temperature_type_, have_ace_temperature_, have_ace_temperature_eq_);
  getVariableType(params->sublist("Pore Pressure"), "None", pore_pressure_type_, have_pore_pressure_, have_pore_pressure_eq_);
  getVariableType(params->sublist("Transport"), "None", transport_type_, have_transport_, have_transport_eq_);
  getVariableType(params->sublist("HydroStress"), "None", hydrostress_type_, have_hydrostress_, have_hydrostress_eq_);
  getVariableType(params->sublist("Damage"), "None", damage_type_, have_damage_, have_damage_eq_);
  getVariableType(params->sublist("Stabilized Pressure"), "None", stab_pressure_type_, have_stab_pressure_, have_stab_pressure_eq_);

  ALBANY_ASSERT(!(have_temperature_ && have_ace_temperature_), "Cannot have two temperatures");
  ALBANY_ASSERT(!(have_temperature_eq_ && have_ace_temperature_eq_), "Cannot have two temperature equations");

  bool const is_ace_problem = have_ace_temperature_ || have_ace_temperature_eq_;
  if (is_ace_problem) {
    ALBANY_ASSERT(have_ace_temperature_ && have_ace_temperature_eq_, "Cannot have ACE temperature without its equation");
  }

  is_ace_sequential_thermomechanical_ = params->isParameter("ACE Sequential Thermomechanical");

  int num_eq{0};
  if (have_mech_eq_) num_eq += num_dims_;
  if (have_temperature_eq_) num_eq++;
  if (have_ace_temperature_eq_) num_eq++;
  if (have_pore_pressure_eq_) num_eq++;
  if (have_transport_eq_) num_eq++;
  if (have_hydrostress_eq_) num_eq++;
  if (have_damage_eq_) num_eq++;
  if (have_stab_pressure_eq_) num_eq++;
  this->setNumEquations(num_eq);

  *out << "Mechanics problem:" << '\n'
       << "\tSpatial dimension             : " << num_dims_ << '\n'
       << "\tMechanics variables           : " << variableTypeToString(mech_type_) << '\n'
       << "\tTemperature variables         : " << variableTypeToString(temperature_type_) << '\n'
       << "\tPore Pressure variables       : " << variableTypeToString(pore_pressure_type_) << '\n'
       << "\tTransport variables           : " << variableTypeToString(transport_type_) << '\n'
       << "\tHydroStress variables         : " << variableTypeToString(hydrostress_type_) << '\n'
       << "\tDamage variables              : " << variableTypeToString(damage_type_) << '\n'
       << "\tStabilized Pressure variables : " << variableTypeToString(stab_pressure_type_) << '\n';

  material_db_ = createMaterialDatabase(params, commT);

  // Determine the thermal source: the "Source Functions" list must be present
  // in the input file, and we must have temperature and a temperature equation.
  if (have_source_ && have_temperature_eq_) {
    if (params->sublist("Source Functions").isSublist("Thermal Source")) {
      Teuchos::ParameterList& thSrcPL = params->sublist("Source Functions").sublist("Thermal Source");
      if (thSrcPL.get<std::string>("Thermal Source Type", "None") == "Block Dependent") {
        if (Teuchos::nonnull(material_db_)) {
          thermal_source_ = SOURCE_TYPE_MATERIAL;
        }
      } else {
        thermal_source_ = SOURCE_TYPE_INPUT;
      }
    }
  }
  // No temperature sources for ACE heat equation.

  // Rigid body mode (RBM) setup for elasticity problems, consumed by
  // Albany_SolverFactory.cpp.
  int const num_PDEs    = neq;
  int const num_eq_mech = have_mech_eq_ ? num_dims_ : 0;
  int const num_eq_aux  = neq - num_eq_mech;

  int null_space_dim{0};
  if (have_mech_eq_) {
    switch (num_dims_) {
      case 1: null_space_dim = 1; break;
      case 2: null_space_dim = 3; break;
      case 3: null_space_dim = 6; break;
      default: ALBANY_ABORT('\n' << "Error: " << __FILE__ << " line " << __LINE__ << ": num_dims_ set incorrectly." << '\n'); break;
    }
  }
  rigidBodyModes->setParameters(num_PDEs, num_eq_mech, num_eq_aux, null_space_dim);

  have_adaptation_ = params->isSublist("Adaptation");
  bool have_erosion{false};
  if (have_adaptation_) {
    Teuchos::ParameterList const& adapt_params         = params->sublist("Adaptation");
    std::string const&            adaptation_method    = adapt_params.get<std::string>("Method");
    have_sizefield_adaptation_                         = (adaptation_method == "RPI Albany Size");
    have_topmod_adaptation_                            = (adaptation_method == "Topmod");
    have_erosion                                       = (adaptation_method == "Erosion");
  }

  // User-defined NOX status test that can be passed to the ModelEvaluators.
  // This allows a ModelEvaluator to indicate to NOX that something has failed,
  // which is useful for adaptive step size reduction.
  if (params->isParameter("Constitutive Model NOX Status Test")) {
    nox_status_test_ = params->get<Teuchos::RCP<NOX::StatusTest::Generic>>("Constitutive Model NOX Status Test");
  } else {
    nox_status_test_ = Teuchos::rcp(new NOX::StatusTest::ModelEvaluatorFlag);
  }

  bool require_lattice_orientation_on_mesh = false;
  if (Teuchos::nonnull(material_db_)) {
    auto const read_orientation = material_db_->getAllMatchingParams<bool>("Read Lattice Orientation From Mesh");
    require_lattice_orientation_on_mesh = std::any_of(read_orientation.begin(), read_orientation.end(), [](bool b) { return b; });
  }
  if (require_lattice_orientation_on_mesh) {
    requirements.push_back("Lattice_Orientation");
  }
  if (have_erosion && num_dims_ == 3) {
    requirements.push_back("cell_boundary_indicator");
    // TODO: Layout for edge does not exist yet
    requirements.push_back("node_boundary_indicator");
  }
}

void
MechanicsProblem::buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct>> meshSpecs, StateManager& stateMgr)
{
  int const physSets = meshSpecs.size();
  *out << "Num MeshSpecs: " << physSets << '\n';
  fm.resize(physSets);

  bool haveSidesets{false};

  *out << "Calling MechanicsProblem::buildEvaluators" << '\n';
  for (int ps = 0; ps < physSets; ++ps) {
    fm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM, Teuchos::null);
    if (meshSpecs[ps]->ssNames.size() > 0) {
      haveSidesets = true;
    }
  }

  *out << "Calling MechanicsProblem::constructDirichletEvaluators" << '\n';
  constructDirichletEvaluators(*meshSpecs[0]);

  // Check if have Neumann sublist; throw error if attempting to specify Neumann
  // BCs but there are no sidesets in the input mesh.
  bool const isNeumannPL = params_->isSublist("Neumann BCs");
  if (isNeumannPL && !haveSidesets) {
    ALBANY_ABORT("You are attempting to set Neumann BCs on a mesh with no sidesets!");
  }

  if (haveSidesets) {
    *out << "Calling MechanicsProblem::constructNeumannEvaluators" << '\n';
    constructNeumannEvaluators(meshSpecs[0]);
  }
}

void
MechanicsProblem::getAllocatedStates(
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> old_state,
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> new_state) const
{
  old_state = old_state_;
  new_state = new_state_;
}

void
MechanicsProblem::applyProblemSpecificSolverSettings(Teuchos::RCP<Teuchos::ParameterList> params)
{
  // Require Piro->NOX->{Solver Options, Status Tests} to be present.
  if (!params->isSublist("Piro")) return;
  auto& piro = params->sublist("Piro");
  if (!piro.isSublist("NOX")) return;
  auto& nox = piro.sublist("NOX");
  if (!nox.isSublist("Solver Options") || !nox.isSublist("Status Tests")) return;

  Teuchos::ParameterList& solver_opts_params  = nox.sublist("Solver Options");
  Teuchos::ParameterList& status_tests_params = nox.sublist("Status Tests");

  // Add the model evaluator flag as a status test.
  Teuchos::ParameterList old_params = status_tests_params;
  Teuchos::ParameterList new_params;
  new_params.set<std::string>("Test Type", "Combo");
  new_params.set<std::string>("Combo Type", "OR");
  new_params.set<int>("Number of Tests", 2);
  new_params.sublist("Test 0").set("Test Type", "User Defined");
  new_params.sublist("Test 0").set("User Status Test", nox_status_test_);
  new_params.sublist("Test 1") = old_params;
  status_tests_params          = new_params;

  // Create a NOX observer that will reset the status flag at the beginning of
  // a nonlinear solve if one does not exist already.
  std::string const ppo_str{"User Defined Pre/Post Operator"};

  Teuchos::RCP<NOX::Abstract::PrePostOperator> ppo;
  if (solver_opts_params.isParameter(ppo_str)) {
    ppo = solver_opts_params.get<decltype(ppo)>(ppo_str);
  } else {
    ppo = Teuchos::rcp(new LCM::SolutionSniffer);
    solver_opts_params.set(ppo_str, ppo);
    ALBANY_ASSERT(solver_opts_params.isParameter(ppo_str));
  }

  bool constexpr throw_on_fail{true};
  auto status_test_op = Teuchos::rcp_dynamic_cast<LCM::SolutionSniffer>(ppo, throw_on_fail);
  auto status_test    = Teuchos::rcp_dynamic_cast<NOX::StatusTest::ModelEvaluatorFlag>(nox_status_test_);
  status_test_op->setStatusTest(status_test);
}

void
MechanicsProblem::constructDirichletEvaluators(MeshSpecsStruct const& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names.
  std::vector<std::string> dirichletNames(neq);
  int                      index{0};

  if (have_mech_eq_) {
    dirichletNames[index++] = "X";
    if (num_dims_ > 1) dirichletNames[index++] = "Y";
    if (num_dims_ > 2) dirichletNames[index++] = "Z";
  }
  if (have_temperature_eq_) dirichletNames[index++] = "T";
  if (have_ace_temperature_eq_) dirichletNames[index++] = "T";
  if (have_pore_pressure_eq_) dirichletNames[index++] = "P";
  if (have_transport_eq_) dirichletNames[index++] = "C";
  if (have_hydrostress_eq_) dirichletNames[index++] = "TAU";
  if (have_damage_eq_) dirichletNames[index++] = "D";
  if (have_stab_pressure_eq_) dirichletNames[index++] = "SP";

  // Pass the Application through; it is needed for the coupled Schwarz BC and
  // ignored otherwise.
  this->params->set<Teuchos::RCP<Application>>("Application", getApplication());

  BCUtils<DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames, this->params, this->paramLib);

  use_sdbcs_  = dirUtils.useSDBCs();
  offsets_    = dirUtils.getOffsets();
  nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

void
MechanicsProblem::constructNeumannEvaluators(Teuchos::RCP<MeshSpecsStruct> const& meshSpecs)
{
  // Note: we only enter this function if sidesets are defined in the mesh file
  // (meshSpecs.ssNames.size() > 0).

  BCUtils<NeumannTraits> neuUtils;
  if (!neuUtils.haveBCSpecified(this->params)) {
    return;
  }

  // Construct BC evaluators for all side sets and names. The string-index
  // ordering sets up the equation offset, so ordering is important.
  std::vector<std::string> neumannNames(neq + 1);

  // Last entry specifies behavior for setting NBC on "DOF all". By "all", we
  // mean components of the traction vector only; other fields cannot use this
  // specifier.
  neumannNames[neq] = (have_temperature_eq_ || have_ace_temperature_eq_) ? "all-disp-dofs" : "all";

  Teuchos::Array<Teuchos::Array<int>> offsets(neq + 1);
  int                                 index{0};

  if (have_mech_eq_) {
    // num_dims_ components of the traction vector
    offsets[neq].resize(num_dims_);
    for (int i{0}; i < num_dims_; ++i) {
      offsets[neq][i] = i;
    }

    char const components[] = "xyz";
    while (index < num_dims_) {
      neumannNames[index] = "sig_" + std::string(1, components[index]);
      offsets[index]      = Teuchos::Array<int>(1, index);
      index++;
    }
  }

  if (have_temperature_eq_ || have_ace_temperature_eq_) {
    neumannNames[index]      = "T";
    auto const num_eq_mech   = have_mech_eq_ ? num_dims_ : 0;
    offsets[index]           = Teuchos::Array<int>(1, num_eq_mech);
    index++;
  }

  // Flux-vector components (dudx, dudy, dudz) or dudn — not both.
  Teuchos::ArrayRCP<std::string> dof_names(1, "Displacement");
  std::vector<std::string>       condNames(6);  // dudx, dudy, dudz, dudn, P, closed_form, wave_pressure

  // Sidesets are supported only in 2D and 3D.
  if (num_dims_ == 2) {
    condNames[0] = "(t_x, t_y)";
  } else if (num_dims_ == 3) {
    condNames[0] = "(t_x, t_y, t_z)";
  } else {
    ALBANY_ABORT('\n' << "Error: Sidesets only supported in 2 and 3D." << '\n');
  }
  condNames[1] = "dudn";
  condNames[2] = "P";
  condNames[3] = "closed_form";
  condNames[4] = "wave_pressure";
  condNames[5] = "wave_pressure_hydrostatic";

  // FIXME: The resize below assumes a single element block.
  nfm.resize(1);

  nfm[0] = neuUtils.constructBCEvaluators(
      meshSpecs,
      neumannNames,
      dof_names,
      true,  // isVectorField
      0,     // offsetToFirstDOF
      condNames,
      offsets,
      dl_,
      this->params,
      this->paramLib);
}

void
MechanicsProblem::getVariableType(
    Teuchos::ParameterList&          param_list,
    std::string const&               default_type,
    MechanicsProblem::MECH_VAR_TYPE& variable_type,
    bool&                            have_variable,
    bool&                            have_equation)
{
  std::string const type = param_list.get("Variable Type", default_type);

  if (type == "None") {
    variable_type = MECH_VAR_TYPE_NONE;
  } else if (type == "Constant") {
    variable_type = MECH_VAR_TYPE_CONSTANT;
  } else if (type == "DOF") {
    variable_type = MECH_VAR_TYPE_DOF;
  } else if (type == "Time Dependent") {
    variable_type = MECH_VAR_TYPE_TIMEDEP;
  } else {
    ALBANY_ABORT("Unknown variable type " << type << '\n');
  }

  have_variable = (variable_type != MECH_VAR_TYPE_NONE);
  have_equation = (variable_type == MECH_VAR_TYPE_DOF);
}

std::string
MechanicsProblem::variableTypeToString(MechanicsProblem::MECH_VAR_TYPE variable_type)
{
  switch (variable_type) {
    case MECH_VAR_TYPE_NONE: return "None";
    case MECH_VAR_TYPE_CONSTANT: return "Constant";
    case MECH_VAR_TYPE_TIMEDEP: return "Time Dependent";
    case MECH_VAR_TYPE_DOF: return "DOF";
  }
  return "DOF";
}

}  // namespace Albany
