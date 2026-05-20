// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

// NOTE: NBC (Neumann BC) propagation after element deactivation is NOT yet
// implemented. When elements are eroded/deactivated, newly exposed surfaces
// do not automatically receive the NBCs that were on the original boundary.
// This requires future work to detect new boundary faces after element death
// and apply the appropriate Neumann conditions to them.

#include "ACE_ThermoMechanical.hpp"

#include <algorithm>
#include <fstream>
#include <set>

#include "ACEcommon.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_IossSTKMeshStruct.hpp"
#include "Albany_PiroObserver.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_Utils.hpp"
#include "MiniTensor.h"
#include "Piro_LOCASolver.hpp"
#include "Piro_ObserverToLOCASaveDataStrategyAdapter.hpp"
#include "Piro_TempusSolver.hpp"
#include "Piro_TrapezoidRuleSolver.hpp"

namespace {

// In "Mechanics 3D", extract "Mechanics".
inline std::string
getName(std::string const& method)
{
  if (method.size() < 3) return method;
  return method.substr(0, method.size() - 3);
}

void
validate_intervals(std::vector<ST> const& initial_times, std::vector<ST> const& final_times, std::vector<ST> const& time_steps)
{
  auto const initials_size = initial_times.size();
  auto const finals_size   = final_times.size();
  auto const steps_size    = time_steps.size();
  auto const all_equal     = initials_size == finals_size == steps_size;
  ALBANY_ASSERT(
      all_equal == true,
      "Event interval arrays have different sizes, " << "Initial Times : " << initials_size << ", Final Times : " << finals_size << ", Steps Size"
                                                     << steps_size);
  auto prev_ti = initial_times[0];
  auto prev_tf = final_times[0];
  for (auto i = 0; i < steps_size; ++i) {
    auto const ti = initial_times[i];
    auto const tf = final_times[i];
    auto const dt = time_steps[i];
    if (i > 0) {
      ALBANY_ASSERT(
          ti > prev_ti,
          "Initial Times must be monotonically increasing. " << "At index " << i << " found initial time " << ti << " less or equal than previous " << prev_ti);
      ALBANY_ASSERT(
          tf > prev_tf,
          "Final Times must be monotonically increasing. " << "At index " << i << " found final time " << tf << " less or equal than previous " << prev_tf);
      ALBANY_ASSERT(ti > prev_tf, "Intervals must not overlap. At index " << i << " found initial time less or equal than previous final time " << prev_tf);
    }
    ALBANY_ASSERT(tf > ti, "At event interval index " << i << " found initial time " << ti << " greater or equal than final time " << tf);
    ALBANY_ASSERT(dt > 0.0, "At event interval index " << i << " found non-positive time step " << dt);
    ALBANY_ASSERT(dt <= tf - ti, "At event interval index " << i << " found time step " << dt << "greater than interval " << tf - ti);
    prev_ti = ti;
    prev_tf = tf;
  }
}

int
find_time_interval_index(std::vector<ST> const& initial_times, std::vector<ST> const& final_times, ST time)
{
  for (size_t i = 0; i < initial_times.size(); ++i) {
    if (time >= initial_times[i] && time <= final_times[i]) {
      return i;  // Time value is within this interval
    }
  }
  return -1;  // Time value is not within any interval
}

int
find_next_interval_index(std::vector<ST> const& initial_times, std::vector<ST> const& final_times, ST time)
{
  for (size_t i = 0; i < initial_times.size(); ++i) {
    if (time < initial_times[i]) {
      return i;
    }
  }
  return initial_times.size();
}

bool
is_within_interval(std::vector<ST> const& initial_times, std::vector<ST> const& final_times, ST time, int interval_index)
{
  if (interval_index == -1) return false;
  return initial_times[interval_index] <= time && time <= final_times[interval_index];
}

}  // anonymous namespace

namespace LCM {

ACEThermoMechanical::ACEThermoMechanical(Teuchos::RCP<Teuchos::ParameterList> const& app_params, Teuchos::RCP<Teuchos::Comm<int> const> const& comm)
    : fos_(Teuchos::VerboseObjectBase::getDefaultOStream()), comm_(comm)
{
  alt_system_params_ = Teuchos::sublist(app_params, "Alternating System");
  // Get names of individual model input files
  model_filenames_ = alt_system_params_->get<Teuchos::Array<std::string>>("Model Input Files");

  maximum_steps_     = alt_system_params_->get<int>("Maximum Steps", 0);
  initial_time_      = alt_system_params_->get<ST>("Initial Time", 0.0);
  final_time_        = alt_system_params_->get<ST>("Final Time", 0.0);
  initial_time_step_ = alt_system_params_->get<ST>("Initial Time Step", 1.0);

  auto const dt  = initial_time_step_;
  auto const dt2 = dt * dt;

  min_time_step_    = alt_system_params_->get<ST>("Minimum Time Step", dt);
  max_time_step_    = alt_system_params_->get<ST>("Maximum Time Step", dt);
  reduction_factor_ = alt_system_params_->get<ST>("Reduction Factor", 1.0);
  increase_factor_  = alt_system_params_->get<ST>("Amplification Factor", 1.0);
  output_interval_  = alt_system_params_->get<int>("Exodus Write Interval", 1);
  std_init_guess_   = alt_system_params_->get<bool>("Standard Initial Guess", false);

  // Check for existence of time intervals for events, and if so, read them.
  auto const have_event_initial_times = alt_system_params_->isParameter("Event Initial Times File");
  auto const have_event_final_times   = alt_system_params_->isParameter("Event Final Times File");
  auto const have_event_time_steps    = alt_system_params_->isParameter("Event Time Steps File");
  auto const have_all                 = have_event_initial_times && have_event_final_times && have_event_time_steps;
  auto const have_at_least_one        = have_event_initial_times || have_event_final_times || have_event_time_steps;
  ALBANY_ASSERT(have_all == have_at_least_one, "Initial Times, Final Times and Time Steps for events either must all exist or be absent.");
  if (have_all == true) {
    std::string ti_filename = alt_system_params_->get<std::string>("Event Initial Times File");
    std::string tf_filename = alt_system_params_->get<std::string>("Event Final Times File");
    std::string dt_filename = alt_system_params_->get<std::string>("Event Time Steps File");
    event_initial_times_    = LCM::vectorFromFile(ti_filename);
    event_final_times_      = LCM::vectorFromFile(ti_filename);
    event_time_steps_       = LCM::vectorFromFile(ti_filename);
    validate_intervals(event_initial_times_, event_final_times_, event_time_steps_);
  }

  // Firewalls
  ALBANY_ASSERT(maximum_steps_ >= 1, "");
  ALBANY_ASSERT(final_time_ >= initial_time_, "");
  ALBANY_ASSERT(initial_time_step_ > 0.0, "");
  ALBANY_ASSERT(max_time_step_ > 0.0, "");
  ALBANY_ASSERT(min_time_step_ > 0.0, "");
  ALBANY_ASSERT(max_time_step_ >= min_time_step_, "");
  ALBANY_ASSERT(reduction_factor_ <= 1.0, "");
  ALBANY_ASSERT(reduction_factor_ > 0.0, "");
  ALBANY_ASSERT(increase_factor_ >= 1.0, "");
  ALBANY_ASSERT(output_interval_ >= 1, "");

  // number of models
  num_subdomains_ = model_filenames_.size();

  // throw error if number of model filenames provided is > 2
  ALBANY_ASSERT(num_subdomains_ <= 2, "ACEThermoMechanical solver requires no more than 2 models!");

  // Arrays to cache useful info for each subdomain for later use
  apps_.resize(num_subdomains_);
  solvers_.resize(num_subdomains_);
  solver_factories_.resize(num_subdomains_);
  stk_mesh_structs_.resize(num_subdomains_);
  discs_.resize(num_subdomains_);
  model_evaluators_.resize(num_subdomains_);
  sub_inargs_.resize(num_subdomains_);
  sub_outargs_.resize(num_subdomains_);
  curr_x_.resize(num_subdomains_);
  prev_step_x_.resize(num_subdomains_);
  internal_states_.resize(num_subdomains_);

  // IKT NOTE 6/4/2020:the xdotdot arrays are
  // not relevant for thermal problems,
  // but they are constructed anyway.
  ics_x_.resize(num_subdomains_);
  ics_xdot_.resize(num_subdomains_);
  ics_xdotdot_.resize(num_subdomains_);
  prev_x_.resize(num_subdomains_);
  prev_xdot_.resize(num_subdomains_);
  prev_xdotdot_.resize(num_subdomains_);
  this_x_.resize(num_subdomains_);
  this_xdot_.resize(num_subdomains_);
  this_xdotdot_.resize(num_subdomains_);
  do_outputs_.resize(num_subdomains_);
  do_outputs_init_.resize(num_subdomains_);
  prob_types_.resize(num_subdomains_);
  init_pls_.resize(num_subdomains_);

  // Create solver factories once at the beginning
  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    solver_factories_[subdomain]   = Teuchos::rcp(new Albany::SolverFactory(model_filenames_[subdomain], comm_));
    Teuchos::ParameterList& params = solver_factories_[subdomain]->getParameters();

    Teuchos::ParameterList& problem_params = params.sublist("Problem", true);
    problem_params.set<bool>("ACE Sequential Thermomechanical", true);

    std::string const problem_name = getName(problem_params.get<std::string>("Name"));
    if (problem_name == "Mechanics") {
      prob_types_[subdomain] = MECHANICAL;
    } else if (problem_name == "ACE Thermal") {
      prob_types_[subdomain] = THERMAL;
    } else {
      ALBANY_ABORT("ACE Sequential thermo-mechanical solver only supports coupling of 'Mechanics' and 'ACE Thermal' problems!");
    }

    auto const problem_type = prob_types_[subdomain];

    // Error checks - only needs to be done once at the beginning
    bool const have_piro = params.isSublist("Piro");
    ALBANY_ASSERT(have_piro == true, "Error! Piro sublist not found.\n");

    Teuchos::ParameterList& piro_params = params.sublist("Piro");

    std::string const msg{"All subdomains must have the same solution method (NOX or Tempus)"};

    if (problem_type == THERMAL) {
      auto const is_dynamic = piro_params.isSublist("Tempus");
      ALBANY_ASSERT(is_dynamic == true, "ACE Thermomechanical Coupling requires Tempus for thermal solve.");

      Teuchos::ParameterList& tempus_params = piro_params.sublist("Tempus");

      tempus_params.set("Abort on Failure", false);

      Teuchos::ParameterList& time_step_control_params =
          piro_params.sublist("Tempus").sublist("Tempus Integrator").sublist("Time Step Control").sublist("Time Step Control Strategy");

      std::string const integrator_step_type = time_step_control_params.get("Strategy Type", "Constant");

      std::string const msg{
          "Non-constant time-stepping through Tempus not supported "
          "with ACE sequential thermo-mechanical coupling; \n"
          "In this case, variable time-stepping is "
          "handled within the coupling loop.\n"
          "Please rerun with 'Strategy Type: "
          "Constant' in 'Time Step Control Strategy' sublist.\n"};
      ALBANY_ASSERT(integrator_step_type == "Constant", msg);
    } else if (problem_type == MECHANICAL) {
      auto const is_tempus         = piro_params.isSublist("Tempus");
      auto const is_trapezoid_rule = piro_params.isSublist("Trapezoid Rule");
      if (is_tempus == true) {
        mechanical_solver_                    = MechanicalSolver::Tempus;
        Teuchos::ParameterList& tempus_params = piro_params.sublist("Tempus");
        tempus_params.set("Abort on Failure", false);

        Teuchos::ParameterList& time_step_control_params =
            piro_params.sublist("Tempus").sublist("Tempus Integrator").sublist("Time Step Control").sublist("Time Step Control Strategy");

        std::string const integrator_step_type = time_step_control_params.get("Strategy Type", "Constant");

        std::string const msg{
            "Non-constant time-stepping through Tempus not supported "
            "with ACE sequential thermo-mechanical coupling; \n"
            "In this case, variable time-stepping is "
            "handled within the coupling loop.\n"
            "Please rerun with 'Strategy Type: "
            "Constant' in 'Time Step Control Strategy' sublist.\n"};
        ALBANY_ASSERT(integrator_step_type == "Constant", msg);
      } else {
        mechanical_solver_ = MechanicalSolver::TrapezoidRule;
        ALBANY_ASSERT(is_trapezoid_rule == true, "ACE Thermomechanical Coupling requires Tempus or Trapezoid Rule for mechanical solve.");
        // The coupling loop controls time stepping, so the TrapezoidRule
        // solver should only do 1 internal sub-step per evalModel call.
        // Set Initial/Final Time to span one coupling time step so that
        // delta_t = (Final - Initial) / Num_Steps matches correctly.
        // This matches the I/O solver's behavior.
        Teuchos::ParameterList& tr_params = piro_params.sublist("Trapezoid Rule", true);
        tr_params.set<int>("Num Time Steps", 1);
        tr_params.set<double>("Initial Time", 0.0);
        tr_params.set<double>("Final Time", initial_time_step_);
      }
    } else {
      ALBANY_ABORT("ACE Thermomechanical Coupling only supports coupling of ACE Thermal and Mechanical problems.");
    }

    // Store a copy of the parameter list for each subdomain
    init_pls_[subdomain] = Teuchos::rcp(new Teuchos::ParameterList(params));
  }

  // Parameters
  Teuchos::ParameterList& problem_params  = app_params->sublist("Problem");
  bool const              have_parameters = problem_params.isSublist("Parameters");
  ALBANY_ASSERT(have_parameters == false, "Parameters not supported.");

  // Responses
  bool const have_responses = problem_params.isSublist("Response Functions");
  ALBANY_ASSERT(have_responses == false, "Responses not supported.");

  // Check that the first problem type is thermal and the second problem type is mechanical,
  // as the current version of the coupling algorithm requires this ordering.
  // If ordering is wrong, through error.
  PROB_TYPE prob_type = prob_types_[0];
  ALBANY_ASSERT(prob_type == THERMAL, "The first problem type needs to be 'ACE Thermal'!");
  if (num_subdomains_ > 1) {
    prob_type = prob_types_[1];
    ALBANY_ASSERT(prob_type == MECHANICAL, "The second problem type needs to be 'Mechanics'!");
  }

  // Create persistent apps in memory
  createPersistentApps();

  return;
}

ACEThermoMechanical::~ACEThermoMechanical() { return; }

Teuchos::RCP<Thyra_VectorSpace const>
ACEThermoMechanical::get_x_space() const
{
  return Teuchos::null;
}

Teuchos::RCP<Thyra_VectorSpace const>
ACEThermoMechanical::get_f_space() const
{
  return Teuchos::null;
}

Teuchos::RCP<Thyra_VectorSpace const>
ACEThermoMechanical::get_p_space(int) const
{
  return Teuchos::null;
}

Teuchos::RCP<Thyra_VectorSpace const>
ACEThermoMechanical::get_g_space(int) const
{
  return Teuchos::null;
}

Teuchos::RCP<const Teuchos::Array<std::string>>
ACEThermoMechanical::get_p_names(int) const
{
  return Teuchos::null;
}

Teuchos::ArrayView<std::string const>
ACEThermoMechanical::get_g_names(int) const
{
  ALBANY_ABORT("not implemented");
  return Teuchos::ArrayView<std::string const>(Teuchos::null);
}

Thyra_ModelEvaluator::InArgs<ST>
ACEThermoMechanical::getNominalValues() const
{
  return this->createInArgsImpl();
}

Thyra_ModelEvaluator::InArgs<ST>
ACEThermoMechanical::getLowerBounds() const
{
  return Thyra_ModelEvaluator::InArgs<ST>();  // Default value
}

Thyra_ModelEvaluator::InArgs<ST>
ACEThermoMechanical::getUpperBounds() const
{
  return Thyra_ModelEvaluator::InArgs<ST>();  // Default value
}

Teuchos::RCP<Thyra::LinearOpBase<ST>>
ACEThermoMechanical::create_W_op() const
{
  return Teuchos::null;
}

Teuchos::RCP<Thyra::PreconditionerBase<ST>>
ACEThermoMechanical::create_W_prec() const
{
  return Teuchos::null;
}

Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST>>
ACEThermoMechanical::get_W_factory() const
{
  return Teuchos::null;
}

Thyra_ModelEvaluator::InArgs<ST>
ACEThermoMechanical::createInArgs() const
{
  return this->createInArgsImpl();
}

Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
ACEThermoMechanical::getApps() const
{
  return apps_;
}

void
ACEThermoMechanical::set_failed(char const* msg)
{
  failed_          = true;
  failure_message_ = msg;
  return;
}

void
ACEThermoMechanical::clear_failed()
{
  failed_ = false;
  return;
}

bool
ACEThermoMechanical::get_failed() const
{
  return failed_;
}

// Create operator form of dg/dx for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST>>
ACEThermoMechanical::create_DgDx_op_impl(int /* j */) const
{
  return Teuchos::null;
}

// Create operator form of dg/dx_dot for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST>>
ACEThermoMechanical::create_DgDx_dot_op_impl(int /* j */) const
{
  return Teuchos::null;
}

// Create InArgs
Thyra_InArgs
ACEThermoMechanical::createInArgsImpl() const
{
  Thyra::ModelEvaluatorBase::InArgsSetup<ST> ias;

  ias.setModelEvalDescription(this->description());

  ias.setSupports(Thyra_ModelEvaluator::IN_ARG_x, true);
  ias.setSupports(Thyra_ModelEvaluator::IN_ARG_x_dot, true);
  ias.setSupports(Thyra_ModelEvaluator::IN_ARG_x_dot_dot, true);
  ias.setSupports(Thyra_ModelEvaluator::IN_ARG_t, true);
  ias.setSupports(Thyra_ModelEvaluator::IN_ARG_alpha, true);
  ias.setSupports(Thyra_ModelEvaluator::IN_ARG_beta, true);
  ias.setSupports(Thyra_ModelEvaluator::IN_ARG_W_x_dot_dot_coeff, true);

  return static_cast<Thyra_InArgs>(ias);
}

// Create OutArgs
Thyra_OutArgs
ACEThermoMechanical::createOutArgsImpl() const
{
  Thyra::ModelEvaluatorBase::OutArgsSetup<ST> oas;

  oas.setModelEvalDescription(this->description());

  oas.setSupports(Thyra_ModelEvaluator::OUT_ARG_f, true);
  oas.setSupports(Thyra_ModelEvaluator::OUT_ARG_W_op, true);
  oas.setSupports(Thyra_ModelEvaluator::OUT_ARG_W_prec, false);

  oas.set_W_properties(Thyra_ModelEvaluator::DerivativeProperties(Thyra_ModelEvaluator::DERIV_LINEARITY_UNKNOWN, Thyra_ModelEvaluator::DERIV_RANK_FULL, true));

  return static_cast<Thyra_OutArgs>(oas);
}

// Evaluate model on InArgs
void
ACEThermoMechanical::evalModelImpl(Thyra_ModelEvaluator::InArgs<ST> const&, Thyra_ModelEvaluator::OutArgs<ST> const&) const
{
  ThermoMechanicalLoopDynamics();
}

namespace {

std::string
centered(std::string const& str, int width)
{
  assert(width >= 0);
  int const length  = static_cast<int>(str.size());
  int const padding = width - length;
  if (padding <= 0) return str;
  int const left  = padding / 2;
  int const right = padding - left;
  return std::string(left, ' ') + str + std::string(right, ' ');
}

}  // anonymous namespace

void
ACEThermoMechanical::createPersistentApps()
{
  Teuchos::TimeMonitor timer(*Teuchos::TimeMonitor::getNewTimer("ACE: Create Persistent Apps"));

  // ----- Per-subdomain param setup (Exodus field names, workset size) -----
  // Done up front so all subdomains' discretization params are settled
  // before we build the shared mesh.
  for (int subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    auto& app_params  = *init_pls_[subdomain];
    auto& disc_params = app_params.sublist("Discretization");

    if (prob_types_[subdomain] == MECHANICAL) {
      disc_params.set<std::string>("Exodus Solution Name", "displacement");
      disc_params.set<std::string>("Exodus SolutionDot Name", "velocity");
      disc_params.set<std::string>("Exodus SolutionDotDot Name", "acceleration");
      // Distinct residual name: with shared mesh, both apps declare a
      // residual field on the same metaData; defaulting both to "residual"
      // produces a dim-mismatch conflict (mech: 3, thermal: 1).
      disc_params.set<std::string>("Exodus Residual Name", "displacement_residual");

      // M4: single coupled output file. The mechanical subdomain solves
      // last in the coupling loop, so its Exodus file captures the fully
      // updated coupled state on the shared mesh; designate it the sole
      // writer. Strip the thermal_/mechanical_ prefix from its filename
      // so the coupled output has a physics-neutral name. This only
      // mutates ACE's in-memory copy of the params (init_pls_); the YAML
      // file is untouched, so a standalone mechanical run still produces
      // mechanical_denudation.e.
      if (disc_params.isType<std::string>("Exodus Output File Name")) {
        std::string fname = disc_params.get<std::string>("Exodus Output File Name");
        for (std::string const& pfx : {std::string("mechanical_"), std::string("thermal_")}) {
          auto const pos = fname.find(pfx);
          if (pos != std::string::npos) {
            fname.erase(pos, pfx.size());
            break;
          }
        }
        disc_params.set<std::string>("Exodus Output File Name", fname);
      }
    }
    if (prob_types_[subdomain] == THERMAL) {
      disc_params.set<std::string>("Exodus Solution Name", "temperature");
      disc_params.set<std::string>("Exodus SolutionDot Name", "temperature_dot");
      disc_params.set<std::string>("Exodus Residual Name", "temperature_residual");

      // M4: non-writer subdomain. Drop its output file param so no
      // output broker is created (setupExodusOutput is gated on
      // exoOutput, which derives from this param). Again only ACE's
      // in-memory copy is changed; standalone thermal runs are
      // unaffected.
      disc_params.remove("Exodus Output File Name", /*throwIfNotExists=*/false);
    }

    // Force matching workset sizes so state arrays line up cell-for-cell
    // between subdomains. -1 = one workset per element block.
    disc_params.set("Workset Size", -1);
  }

  // ----- Build the shared STK mesh once, deferring commit -----
  // ACE thermo-mechanical always uses the same mesh for both physics,
  // so we open the Exodus file once via the first subdomain's params,
  // then both Applications declare their fields on the shared metaData
  // before a single commit fires.
  auto first_disc_params    = Teuchos::sublist(init_pls_[0], "Discretization", true);
  auto first_problem_params = Teuchos::sublist(init_pls_[0], "Problem", true);
  Teuchos::RCP<Teuchos::ParameterList> first_adapt_params;
  if (first_problem_params->isSublist("Adaptation")) {
    first_adapt_params = Teuchos::sublist(first_problem_params, "Adaptation", true);
  }

  // In the normal single-app path, Application::initialSetUp seeds
  // "Number Of Time Derivatives" on the Discretization sublist by
  // reading from the Problem sublist (which AbstractProblem's ctor
  // populates). We're constructing the mesh BEFORE any Application
  // exists, so AbstractProblem hasn't run and neither sublist has the
  // value yet. Derive it directly from prob_types_: MECHANICAL needs 2
  // (displacement/velocity/acceleration), THERMAL needs 1. Use the max
  // across subdomains so the shared mesh has enough headroom for side-set
  // inheritance.
  int shared_num_time_deriv = 0;
  for (int s = 0; s < num_subdomains_; ++s) {
    int const ntd = (prob_types_[s] == MECHANICAL) ? 2 : 1;
    if (ntd > shared_num_time_deriv) shared_num_time_deriv = ntd;
  }
  first_disc_params->set<int>("Number Of Time Derivatives", shared_num_time_deriv);

  auto shared_mesh_abs = Albany::DiscretizationFactory::createMeshStruct(
      first_disc_params, first_adapt_params, comm_);
  auto shared_mesh = Teuchos::rcp_dynamic_cast<Albany::IossSTKMeshStruct>(shared_mesh_abs);
  ALBANY_PANIC(
      shared_mesh.is_null(),
      "ACE_ThermoMechanical shared-mesh path requires IossSTKMeshStruct "
      "(Discretization Method: Ioss/Exodus/Pamgen).\n");
  shared_mesh->deferCommit = true;

  // ----- Construct per-subdomain mesh struct: donor for 0, borrowing for the rest -----
  std::vector<Teuchos::RCP<Albany::AbstractMeshStruct>> per_app_mesh(num_subdomains_);
  per_app_mesh[0] = shared_mesh;
  for (int s = 1; s < num_subdomains_; ++s) {
    auto s_disc_params    = Teuchos::sublist(init_pls_[s], "Discretization", true);
    auto s_problem_params = Teuchos::sublist(init_pls_[s], "Problem", true);
    Teuchos::RCP<Teuchos::ParameterList> s_adapt_params;
    if (s_problem_params->isSublist("Adaptation")) {
      s_adapt_params = Teuchos::sublist(s_problem_params, "Adaptation", true);
    }
    per_app_mesh[s] = Teuchos::rcp(new Albany::IossSTKMeshStruct(
        shared_mesh, s_disc_params, s_adapt_params, comm_));
  }

  // ----- Construct each Application with its mesh, deferring post-commit -----
  // Each app's createDiscretization runs setFieldAndBulkData on its mesh
  // struct, which (because deferCommit is set) declares the app's field
  // container on the shared metaData and stops before commit.
  for (int s = 0; s < num_subdomains_; ++s) {
    apps_[s] = Teuchos::rcp(new Albany::Application(
        comm_, init_pls_[s], per_app_mesh[s], /*deferPostCommit=*/true));
  }

  // ----- Single commit + populate on the shared mesh -----
  // After this, both apps' fields are alive on the committed metaData
  // and the bulk data is populated from the Exodus file. We pass the
  // donor's params + the first app's StateInfoStruct; per-app state
  // marking happens through each app's StateManager anyway.
  shared_mesh->commitAndPopulate(
      comm_, first_disc_params, apps_[0]->getStateMgr().getStateInfoStruct());

  // ----- Finalize each app: runs disc->updateMesh() + finalSetUp -----
  for (int s = 0; s < num_subdomains_; ++s) {
    apps_[s]->finalizePostCommit();
  }

  // ----- Build SolverFactory + Piro per app, reusing the existing Application -----
  for (int s = 0; s < num_subdomains_; ++s) {
    solver_factories_[s] = Teuchos::rcp(new Albany::SolverFactory(init_pls_[s], comm_));

    Teuchos::RCP<Albany::Application> app_ref = apps_[s];
    solvers_[s] = solver_factories_[s]->createAndGetAlbanyApp(
        app_ref, comm_, comm_, Teuchos::null, /*createAlbanyApp=*/false);

    auto stk_disc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(apps_[s]->getDiscretization());
    discs_[s]            = stk_disc;
    stk_mesh_structs_[s] = stk_disc->getSTKMeshStruct();

    do_outputs_[s]      = stk_mesh_structs_[s]->exoOutput;
    do_outputs_init_[s] = stk_mesh_structs_[s]->exoOutput;

    model_evaluators_[s] = solver_factories_[s]->returnModel();

    curr_x_[s] = Teuchos::null;

    if (prob_types_[s] == THERMAL) {
      Teuchos::RCP<const Thyra_MultiVector> coord_mv = stk_disc->getCoordMV();
      Teuchos::RCP<const Thyra_Vector>      z_coord  = coord_mv->col(2);
      zmin_                                          = Thyra::min(*z_coord);
    }
  }

  // Cache initial internal states
  for (int subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    auto& state_mgr = apps_[subdomain]->getStateMgr();
    fromTo(state_mgr.getStateArrays(), internal_states_[subdomain]);
  }
}

void
ACEThermoMechanical::transferThermalToMechanical(int thermal_sub, int mech_sub) const
{
  // Restored after the M2 no-op was found to break the coupling.
  //
  // M2 assumed that, because both apps share one STKMeshStruct, they
  // also share element-state fields like ACE_Ice_Saturation, making
  // this copy an identity. That is false: the shared mesh provides
  // shared topology, but each Application has its OWN Discretization
  // with its OWN StateManager element-state arrays
  // (Discretization::stateArrays). Those arrays are Albany scratch,
  // populated by each app's evaluators; they do NOT round-trip through
  // a shared STK field during the solve. So the thermal app's computed
  // ACE_Ice_Saturation must be copied explicitly into the mechanical
  // app's element-state arrays, or the mechanical app reads stale /
  // initial-condition values (ice never thaws, no erosion).
  //
  // With M1's shared mesh, both apps' Discretizations build worksets
  // from the same BulkData, so the workset/cell layout matches and the
  // size assertion below holds.
  auto& thermal_state_mgr = apps_[thermal_sub]->getStateMgr();
  auto& mech_state_mgr    = apps_[mech_sub]->getStateMgr();

  auto& thermal_states = thermal_state_mgr.getStateArrays().elemStateArrays;
  auto& mech_states    = mech_state_mgr.getStateArrays().elemStateArrays;

  std::vector<std::string> shared_fields = {"ACE_Ice_Saturation"};
  for (auto const& field_name : shared_fields) {
    for (size_t ws = 0; ws < thermal_states.size() && ws < mech_states.size(); ++ws) {
      auto it_thermal = thermal_states[ws].find(field_name);
      auto it_mech    = mech_states[ws].find(field_name);
      if (it_thermal != thermal_states[ws].end() && it_mech != mech_states[ws].end()) {
        auto const& src = it_thermal->second;
        auto&       dst = it_mech->second;
        ALBANY_ASSERT(
            src.size() == dst.size(),
            "ACE_Ice_Saturation size mismatch at ws=" << ws << ": thermal=" << src.size() << " mechanical=" << dst.size()
                                                      << ". Both apps must build worksets from the shared mesh.");
        for (size_t i = 0; i < src.size(); ++i) {
          dst[i] = src[i];
        }
      }
    }
  }
}

void
ACEThermoMechanical::transferMechanicalToThermal(int /*mech_sub*/, int /*thermal_sub*/) const
{
  // No element-state copy needed mechanical->thermal. Element death is
  // the only mechanical-side change the thermal app must see, and that
  // propagates automatically: with M1's shared mesh, dead cells leave
  // the shared activePart, so the thermal app's worksets (rebuilt from
  // the shared BulkData) exclude them too. This direction was already
  // a no-op before M2.
}

bool
ACEThermoMechanical::continueSolve() const
{
  ++num_iter_;
  // IKT 6/5/2020: right now, we want to do just 1 coupling iteration.
  // Therefore return false if we've hit num_iter_ = 1;
  // Also set converged_ to true, which is equally irrelevant unless doing
  // Schwarz-like coupling
  converged_ = true;
  if (num_iter_ > 0)
    return false;
  else
    return true;
}

// Sequential ThermoMechanical coupling loop, dynamic
void
ACEThermoMechanical::ThermoMechanicalLoopDynamics() const
{
  std::string const delim(72, '=');

  *fos_ << std::scientific << std::setprecision(17);

  // If initial time is within an interval, reset to its beginning
  ST   time_step{initial_time_step_};
  auto interval_index = find_time_interval_index(event_initial_times_, event_final_times_, initial_time_);
  if (interval_index != -1) {
    initial_time_ = event_initial_times_[interval_index];
    time_step     = event_time_steps_[interval_index];
  }
  int stop{0};
  ST  current_time{initial_time_};

  // Identify thermal and mechanical subdomain indices
  int thermal_sub = -1;
  int mech_sub = -1;
  for (int s = 0; s < num_subdomains_; ++s) {
    if (prob_types_[s] == THERMAL) thermal_sub = s;
    if (prob_types_[s] == MECHANICAL) mech_sub = s;
  }

  // Time-stepping loop
  while (stop < maximum_steps_ && current_time < final_time_) {
    if (interval_index != -1) {
      *fos_ << delim << std::endl;
      *fos_ << "Subclycling within an event interval.\n";
    }

    *fos_ << delim << std::endl;
    *fos_ << "Time stop          :" << stop << '\n';
    *fos_ << "Time               :" << current_time << '\n';
    *fos_ << "Time step          :" << time_step << '\n';
    *fos_ << delim << std::endl;

    ST const next_time{current_time + time_step};
    num_iter_ = 0;

    // Coupling loop
    do {
      bool const is_initial_state = stop == 0 && num_iter_ == 0;
      for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
        auto const prob_type = prob_types_[subdomain];

        // Before the coupling loop, figure out whether
        // output needs to be done or not.
        if (num_iter_ == 0) {
          do_outputs_[subdomain] = true;  // We always want output in the initial step
        } else {
          if (do_outputs_init_[subdomain] == true) {
            do_outputs_[subdomain] = output_interval_ > 0 ? (stop + 1) % output_interval_ == 0 : false;
          }
        }
        *fos_ << delim << std::endl;
        *fos_ << "Subdomain          :" << subdomain << '\n';
        if (prob_type == THERMAL) {
          *fos_ << "Problem            :Thermal\n";
          auto& app       = *apps_[subdomain];
          auto& state_mgr = app.getStateMgr();
          {
            Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE: Restore Thermal States"));
            fromTo(internal_states_[subdomain], state_mgr.getStateArrays());
          }
          {
            Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE: Thermal Solve"));
            AdvanceThermalDynamics(subdomain, is_initial_state, current_time, next_time, time_step);
          }
          if (failed_ == false) {
            {
              Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE: Save Thermal States"));
              fromTo(state_mgr.getStateArrays(), internal_states_[subdomain]);
            }
            {
              Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE: Thermal Output"));
              doDynamicInitialOutput(next_time, subdomain);
            }
          }
        }
        if (prob_type == MECHANICAL && failed_ == false) {
          *fos_ << "Problem            :Mechanical\n";
          auto& app       = *apps_[subdomain];
          auto& state_mgr = app.getStateMgr();
          {
            Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE: Restore Mechanical States"));
            fromTo(internal_states_[subdomain], state_mgr.getStateArrays());
          }
          // Transfer thermal results AFTER restoring mechanical states,
          // so the thermal-to-mechanical transfer is not overwritten.
          if (thermal_sub >= 0) {
            Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE: Transfer Thermal->Mechanical"));
            transferThermalToMechanical(thermal_sub, subdomain);
          }
          // Set death status on the Application for scatter skip and orphan
          // fix. Read the per-cell `cell_death` state variable, which J2Erosion
          // sets to 1.0 once every integration point in the cell has failed
          // (in any mode) and 0.0 otherwise. cell_death is a cell-scalar so
          // there is no qp dimension to skip past.
          {
            auto& app = *apps_[subdomain];
            auto& esa = internal_states_[subdomain].element_state_arrays;
            app.death_status_vecs_.resize(esa.size());
            for (size_t ws = 0; ws < esa.size(); ++ws) {
              auto it = esa[ws].find("cell_death");
              if (it != esa[ws].end()) {
                auto&     flat      = it->second;
                int const num_cells = flat.size();
                auto      ds        = Teuchos::rcp(new std::vector<double>(num_cells, 0.0));
                for (int c = 0; c < num_cells; ++c) {
                  (*ds)[c] = flat[c];
                }
                app.death_status_vecs_[ws] = ds;
              }
            }
          }
          {
            Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE: Mechanical Solve"));
            AdvanceMechanicalDynamics(subdomain, is_initial_state, current_time, next_time, time_step);
          }
          if (failed_ == false) {
            {
              Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE: Save Mechanical States"));
              fromTo(state_mgr.getStateArrays(), internal_states_[subdomain]);
            }
            {
              Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE: Mechanical Output"));
              doDynamicInitialOutput(next_time, subdomain);
            }
          }
        }
        if (failed_ == true) {
          // Break out of the subdomain loop
          break;
        }
      }  // Subdomains loop

      if (failed_ == true) {
        // Break out of the coupling loop.
        break;
      }
    } while (continueSolve() == true);

    // One of the subdomains failed to solve. Reduce step.
    if (failed_ == true) {
      auto const reduced_step = reduction_factor_ * time_step;

      if (time_step <= min_time_step_) {
        *fos_ << "ERROR: Cannot reduce step. Stopping execution.\n";
        *fos_ << "INFO: Requested step    :" << reduced_step << '\n';
        *fos_ << "INFO: Minimum time step :" << min_time_step_ << '\n';
        return;
      }

      if (reduced_step > min_time_step_) {
        *fos_ << "INFO: Reducing step from " << time_step << " to ";
        *fos_ << reduced_step << '\n';
        time_step = reduced_step;
      } else {
        *fos_ << "INFO: Reducing step from " << time_step << " to ";
        *fos_ << min_time_step_ << '\n';
        time_step = min_time_step_;
      }

      // Jump to the beginning of the time-step loop without advancing
      // time to try to use a reduced step.
      continue;
    }

    // Update IC vecs
    for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
      setICVecs(next_time, subdomain);
    }

    if (interval_index != -1) {
      time_step = std::min(time_step, event_final_times_[interval_index] - current_time);
      time_step = std::max(time_step, min_time_step_);
    } else {
      auto const next_interval_index = find_time_interval_index(event_initial_times_, event_final_times_, current_time);
      if (next_interval_index < event_initial_times_.size()) {
        time_step = std::min(time_step, event_initial_times_[next_interval_index] - current_time);
        time_step = std::max(time_step, min_time_step_);
      }
    }

    ++stop;
    current_time += time_step;
    interval_index         = find_time_interval_index(event_initial_times_, event_final_times_, current_time);
    auto const in_interval = is_within_interval(event_initial_times_, event_final_times_, current_time, interval_index);

    // Step successful. Try to increase the time step.
    auto const increased_step = std::min(max_time_step_, increase_factor_ * time_step);

    if (increased_step > time_step && in_interval == false) {
      *fos_ << "\nINFO: Increasing step from " << time_step << " to ";
      *fos_ << increased_step << '\n';
      time_step = increased_step;
    } else {
      *fos_ << "\nINFO: Cannot increase step. Using " << time_step << '\n';
    }

  }  // Time-step loop

  return;
}

void
ACEThermoMechanical::AdvanceThermalDynamics(
    int const    subdomain,
    bool const   is_initial_state,
    double const current_time,
    double const next_time,
    double const time_step) const
{
  failed_ = false;
  // Solve for each subdomain
  Thyra::ResponseOnlyModelEvaluatorBase<ST>& solver = *(solvers_[subdomain]);

  Piro::TempusSolver<ST>& piro_tempus_solver = dynamic_cast<Piro::TempusSolver<ST>&>(solver);

  piro_tempus_solver.setStartTime(current_time);
  piro_tempus_solver.setFinalTime(next_time);
  piro_tempus_solver.setInitTimeStep(time_step);

  std::string const delim(72, '=');
  *fos_ << "Initial time       :" << current_time << '\n';
  *fos_ << "Final time         :" << next_time << '\n';
  *fos_ << "Time step          :" << time_step << '\n';
  *fos_ << delim << std::endl;

  Thyra_ModelEvaluator::InArgs<ST>  in_args  = solver.createInArgs();
  Thyra_ModelEvaluator::OutArgs<ST> out_args = solver.createOutArgs();

  auto& me = dynamic_cast<Albany::ModelEvaluator&>(*model_evaluators_[subdomain]);

  Teuchos::RCP<Tempus::SolutionHistory<ST>> solution_history;
  Teuchos::RCP<Tempus::SolutionState<ST>>   current_state;

  solver.evalModel(in_args, out_args);

  // Allocate current solution vectors
  this_x_[subdomain]    = Thyra::createMember(me.get_x_space());
  this_xdot_[subdomain] = Thyra::createMember(me.get_x_space());

  // Check whether solver did OK.
  auto const status = piro_tempus_solver.getTempusIntegratorStatus();

  if (status == Tempus::Status::FAILED) {
    *fos_ << "\nINFO: Unable to solve Thermal problem for subdomain " << subdomain << '\n';
    failed_ = true;
    return;
  }

  // If solver is OK, extract solution

  solution_history = piro_tempus_solver.getSolutionHistory();
  current_state    = solution_history->getCurrentState();

  Thyra::copy(*current_state->getX(), this_x_[subdomain].ptr());
  Thyra::copy(*current_state->getXDot(), this_xdot_[subdomain].ptr());

  // Write solution back to model evaluator, app, and discretization
  // so the next time step starts from this solution (warm start).
  {
    auto nv = me.getNominalValues();
    nv.set_x(this_x_[subdomain]);
    nv.set_x_dot(this_xdot_[subdomain]);
    me.setNominalValues(nv);
  }
  auto& app = *apps_[subdomain];
  app.setX(this_x_[subdomain]);
  app.setXdot(this_xdot_[subdomain]);
  app.getDiscretization()->writeSolutionToMeshDatabase(
      *this_x_[subdomain], *this_xdot_[subdomain], next_time);

  failed_ = false;
}

void
ACEThermoMechanical::AdvanceMechanicalDynamics(
    int const    subdomain,
    bool const   is_initial_state,
    double const current_time,
    double const next_time,
    double const time_step) const
{
  failed_ = false;
  // Solve for each subdomain
  Thyra::ResponseOnlyModelEvaluatorBase<ST>& solver = *(solvers_[subdomain]);

  if (mechanical_solver_ == MechanicalSolver::Tempus) {
    auto& piro_tempus_solver = dynamic_cast<Piro::TempusSolver<ST>&>(solver);
    piro_tempus_solver.setStartTime(current_time);
    piro_tempus_solver.setFinalTime(next_time);
    piro_tempus_solver.setInitTimeStep(time_step);

    std::string const delim(72, '=');
    *fos_ << "Initial time       :" << current_time << '\n';
    *fos_ << "Final time         :" << next_time << '\n';
    *fos_ << "Time step          :" << time_step << '\n';
    *fos_ << delim << std::endl;

    Thyra_ModelEvaluator::InArgs<ST>  in_args  = solver.createInArgs();
    Thyra_ModelEvaluator::OutArgs<ST> out_args = solver.createOutArgs();

    auto& me = dynamic_cast<Albany::ModelEvaluator&>(*model_evaluators_[subdomain]);

    Teuchos::RCP<Tempus::SolutionHistory<ST>> solution_history;
    Teuchos::RCP<Tempus::SolutionState<ST>>   current_state;

    solver.evalModel(in_args, out_args);

    // Allocate current solution vectors
    this_x_[subdomain]       = Thyra::createMember(me.get_x_space());
    this_xdot_[subdomain]    = Thyra::createMember(me.get_x_space());
    this_xdotdot_[subdomain] = Thyra::createMember(me.get_x_space());

    // Check whether solver did OK.
    auto const status = piro_tempus_solver.getTempusIntegratorStatus();

    if (status == Tempus::Status::FAILED) {
      *fos_ << "\nINFO: Unable to solve Mechanical problem for subdomain " << subdomain << '\n';
      failed_ = true;
      return;
    }
    // If solver is OK, extract solution

    solution_history = piro_tempus_solver.getSolutionHistory();
    current_state    = solution_history->getCurrentState();

    Thyra::copy(*current_state->getX(), this_x_[subdomain].ptr());
    Thyra::copy(*current_state->getXDot(), this_xdot_[subdomain].ptr());
    Thyra::copy(*current_state->getXDotDot(), this_xdotdot_[subdomain].ptr());

  } else if (mechanical_solver_ == MechanicalSolver::TrapezoidRule) {
    auto&             piro_tr_solver = dynamic_cast<Piro::TrapezoidRuleSolver<ST>&>(solver);
    auto& me = dynamic_cast<Albany::ModelEvaluator&>(*model_evaluators_[subdomain]);

    std::string const delim(72, '=');
    *fos_ << "Initial time       :" << current_time << '\n';
    *fos_ << "Final time         :" << next_time << '\n';
    *fos_ << "Time step          :" << time_step << '\n';
    *fos_ << delim << std::endl;
    // Disable initial acceleration solve unless in initial time-step.
    // This should speed up code by ~2x.
    if (current_time != initial_time_) {
      piro_tr_solver.disableCalcInitAccel();
    }

    Thyra_ModelEvaluator::InArgs<ST>  in_args  = solver.createInArgs();
    Thyra_ModelEvaluator::OutArgs<ST> out_args = solver.createOutArgs();

    // Request the solution as response num_g (the extra response slot
    // that TrapezoidRuleSolver adds for the solution vector).
    auto const num_g = solver.Ng() - 1;
    auto       gx_out = Thyra::createMember(solver.get_g_space(num_g));
    out_args.set_g(num_g, gx_out);

    solver.evalModel(in_args, out_args);

    // Check whether solver did OK.
    auto&      tr_nox_solver              = *(piro_tr_solver.getNOXSolver());
    auto&      thyra_nox_nonlinear_solver = *(tr_nox_solver.getSolver());
    auto&      const_nox_generic_solver   = *(thyra_nox_nonlinear_solver.getNOXSolver());
    auto&      nox_generic_solver         = const_cast<NOX::Solver::Generic&>(const_nox_generic_solver);
    auto const status                     = nox_generic_solver.getStatus();

    if (status == NOX::StatusTest::Failed) {
      *fos_ << "\nINFO: Unable to solve Mechanical problem for subdomain " << subdomain << '\n';
      failed_ = true;
      return;
    }

    // Obtain the solution from the response and time derivatives from
    // the decorator.  Note: tr_decorator.get_x() returns the predictor,
    // not the solved solution.  The solved x is in gx_out (response num_g).
    auto& tr_decorator = *(piro_tr_solver.getDecorator());
    auto  xdot_rcp     = tr_decorator.get_x_dot();
    auto  xdotdot_rcp  = tr_decorator.get_x_dotdot();

    this_x_[subdomain]       = Thyra::createMember(me.get_x_space());
    this_xdot_[subdomain]    = Thyra::createMember(me.get_x_space());
    this_xdotdot_[subdomain] = Thyra::createMember(me.get_x_space());

    Thyra::copy(*gx_out, this_x_[subdomain].ptr());
    Thyra::copy(*xdot_rcp, this_xdot_[subdomain].ptr());
    Thyra::copy(*xdotdot_rcp, this_xdotdot_[subdomain].ptr());

    // Write solution back to model evaluator, app, and discretization
    // so the next time step starts from this solution (warm start).
    {
      auto nv = me.getNominalValues();
      nv.set_x(this_x_[subdomain]);
      nv.set_x_dot(this_xdot_[subdomain]);
      me.setNominalValues(nv);
    }
    auto& app = *apps_[subdomain];
    app.setX(this_x_[subdomain]);
    app.setXdot(this_xdot_[subdomain]);
    app.setXdotdot(this_xdotdot_[subdomain]);
    app.getDiscretization()->writeSolutionToMeshDatabase(
        *this_x_[subdomain], *this_xdot_[subdomain], *this_xdotdot_[subdomain], next_time);

    // Element death is handled inside J2Erosion::operator() via the
    // failure_state field.  Elements marked as failed in a previous step
    // get near-zero stiffness (residual elastic modulus) on subsequent
    // steps, effectively removing them from the structural response
    // without modifying STK parts or invalidating the bucket structure.
  } else {
    ALBANY_ABORT("Unknown time integrator for mechanics. Only Tempus and Piro Trapezoid Rule supported.");
  }

  failed_ = false;
}

void
ACEThermoMechanical::setExplicitUpdateInitialGuessForCoupling(ST const current_time, ST const time_step) const
{
  // do an explicit update to form the initial guess for the schwarz
  // iteration
  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    auto& app = *apps_[subdomain];

    Thyra_Vector& ic_x       = *ics_x_[subdomain];
    Thyra_Vector& ic_xdot    = *ics_xdot_[subdomain];
    Thyra_Vector& ic_xdotdot = *ics_xdotdot_[subdomain];

    auto& me = dynamic_cast<Albany::ModelEvaluator&>(*model_evaluators_[subdomain]);
    if (current_time == 0) {
      this_x_[subdomain]       = Thyra::createMember(me.get_x_space());
      this_xdot_[subdomain]    = Thyra::createMember(me.get_x_space());
      this_xdotdot_[subdomain] = Thyra::createMember(me.get_x_space());
    }

    const ST aConst = time_step * time_step / 2.0;
    Thyra::V_StVpStV(this_x_[subdomain].ptr(), time_step, ic_xdot, aConst, ic_xdotdot);
    Thyra::Vp_V(this_x_[subdomain].ptr(), ic_x, 1.0);

    // This is the initial guess that I want to apply to the subdomains before
    // the schwarz solver starts
    auto x_rcp       = this_x_[subdomain];
    auto xdot_rcp    = this_xdot_[subdomain];
    auto xdotdot_rcp = this_xdotdot_[subdomain];

    // setting x, xdot and xdotdot in the albany application
    app.setX(x_rcp);
    app.setXdot(xdot_rcp);
    app.setXdotdot(xdotdot_rcp);

    // in order to get the Schwarz boundary conditions right, we need to set the
    // state in the discretization - IKT FIXME: may not be relevant for ACE coupling
    Teuchos::RCP<Albany::AbstractDiscretization> const& app_disc = app.getDiscretization();

    app_disc->writeSolutionToMeshDatabase(*x_rcp, *xdot_rcp, *xdotdot_rcp, current_time);
  }
}

void
ACEThermoMechanical::setICVecs(ST const time, int const subdomain) const
{
  auto const prob_type       = prob_types_[subdomain];
  auto const is_initial_time = time <= initial_time_ + initial_time_step_;

  if (is_initial_time == true) {
    // initial time-step: get initial solution from nominalValues in ME
    auto&       me = dynamic_cast<Albany::ModelEvaluator&>(*model_evaluators_[subdomain]);
    auto const& nv = me.getNominalValues();

    ics_x_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x()), ics_x_[subdomain].ptr());

    ics_xdot_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x_dot()), ics_xdot_[subdomain].ptr());

    if (prob_type == MECHANICAL) {
      ics_xdotdot_[subdomain] = Thyra::createMember(me.get_x_space());
      Thyra::copy(*(nv.get_x_dot_dot()), ics_xdotdot_[subdomain].ptr());
      auto nrm = norm_2(*ics_xdotdot_[subdomain]);
    }
  }

  else {
    // subsequent time steps: update ic vecs based on fields in stk discretization
    auto& abs_disc = *discs_[subdomain];
    auto& stk_disc = static_cast<Albany::STKDiscretization&>(abs_disc);
    auto  x_mv     = stk_disc.getSolutionMV();

    // Update ics_x_ and its time-derivatives
    ics_x_[subdomain] = Thyra::createMember(x_mv->col(0)->space());
    Thyra::copy(*x_mv->col(0), ics_x_[subdomain].ptr());

    ics_xdot_[subdomain] = Thyra::createMember(x_mv->col(1)->space());
    Thyra::copy(*x_mv->col(1), ics_xdot_[subdomain].ptr());

    if (prob_type == MECHANICAL) {
      ics_xdotdot_[subdomain] = Thyra::createMember(x_mv->col(2)->space());
      Thyra::copy(*x_mv->col(2), ics_xdotdot_[subdomain].ptr());
      auto nrm = norm_2(*ics_xdotdot_[subdomain]);
    }
  }
}

void
ACEThermoMechanical::doQuasistaticOutput(ST const time) const
{
  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    if (do_outputs_[subdomain] == true) {
      auto& stk_mesh_struct = *stk_mesh_structs_[subdomain];

      stk_mesh_struct.exoOutputInterval = 1;
      stk_mesh_struct.exoOutput         = true;

      auto& abs_disc = *discs_[subdomain];
      auto& stk_disc = static_cast<Albany::STKDiscretization&>(abs_disc);

      // Do not dereference this RCP. Leads to SEGFAULT (!?)
      auto x_mv_rcp = stk_disc.getSolutionMV();
      stk_disc.writeSolutionMV(*x_mv_rcp, time);
      stk_mesh_struct.exoOutput = false;
    }
  }
}

void
ACEThermoMechanical::doDynamicInitialOutput(ST const time, int const subdomain) const
{
  if (do_outputs_[subdomain] == false) return;

  // M4: single coupled output. Only the mechanical subdomain writes the
  // Exodus file (it solves last, so its snapshot has the fully updated
  // coupled state). The thermal subdomain has no output broker anyway
  // (createPersistentApps removed its "Exodus Output File Name"); this
  // early return also guards against do_outputs_ being forced true.
  if (prob_types_[subdomain] != MECHANICAL) return;

  auto const xMV_rcp         = apps_[subdomain]->getAdaptSolMgr()->getOverlappedSolution();
  auto&      abs_disc        = *discs_[subdomain];
  auto&      stk_disc        = static_cast<Albany::STKDiscretization&>(abs_disc);
  auto&      stk_mesh_struct = *stk_mesh_structs_[subdomain];

  stk_mesh_struct.exoOutputInterval = 1;
  stk_mesh_struct.exoOutput         = true;
  stk_disc.writeSolutionMV(*xMV_rcp, time, true);
  stk_mesh_struct.exoOutput = false;
}

// Sequential ThermoMechanical coupling loop, quasistatic
void
ACEThermoMechanical::ThermoMechanicalLoopQuasistatics() const
{
  // IKT 6/5/2020: not implemented for now.
}

}  // namespace LCM
