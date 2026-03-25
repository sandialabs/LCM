// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

// NOTE: NBC (Neumann BC) propagation after element deactivation is NOT yet
// implemented. When elements are eroded/deactivated, newly exposed surfaces
// do not automatically receive the NBCs that were on the original boundary.
// This requires future work to detect new boundary faces after element death
// and apply the appropriate Neumann conditions to them.

#include "ACE_ThermoMechanicalIM.hpp"

#include <algorithm>
#include <fstream>
#include <set>

#include "AAdapt_Erosion.hpp"
#include "ACEcommon.hpp"
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
#include "Topology.hpp"
#include "Topology_FailureCriterion.hpp"

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

ACEThermoMechanicalIM::ACEThermoMechanicalIM(Teuchos::RCP<Teuchos::ParameterList> const& app_params, Teuchos::RCP<Teuchos::Comm<int> const> const& comm)
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
  ALBANY_ASSERT(num_subdomains_ <= 2, "ACEThermoMechanicalIM solver requires no more than 2 models!");

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

ACEThermoMechanicalIM::~ACEThermoMechanicalIM() { return; }

Teuchos::RCP<Thyra_VectorSpace const>
ACEThermoMechanicalIM::get_x_space() const
{
  return Teuchos::null;
}

Teuchos::RCP<Thyra_VectorSpace const>
ACEThermoMechanicalIM::get_f_space() const
{
  return Teuchos::null;
}

Teuchos::RCP<Thyra_VectorSpace const>
ACEThermoMechanicalIM::get_p_space(int) const
{
  return Teuchos::null;
}

Teuchos::RCP<Thyra_VectorSpace const>
ACEThermoMechanicalIM::get_g_space(int) const
{
  return Teuchos::null;
}

Teuchos::RCP<const Teuchos::Array<std::string>>
ACEThermoMechanicalIM::get_p_names(int) const
{
  return Teuchos::null;
}

Teuchos::ArrayView<std::string const>
ACEThermoMechanicalIM::get_g_names(int) const
{
  ALBANY_ABORT("not implemented");
  return Teuchos::ArrayView<std::string const>(Teuchos::null);
}

Thyra_ModelEvaluator::InArgs<ST>
ACEThermoMechanicalIM::getNominalValues() const
{
  return this->createInArgsImpl();
}

Thyra_ModelEvaluator::InArgs<ST>
ACEThermoMechanicalIM::getLowerBounds() const
{
  return Thyra_ModelEvaluator::InArgs<ST>();  // Default value
}

Thyra_ModelEvaluator::InArgs<ST>
ACEThermoMechanicalIM::getUpperBounds() const
{
  return Thyra_ModelEvaluator::InArgs<ST>();  // Default value
}

Teuchos::RCP<Thyra::LinearOpBase<ST>>
ACEThermoMechanicalIM::create_W_op() const
{
  return Teuchos::null;
}

Teuchos::RCP<Thyra::PreconditionerBase<ST>>
ACEThermoMechanicalIM::create_W_prec() const
{
  return Teuchos::null;
}

Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST>>
ACEThermoMechanicalIM::get_W_factory() const
{
  return Teuchos::null;
}

Thyra_ModelEvaluator::InArgs<ST>
ACEThermoMechanicalIM::createInArgs() const
{
  return this->createInArgsImpl();
}

Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
ACEThermoMechanicalIM::getApps() const
{
  return apps_;
}

void
ACEThermoMechanicalIM::set_failed(char const* msg)
{
  failed_          = true;
  failure_message_ = msg;
  return;
}

void
ACEThermoMechanicalIM::clear_failed()
{
  failed_ = false;
  return;
}

bool
ACEThermoMechanicalIM::get_failed() const
{
  return failed_;
}

// Create operator form of dg/dx for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST>>
ACEThermoMechanicalIM::create_DgDx_op_impl(int /* j */) const
{
  return Teuchos::null;
}

// Create operator form of dg/dx_dot for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST>>
ACEThermoMechanicalIM::create_DgDx_dot_op_impl(int /* j */) const
{
  return Teuchos::null;
}

// Create InArgs
Thyra_InArgs
ACEThermoMechanicalIM::createInArgsImpl() const
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
ACEThermoMechanicalIM::createOutArgsImpl() const
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
ACEThermoMechanicalIM::evalModelImpl(Thyra_ModelEvaluator::InArgs<ST> const&, Thyra_ModelEvaluator::OutArgs<ST> const&) const
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
ACEThermoMechanicalIM::createPersistentApps()
{
  Teuchos::TimeMonitor timer(*Teuchos::TimeMonitor::getNewTimer("ACE IM: Create Persistent Apps"));
  for (int subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    auto& app_params = *init_pls_[subdomain];
    auto& disc_params = app_params.sublist("Discretization");

    // For the mechanical problem, use element death instead of destructive
    // mesh erosion.  Remove the Adaptation section so that the
    // TrapezoidRuleSolver does not trigger mesh adaptation.
    if (prob_types_[subdomain] == MECHANICAL) {
      auto& problem_params = app_params.sublist("Problem");
      problem_params.remove("Adaptation", false);
      problem_params.set("Disable Erosion", true);
      disc_params.set<std::string>("Exodus Solution Name", "displacement");
      disc_params.set<std::string>("Exodus SolutionDot Name", "velocity");
      disc_params.set<std::string>("Exodus SolutionDotDot Name", "acceleration");
    }

    // Name thermal fields
    if (prob_types_[subdomain] == THERMAL) {
      disc_params.set<std::string>("Exodus Solution Name", "temperature");
      disc_params.set<std::string>("Exodus SolutionDot Name", "temperature_dot");
    }

    // Force all subdomains to use the same workset size (-1 = one workset
    // per element block) so that in-memory state transfer between apps
    // operates on arrays with matching sizes.
    disc_params.set("Workset Size", -1);

    solver_factories_[subdomain] = Teuchos::rcp(new Albany::SolverFactory(init_pls_[subdomain], comm_));

    Teuchos::RCP<Albany::Application> app;
    solvers_[subdomain] = solver_factories_[subdomain]->createAndGetAlbanyApp(app, comm_, comm_);
    apps_[subdomain] = app;

    auto stk_disc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(app->getDiscretization());
    discs_[subdomain] = stk_disc;
    stk_mesh_structs_[subdomain] = stk_disc->getSTKMeshStruct();

    do_outputs_[subdomain]      = stk_mesh_structs_[subdomain]->exoOutput;
    do_outputs_init_[subdomain] = stk_mesh_structs_[subdomain]->exoOutput;

    model_evaluators_[subdomain] = solver_factories_[subdomain]->returnModel();

    curr_x_[subdomain] = Teuchos::null;

    // Calculate and store the min value of the z-coordinate for wave pressure NBC
    if (prob_types_[subdomain] == THERMAL) {
      Teuchos::RCP<const Thyra_MultiVector> coord_mv = stk_disc->getCoordMV();
      Teuchos::RCP<const Thyra_Vector> z_coord = coord_mv->col(2);
      zmin_ = Thyra::min(*z_coord);
    }
  }

  // Cache initial internal states
  for (int subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    auto& state_mgr = apps_[subdomain]->getStateMgr();
    fromTo(state_mgr.getStateArrays(), internal_states_[subdomain]);
  }
}

void
ACEThermoMechanicalIM::transferThermalToMechanical(int thermal_sub, int mech_sub) const
{
  auto& thermal_state_mgr = apps_[thermal_sub]->getStateMgr();
  auto& mech_state_mgr = apps_[mech_sub]->getStateMgr();

  auto& thermal_states = thermal_state_mgr.getStateArrays().elemStateArrays;
  auto& mech_states = mech_state_mgr.getStateArrays().elemStateArrays;

  // Transfer ACE_Ice_Saturation (and any other shared QP state fields).
  // Both apps must use the same workset configuration (enforced in
  // createPersistentApps) so that state arrays have matching sizes.
  std::vector<std::string> shared_fields = {"ACE_Ice_Saturation"};
  for (auto const& field_name : shared_fields) {
    for (size_t ws = 0; ws < thermal_states.size() && ws < mech_states.size(); ++ws) {
      auto it_thermal = thermal_states[ws].find(field_name);
      auto it_mech = mech_states[ws].find(field_name);
      if (it_thermal != thermal_states[ws].end() && it_mech != mech_states[ws].end()) {
        auto const& src = it_thermal->second;
        auto&       dst = it_mech->second;
        ALBANY_ASSERT(src.size() == dst.size(),
            "ACE_Ice_Saturation size mismatch at ws=" << ws
            << ": thermal=" << src.size() << " mechanical=" << dst.size()
            << ". Both apps must use the same Workset Size.");
        for (size_t i = 0; i < src.size(); ++i) {
          dst[i] = src[i];
        }
      }
    }
  }
}

void
ACEThermoMechanicalIM::transferMechanicalToThermal(int mech_sub, int thermal_sub) const
{
  // Mesh topology changes (element deactivation) are automatically
  // visible to the thermal app since both apps share the same
  // STK BulkData when using the active_part mechanism.
  // No explicit field transfer needed.
}

bool
ACEThermoMechanicalIM::continueSolve() const
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

// Sequential ThermoMechanical coupling loop, dynamic (in-memory version)
void
ACEThermoMechanicalIM::ThermoMechanicalLoopDynamics() const
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
            Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE IM: Restore Thermal States"));
            fromTo(internal_states_[subdomain], state_mgr.getStateArrays());
          }
          {
            Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE IM: Thermal Solve"));
            AdvanceThermalDynamics(subdomain, is_initial_state, current_time, next_time, time_step);
          }
          if (failed_ == false) {
            {
              Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE IM: Save Thermal States"));
              fromTo(state_mgr.getStateArrays(), internal_states_[subdomain]);
            }
            {
              Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE IM: Thermal Output"));
              doDynamicInitialOutput(next_time, subdomain);
            }
          }
        }
        if (prob_type == MECHANICAL && failed_ == false) {
          *fos_ << "Problem            :Mechanical\n";
          auto& app       = *apps_[subdomain];
          auto& state_mgr = app.getStateMgr();
          {
            Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE IM: Restore Mechanical States"));
            fromTo(internal_states_[subdomain], state_mgr.getStateArrays());
          }
          // Transfer thermal results AFTER restoring mechanical states,
          // so the thermal-to-mechanical transfer is not overwritten.
          if (thermal_sub >= 0) {
            Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE IM: Transfer Thermal->Mechanical"));
            transferThermalToMechanical(thermal_sub, subdomain);
          }
          // Set death status on the Application for scatter skip and orphan fix.
          // Extract per-cell death indicators from the PREVIOUS step's
          // cached failure_state values: index (cell, 0) in the flat array.
          {
            auto& app = *apps_[subdomain];
            auto& esa = internal_states_[subdomain].element_state_arrays;
            app.death_status_vecs_.resize(esa.size());
            for (size_t ws = 0; ws < esa.size(); ++ws) {
              auto it = esa[ws].find("failure_state");
              if (it != esa[ws].end()) {
                auto& flat = it->second;
                // failure_state layout is (cell, qp).  Determine num_qps
                // from the Albany state array dimensions.
                auto& sa    = state_mgr.getStateArray(Albany::StateManager::ELEM, ws);
                auto  sa_it = sa.find("failure_state");
                int   num_qps   = 1;
                int   num_cells = flat.size();
                if (sa_it != sa.end()) {
                  Albany::StateStruct::FieldDims dims;
                  sa_it->second.dimensions(dims);
                  num_cells = dims[0];
                  if (dims.size() > 1) num_qps = dims[1];
                }
                auto ds = Teuchos::rcp(new std::vector<double>(num_cells, 0.0));
                for (int c = 0; c < num_cells; ++c) {
                  (*ds)[c] = flat[c * num_qps];
                }
                app.death_status_vecs_[ws] = ds;
              }
            }
          }
          {
            Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE IM: Mechanical Solve"));
            AdvanceMechanicalDynamics(subdomain, is_initial_state, current_time, next_time, time_step);
          }
          if (failed_ == false) {
            {
              Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE IM: Save Mechanical States"));
              fromTo(state_mgr.getStateArrays(), internal_states_[subdomain]);
            }
            {
              Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("ACE IM: Mechanical Output"));
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
ACEThermoMechanicalIM::AdvanceThermalDynamics(
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
ACEThermoMechanicalIM::AdvanceMechanicalDynamics(
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
ACEThermoMechanicalIM::setExplicitUpdateInitialGuessForCoupling(ST const current_time, ST const time_step) const
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
ACEThermoMechanicalIM::setICVecs(ST const time, int const subdomain) const
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
ACEThermoMechanicalIM::doQuasistaticOutput(ST const time) const
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
ACEThermoMechanicalIM::doDynamicInitialOutput(ST const time, int const subdomain) const
{
  if (do_outputs_[subdomain] == false) return;

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
ACEThermoMechanicalIM::ThermoMechanicalLoopQuasistatics() const
{
  // IKT 6/5/2020: not implemented for now.
}

}  // namespace LCM
