// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "ACE_ThermoMechanical.hpp"

#include "Albany_ModelEvaluator.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_Utils.hpp"
#include "MiniTensor.h"
#include "Piro_LOCASolver.hpp"
#include "Piro_TempusSolver.hpp"

namespace {
// In "Mechanics 3D", extract "Mechanics".
inline std::string
getName(std::string const& method)
{
  if (method.size() < 3) return method;
  return method.substr(0, method.size() - 3);
}
}  // namespace

namespace LCM {

ACEThermoMechanical::ACEThermoMechanical(
    Teuchos::RCP<Teuchos::ParameterList> const&   app_params,
    Teuchos::RCP<Teuchos::Comm<int> const> const& comm)
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

  // IKT QUESTION 6/4/2020: do we want to support quasistatic for thermo-mechanical
  // coupling??  Leaving it in for now.
  bool is_static{false};
  bool is_dynamic{false};

  // Create solver factories once at the beginning
  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    // Get parameters for each subdomain
    solver_factories_[subdomain] = Teuchos::rcp(new Albany::SolverFactory(model_filenames_[subdomain], comm_));
    // Get parameters from solver_factories_
    Teuchos::ParameterList& params = solver_factories_[subdomain]->getParameters();

    Teuchos::ParameterList& problem_params = params.sublist("Problem", true);

    // Get problem name to figure out if we have a thermal or mechanical problem for
    // this subdomain, and populate prob_types_ vector using this information.
    // IKT 6/4/2020: This is added to allow user to specify mechanical and thermal problems in
    // different orders.  I'm not sure if this will be of interest ultimately or not.
    const std::string problem_name = getName(problem_params.get<std::string>("Name"));
    if (problem_name == "Mechanics") {
      prob_types_[subdomain] = MECHANICAL;
    } else if (problem_name == "ACE Thermal") {
      prob_types_[subdomain] = THERMAL;
    } else {
      // Throw error if problem name is not Mechanics or ACE Thermal.
      // IKT 6/4/2020: I assume we only want to support Mechanics and ACE Thermal coupling.
      ALBANY_ABORT(
          "ACE Sequential thermo-mechanical solver only supports coupling of 'Mechanics' and 'ACE Thermal' problems!");
    }

    // Error checks - only needs to be done once at the beginning
    bool const have_piro = params.isSublist("Piro");
    ALBANY_ASSERT(have_piro == true, "Error! Piro sublist not found.\n");

    Teuchos::ParameterList& piro_params = params.sublist("Piro");

    std::string const msg{"All subdomains must have the same solution method (NOX or Tempus)"};

    if (subdomain == 0) {
      is_dynamic  = piro_params.isSublist("Tempus");
      is_static   = !is_dynamic;
      is_static_  = is_static;
      is_dynamic_ = is_dynamic;
    }
    if (is_static == true) { ALBANY_ASSERT(piro_params.isSublist("NOX") == true, msg); }
    if (is_dynamic == true) {
      ALBANY_ASSERT(piro_params.isSublist("Tempus") == true, msg);

      Teuchos::ParameterList& tempus_params = piro_params.sublist("Tempus");

      tempus_params.set("Abort on Failure", false);

      Teuchos::ParameterList& time_step_control_params =
          piro_params.sublist("Tempus").sublist("Tempus Integrator").sublist("Time Step Control");

      std::string const integrator_step_type = time_step_control_params.get("Integrator Step Type", "Constant");

      std::string const msg2{
          "Non-constant time-stepping through Tempus not supported "
          "with ACE sequential thermo-mechanical coupling; \n"
          "In this case, variable time-stepping is "
          "handled within the coupling loop.\n"
          "Please rerun with 'Integrator Step Type: "
          "Constant' in 'Time Step Control' sublist.\n"};
      ALBANY_ASSERT(integrator_step_type == "Constant", msg2);
    }
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

  oas.set_W_properties(Thyra_ModelEvaluator::DerivativeProperties(
      Thyra_ModelEvaluator::DERIV_LINEARITY_UNKNOWN, Thyra_ModelEvaluator::DERIV_RANK_FULL, true));

  return static_cast<Thyra_OutArgs>(oas);
}

// Evaluate model on InArgs
void
ACEThermoMechanical::evalModelImpl(Thyra_ModelEvaluator::InArgs<ST> const&, Thyra_ModelEvaluator::OutArgs<ST> const&)
    const
{
  if (is_dynamic_ == true) { ThermoMechanicalLoopDynamics(); }
  // IKT 6/4/2020: for now, throw error if trying to run quasi-statically.
  // Not sure if we want to ultimately support that case or not.
  ALBANY_ASSERT(is_static_ == false, "ACE Sequential Thermo-Mechanical solver currently supports dynamics only!");
  // if (is_static_ == true) { ThermoMechanicalLoopQuasistatics(); }
  return;
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

void
renameExodusFile(int const file_index, std::string& filename)
{
  std::ostringstream ss;
  if (filename.find(".exo") != std::string::npos) {
    ss << ".e-s." << file_index;
    filename.replace(filename.find(".exo"), std::string::npos, ss.str());
  } else {
    if (filename.find(".e") != std::string::npos) {
      ss << ".e-s." << file_index;
      filename.replace(filename.find(".e"), std::string::npos, ss.str());
    } else {
      ALBANY_ABORT("Exodus output file does not end in '.e' or '.exo' - cannot rename!\n");
    }
  }
}

}  // anonymous namespace

// The following routine creates the solvers, applications, discretizations
// and model evaluators for the run.  It is a separate routine to easily allow
// for recreation of these options from the coupling loops.
void
ACEThermoMechanical::createSolversAppsDiscsMEs(int const file_index, double const this_time) const
{
  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    auto const prob_type = prob_types_[subdomain];
    if (prob_type == THERMAL) {
      createThermalSolverAppDiscME(file_index, this_time);
    } else if (prob_type == MECHANICAL) {
      createMechanicalSolverAppDiscME(file_index, this_time);
    }
  }
}

void
ACEThermoMechanical::createThermalSolverAppDiscME(int const file_index, double const this_time) const
{
  auto const              subdomain      = 0;
  Teuchos::ParameterList& params         = solver_factories_[subdomain]->getParameters();
  Teuchos::ParameterList& problem_params = params.sublist("Problem", true);
  Teuchos::ParameterList& disc_params    = params.sublist("Discretization", true);
  std::string             filename       = disc_params.get<std::string>("Exodus Output File Name");
  renameExodusFile(file_index, filename);
  *fos_ << "Renaming output file to - " << filename << '\n';
  disc_params.set<const std::string>("Exodus Output File Name", filename);

  disc_params.set<std::string>("Exodus Solution Name", "temperature");
  disc_params.set<std::string>("Exodus SolutionDot Name", "temperature_dot");
  disc_params.set<bool>("Output DTK Field to Exodus", false);
  // After the initial run, we will do restarts from the previously written Exodus output file.
  if (file_index > 0) {
    // Change input Exodus file to previous mechanical Exodus output file, for restarts.
    disc_params.set<const std::string>("Exodus Input File Name", prev_mechanical_exo_outfile_name_);
    // Restart from time at beginning of when this function is called
    disc_params.set<double>("Restart Time", this_time);
    // Remove Initial Condition sublist
    problem_params.remove("Initial Condition", true);
  }

  Teuchos::RCP<Albany::Application>                       app{Teuchos::null};
  Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>> solver =
      solver_factories_[subdomain]->createAndGetAlbanyApp(app, comm_, comm_);

  solvers_[subdomain] = solver;
  apps_[subdomain]    = app;

  // Get STK mesh structs to control Exodus output interval
  Teuchos::RCP<Albany::AbstractDiscretization> disc = app->getDiscretization();
  discs_[subdomain]                                 = disc;

  Albany::STKDiscretization& stk_disc = *static_cast<Albany::STKDiscretization*>(disc.get());
  if (file_index == 0) { stk_disc.outputExodusSolutionInitialTime(true); }

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> ams = stk_disc.getSTKMeshStruct();
  do_outputs_[subdomain]                          = ams->exoOutput;
  do_outputs_init_[subdomain]                     = ams->exoOutput;
  stk_mesh_structs_[subdomain]                    = ams;
  model_evaluators_[subdomain]                    = solver_factories_[subdomain]->returnModel();
  curr_x_[subdomain]                              = Teuchos::null;
  prev_thermal_exo_outfile_name_                  = filename;
}

void
ACEThermoMechanical::createMechanicalSolverAppDiscME(int const file_index, double const this_time) const
{
  auto const              subdomain      = 1;
  Teuchos::ParameterList& params         = solver_factories_[subdomain]->getParameters();
  Teuchos::ParameterList& problem_params = params.sublist("Problem", true);
  Teuchos::ParameterList& disc_params    = params.sublist("Discretization", true);
  std::string             filename       = disc_params.get<std::string>("Exodus Output File Name");
  renameExodusFile(file_index, filename);
  *fos_ << "Renaming output file to - " << filename << '\n';
  disc_params.set<const std::string>("Exodus Output File Name", filename);

  disc_params.set<std::string>("Exodus Solution Name", "disp");
  disc_params.set<std::string>("Exodus SolutionDot Name", "disp_dot");
  disc_params.set<std::string>("Exodus SolutionDotDot Name", "disp_dotdot");
  disc_params.set<bool>("Output DTK Field to Exodus", false);
  // After the initial run, we will do restarts from the previously written Exodus output file.
  // Change input Exodus file to previous thermal Exodus output file, for restarts.
  disc_params.set<const std::string>("Exodus Input File Name", prev_thermal_exo_outfile_name_);
  // Restart from time at beginning of when this function is called
  disc_params.set<double>("Restart Time", this_time);
  // Remove Initial Condition sublist
  problem_params.remove("Initial Condition", true);
  // Set a flag to inform the mechanical problem to register the field ACE_Ice_Saturation
  problem_params.set("ACE Sequential Thermomechanical", true, "ACE Sequential Thermomechanical Problem");

  Teuchos::RCP<Albany::Application>                       app{Teuchos::null};
  Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>> solver =
      solver_factories_[subdomain]->createAndGetAlbanyApp(app, comm_, comm_);

  solvers_[subdomain] = solver;
  apps_[subdomain]    = app;

  // Get STK mesh structs to control Exodus output interval
  Teuchos::RCP<Albany::AbstractDiscretization> disc = app->getDiscretization();
  discs_[subdomain]                                 = disc;

  Albany::STKDiscretization& stk_disc = *static_cast<Albany::STKDiscretization*>(disc.get());
  if (file_index == 0) { stk_disc.outputExodusSolutionInitialTime(true); }

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> ams = stk_disc.getSTKMeshStruct();
  do_outputs_[subdomain]                          = ams->exoOutput;
  do_outputs_init_[subdomain]                     = ams->exoOutput;
  stk_mesh_structs_[subdomain]                    = ams;
  model_evaluators_[subdomain]                    = solver_factories_[subdomain]->returnModel();
  curr_x_[subdomain]                              = Teuchos::null;
  prev_mechanical_exo_outfile_name_               = filename;
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

  ST  time_step{initial_time_step_};
  int stop{0};
  ST  current_time{initial_time_};

  // Time-stepping loop
  while (stop < maximum_steps_ && current_time < final_time_) {
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
        // Create new solvers, apps, discs and model evaluators
        auto const prob_type = prob_types_[subdomain];
        if (prob_type == THERMAL) {
          createThermalSolverAppDiscME(stop, current_time);
        } else if (prob_type == MECHANICAL) {
          createMechanicalSolverAppDiscME(stop, current_time);
        }
        if (stop == 0) { setICVecsAndOutput(initial_time_, subdomain); }
        // Before the coupling loop, get internal states
        if (num_iter_ == 0) {
          auto& app       = *apps_[subdomain];
          auto& state_mgr = app.getStateMgr();
          fromTo(state_mgr.getStateArrays(), internal_states_[subdomain]);
        }
        *fos_ << delim << std::endl;
        *fos_ << "Subdomain          :" << subdomain << '\n';
        if (prob_type == MECHANICAL) {
          *fos_ << "Problem            :Mechanical\n";
          AdvanceMechanicalDynamics(subdomain, is_initial_state, current_time, next_time, time_step);
        } else {
          *fos_ << "Problem            :Thermal\n";
          AdvanceThermalDynamics(subdomain, is_initial_state, current_time, next_time, time_step);
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
      failed_ = false;

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

      // Restore previous solutions
      for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
        auto& me           = dynamic_cast<Albany::ModelEvaluator&>(*model_evaluators_[subdomain]);
        this_x_[subdomain] = Thyra::createMember(me.get_x_space());
        Thyra::copy(*ics_x_[subdomain], this_x_[subdomain].ptr());
        this_xdot_[subdomain] = Thyra::createMember(me.get_x_space());
        Thyra::copy(*ics_xdot_[subdomain], this_xdot_[subdomain].ptr());
        auto const prob_type = prob_types_[subdomain];
        if (prob_type == MECHANICAL) {
          this_xdotdot_[subdomain] = Thyra::createMember(me.get_x_space());
          Thyra::copy(*ics_xdotdot_[subdomain], this_xdotdot_[subdomain].ptr());
        }

        // restore the state manager with the state variables from the previous
        // loadstep.
        auto& app       = *apps_[subdomain];
        auto& state_mgr = app.getStateMgr();
        fromTo(internal_states_[subdomain], state_mgr.getStateArrays());

        // restore the solution in the discretization so the schwarz solver gets
        // the right boundary conditions!
        Teuchos::RCP<Thyra_Vector const> x_rcp_thyra    = ics_x_[subdomain];
        Teuchos::RCP<Thyra_Vector const> xdot_rcp_thyra = ics_xdot_[subdomain];
        Teuchos::RCP<Thyra_Vector const> xdotdot_rcp_thyra =
            (prob_type == MECHANICAL) ? ics_xdotdot_[subdomain] : Teuchos::null;

        Teuchos::RCP<Albany::AbstractDiscretization> const& app_disc = app.getDiscretization();

        if (prob_type == MECHANICAL) {
          app_disc->writeSolutionToMeshDatabase(*x_rcp_thyra, *xdot_rcp_thyra, *xdotdot_rcp_thyra, current_time);
        } else {
          app_disc->writeSolutionToMeshDatabase(*x_rcp_thyra, *xdot_rcp_thyra, current_time);
        }
      }

      // Jump to the beginning of the time-step loop without advancing
      // time to try to use a reduced step.
      continue;
    }

    // Update IC vecs and output solution to exodus file

    for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
      if (do_outputs_init_[subdomain] == true) {
        do_outputs_[subdomain] = output_interval_ > 0 ? (stop + 1) % output_interval_ == 0 : false;
      }
      setICVecsAndOutput(next_time, subdomain);
    }

    ++stop;
    current_time += time_step;

    // Step successful. Try to increase the time step.
    auto const increased_step = std::min(max_time_step_, increase_factor_ * time_step);

    if (increased_step > time_step) {
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
  // Restore solution from previous coupling iteration before solve
  // IKT 6/20/2020: do we still need this stuff when we do restarts?
  if (is_initial_state == true) {
    auto&       me     = dynamic_cast<Albany::ModelEvaluator&>(*model_evaluators_[subdomain]);
    auto const& nv     = me.getNominalValues();
    prev_x_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x()), prev_x_[subdomain].ptr());
    prev_xdot_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x_dot()), prev_xdot_[subdomain].ptr());
  } else {
    Thyra::put_scalar(0.0, prev_x_[subdomain].ptr());
    Thyra::copy(*this_x_[subdomain], prev_x_[subdomain].ptr());
    Thyra::put_scalar(0.0, prev_xdot_[subdomain].ptr());
    Thyra::copy(*this_xdot_[subdomain], prev_xdot_[subdomain].ptr());
  }
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

  // IKT FIXME 6/5/2020: need to check if this does the right thing for thermal problem
  // The only relevant internal state here would be the ice saturation
  // Restore internal states
  auto& app       = *apps_[subdomain];
  auto& state_mgr = app.getStateMgr();
  fromTo(internal_states_[subdomain], state_mgr.getStateArrays());

  Teuchos::RCP<Tempus::SolutionHistory<ST>> solution_history;
  Teuchos::RCP<Tempus::SolutionState<ST>>   current_state;

  if (std_init_guess_ == false) { piro_tempus_solver.setInitialGuess(prev_x_[subdomain]); }

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

  Teuchos::RCP<Thyra_Vector> x_diff_rcp = Thyra::createMember(me.get_x_space());
  Thyra::put_scalar<ST>(0.0, x_diff_rcp.ptr());
  Thyra::V_VpStV(x_diff_rcp.ptr(), *this_x_[subdomain], -1.0, *prev_x_[subdomain]);

  Teuchos::RCP<Thyra_Vector> xdot_diff_rcp = Thyra::createMember(me.get_x_space());
  Thyra::put_scalar<ST>(0.0, xdot_diff_rcp.ptr());
  Thyra::V_VpStV(xdot_diff_rcp.ptr(), *this_xdot_[subdomain], -1.0, *prev_xdot_[subdomain]);

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
  // Restore solution from previous coupling iteration before solve
  if (is_initial_state == true) {
    auto&       me     = dynamic_cast<Albany::ModelEvaluator&>(*model_evaluators_[subdomain]);
    auto const& nv     = me.getNominalValues();
    prev_x_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x()), prev_x_[subdomain].ptr());
    prev_xdot_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x_dot()), prev_xdot_[subdomain].ptr());
    prev_xdotdot_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x_dot_dot()), prev_xdotdot_[subdomain].ptr());
  } else {
    Thyra::put_scalar(0.0, prev_x_[subdomain].ptr());
    Thyra::copy(*this_x_[subdomain], prev_x_[subdomain].ptr());
    Thyra::put_scalar(0.0, prev_xdot_[subdomain].ptr());
    Thyra::copy(*this_xdot_[subdomain], prev_xdot_[subdomain].ptr());
    Thyra::put_scalar(0.0, prev_xdotdot_[subdomain].ptr());
    Thyra::copy(*this_xdotdot_[subdomain], prev_xdotdot_[subdomain].ptr());
  }
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

  // Restore internal states
  auto& app       = *apps_[subdomain];
  auto& state_mgr = app.getStateMgr();

  fromTo(internal_states_[subdomain], state_mgr.getStateArrays());

  Teuchos::RCP<Tempus::SolutionHistory<ST>> solution_history;
  Teuchos::RCP<Tempus::SolutionState<ST>>   current_state;

  if (std_init_guess_ == false) { piro_tempus_solver.setInitialGuess(prev_x_[subdomain]); }

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

  Teuchos::RCP<Thyra_Vector> x_diff_rcp = Thyra::createMember(me.get_x_space());
  Thyra::put_scalar<ST>(0.0, x_diff_rcp.ptr());
  Thyra::V_VpStV(x_diff_rcp.ptr(), *this_x_[subdomain], -1.0, *prev_x_[subdomain]);

  Teuchos::RCP<Thyra_Vector> xdot_diff_rcp = Thyra::createMember(me.get_x_space());
  Thyra::put_scalar<ST>(0.0, xdot_diff_rcp.ptr());
  Thyra::V_VpStV(xdot_diff_rcp.ptr(), *this_xdot_[subdomain], -1.0, *prev_xdot_[subdomain]);

  Teuchos::RCP<Thyra_Vector> xdotdot_diff_rcp = Thyra::createMember(me.get_x_space());
  Thyra::put_scalar<ST>(0.0, xdotdot_diff_rcp.ptr());
  Thyra::V_VpStV(xdotdot_diff_rcp.ptr(), *this_xdotdot_[subdomain], -1.0, *prev_xdotdot_[subdomain]);

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
ACEThermoMechanical::setICVecsAndOutput(ST const time, int const subdomain) const
{
  auto const prob_type       = prob_types_[subdomain];
  auto const is_initial_time = time <= initial_time_ + initial_time_step_;
  auto&      stk_mesh_struct = *stk_mesh_structs_[subdomain];
  auto&      abs_disc        = *discs_[subdomain];
  auto&      stk_disc        = static_cast<Albany::STKDiscretization&>(abs_disc);

  stk_mesh_struct.exoOutputInterval = 1;
  stk_mesh_struct.exoOutput         = do_outputs_[subdomain];

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
    }

    // Write initial condition to STK mesh
    Teuchos::RCP<Thyra_MultiVector const> const xMV = apps_[subdomain]->getAdaptSolMgr()->getOverlappedSolution();
    stk_disc.writeSolutionMV(*xMV, time, true);

  }

  else {
    // subsequent time steps: update ic vecs based on fields in stk discretization

    Teuchos::RCP<Thyra_MultiVector> x_mv = stk_disc.getSolutionMV();

    // Update ics_x_ and its time-derivatives
    ics_x_[subdomain] = Thyra::createMember(x_mv->col(0)->space());
    Thyra::copy(*x_mv->col(0), ics_x_[subdomain].ptr());

    ics_xdot_[subdomain] = Thyra::createMember(x_mv->col(1)->space());
    Thyra::copy(*x_mv->col(1), ics_xdot_[subdomain].ptr());

    if (prob_type == MECHANICAL) {
      ics_xdotdot_[subdomain] = Thyra::createMember(x_mv->col(2)->space());
      Thyra::copy(*x_mv->col(2), ics_xdotdot_[subdomain].ptr());
    }
  }

  stk_mesh_struct.exoOutput = false;
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

// Sequential ThermoMechanical coupling loop, quasistatic
void
ACEThermoMechanical::ThermoMechanicalLoopQuasistatics() const
{
  // IKT 6/5/2020: not implemented for now.
}

}  // namespace LCM
