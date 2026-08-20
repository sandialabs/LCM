
// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_ACEThermoMechanical_hpp)
#define LCM_ACEThermoMechanical_hpp


#include <functional>
#include <map>
#include <unordered_map>

#include "ACE_AdaptiveState.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Albany_Application.hpp"
#include "Albany_MaterialDatabase.hpp"
#include "Albany_ModelEvaluator.hpp"
#include "Albany_NodalFieldProjector.hpp"
#include "Albany_SolverFactory.hpp"
#include "Piro_NOXSolver.hpp"
#include "StateVarUtils.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"
#include "Thyra_ResponseOnlyModelEvaluatorBase.hpp"

namespace LCM {

///
/// ACEThermoMechanical coupling class
///
class ACEThermoMechanical : public Thyra::ResponseOnlyModelEvaluatorBase<ST>
{
 public:
  /// Constructor
  ACEThermoMechanical(Teuchos::RCP<Teuchos::ParameterList> const& app_params, Teuchos::RCP<Teuchos::Comm<int> const> const& comm);

  /// Destructor
  ~ACEThermoMechanical();

  /// Return solution vector map
  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  get_x_space() const;

  /// Return residual vector map
  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  get_f_space() const;

  /// Return parameter vector map
  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  get_p_space(int l) const;

  /// Return response function map
  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  get_g_space(int j) const;

  /// Return array of parameter names
  Teuchos::RCP<Teuchos::Array<std::string> const>
  get_p_names(int l) const;

  Teuchos::ArrayView<std::string const>
  get_g_names(int j) const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getNominalValues() const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getLowerBounds() const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getUpperBounds() const;

  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_W_op() const;

  /// Create preconditioner operator
  Teuchos::RCP<Thyra::PreconditionerBase<ST>>
  create_W_prec() const;

  Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const>
  get_W_factory() const;

  /// Create InArgs
  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgs() const;

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
  getApps() const;

  void
  set_failed(char const* msg);

  void
  clear_failed();

  bool
  get_failed() const;

  enum class ConvergenceCriterion
  {
    ABSOLUTE,
    RELATIVE,
    BOTH
  };
  enum class ConvergenceLogicalOperator
  {
    AND,
    OR
  };

 private:
  /// Create operator form of dg/dx for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_DgDx_op_impl(int j) const;

  /// Create operator form of dg/dx_dot for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_DgDx_dot_op_impl(int j) const;

  /// Create OutArgs
  Thyra::ModelEvaluatorBase::OutArgs<ST>
  createOutArgsImpl() const;

  /// Evaluate model on InArgs
  void
  evalModelImpl(Thyra::ModelEvaluatorBase::InArgs<ST> const& in_args, Thyra::ModelEvaluatorBase::OutArgs<ST> const& out_args) const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgsImpl() const;

  /// Sequential thermo-mechanical coupling loops
  void
  ThermoMechanicalLoopQuasistatics() const;

  void
  ThermoMechanicalLoopDynamics() const;

  void
  AdvanceThermalDynamics(int const subdomain, bool const is_initial_state, double const current_time, double const next_time, double const time_step) const;

  void
  AdvanceMechanicalDynamics(
      int const    subdomain,
      bool const   is_initial_state,
      double const current_time,
      double const next_time,
      double const time_step,
      bool const   death_resolve = false) const;

  // Reseed the mechanical solve's initial condition to the step-start state
  // (x_n, v_n, a_n) before every solve of the step -- solve #0, every outer
  // death-iteration re-solve, and every dt-cut retry. Each re-solve is then a
  // clean replay of the same time window with softer material, advancing only
  // the death state; the trapezoid decorator can no longer ratchet fictitious
  // velocity/acceleration across re-solves, and a diverged solve's poisoned
  // rate buffers are overwritten before the next attempt. Also reseeds the
  // decorator's acceleration IC buffer, which the solver reads directly.
  void reseedMechIC(int subdomain) const;

  bool
  continueSolve() const;

  // All-reduce the per-solve failure flag so every rank agrees on it. A solve
  // that fails on only some ranks (e.g. a gradual-death re-solve that goes
  // singular on the subdomain owning the softening front) would otherwise send
  // those ranks down the step-cut/restore branch while the rest proceed --
  // desynchronizing the collective workset rebuilds and deadlocking. Must be
  // called after every mechanical solve, before any rank branches on failed_.
  void globalizeFailed() const;

  void
  createPersistentApps();

  void
  doQuasistaticOutput(ST const time) const;

  void
  doDynamicInitialOutput(ST const time, int const subdomain) const;

  void
  setExplicitUpdateInitialGuessForCoupling(ST const current_time, ST const time_step) const;

  void
  setICVecs(ST const time, int const subdomain) const;

  //! Zero velocity and acceleration (this_xdot_/this_xdotdot_) at every node
  //! whose incident cells are all dead, so a calved/eroded block carries no
  //! momentum into the next step's warm start. Prevents the linearly growing
  //! acceleration of a decoupled, displacement-frozen dead DOF under the
  //! trapezoid rule from eventually breaking the mechanical solve.
  void
  zeroDeadNodeRates(int const subdomain) const;

  mutable std::vector<Teuchos::RCP<Albany::SolverFactory>>                     solver_factories_;
  mutable std::vector<Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>> solvers_;
  mutable Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>                 apps_;
  mutable std::vector<Teuchos::RCP<Albany::AbstractSTKMeshStruct>>             stk_mesh_structs_;
  mutable std::vector<Teuchos::RCP<Albany::AbstractDiscretization>>            discs_;
  std::vector<Teuchos::RCP<Teuchos::ParameterList>>                            init_pls_;

  char const*  failure_message_{"No failure detected"};
  int          num_subdomains_{0};
  int          maximum_steps_{0};
  mutable ST   initial_time_{0.0};
  mutable ST   final_time_{0.0};
  ST           initial_time_step_{0.0};
  ST           min_time_step_{0.0};
  ST           max_time_step_{0.0};
  ST           reduction_factor_{0.0};
  ST           increase_factor_{0.0};
  int          output_interval_{1};
  bool         output_preload_{false};
  mutable bool failed_{false};
  mutable bool converged_{false};
  mutable int  num_iter_{0};

  mutable ConvergenceCriterion       criterion_{ConvergenceCriterion::BOTH};
  mutable ConvergenceLogicalOperator operator_{ConvergenceLogicalOperator::AND};

  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST> const>> curr_x_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST> const>> prev_step_x_;

  mutable std::vector<Thyra::ModelEvaluatorBase::InArgs<ST>>   sub_inargs_;
  mutable std::vector<Thyra::ModelEvaluatorBase::OutArgs<ST>>  sub_outargs_;
  mutable std::vector<Teuchos::RCP<Thyra::ModelEvaluator<ST>>> model_evaluators_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     ics_x_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     ics_xdot_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     ics_xdotdot_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     prev_x_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     prev_xdot_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     prev_xdotdot_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     this_x_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     this_xdot_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     this_xdotdot_;
  // Step-start integrator state (x_n, v_n, a_n), snapshotted once per attempted
  // time step before solve #0. reseedMechIC restores the mechanical solve to
  // this state before every solve/re-solve/retry so each outer death-iteration
  // re-solve replays the same window from a clean initial condition.
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     step_start_x_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     step_start_xdot_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     step_start_xdotdot_;

  // Snapshot of every element-rank double field on the shared STK mesh
  // at the start of each global time step, restored when a subdomain
  // solve fails and the step is retried with a reduced size. States live
  // in the shared STK mesh and are written DURING residual fills, so a
  // diverged nonlinear solve leaves poisoned states behind; without the
  // restore every retry assembles from them. Keyed by field name and
  // element ID -- NOT by workset position: rebuildWorksets re-packs the
  // buckets between snapshot and restore, so positional copies (the
  // Schwarz fromTo idiom) are not layout-safe here.
  using SharedStateStore = std::map<std::string, std::unordered_map<stk::mesh::EntityId, std::vector<double>>>;
  mutable SharedStateStore pre_step_states_;

  // Second snapshot, taken AFTER the thermal solve and BEFORE the first
  // mechanical solve of the step. The outer death iteration re-solves the
  // mechanical equilibrium at fixed time, rewinding the mechanical physics to
  // this post-thermal state (restoreMechStates(keep_death=true)) between
  // re-solves while keeping the committed deaths. It must be post-thermal so a
  // re-solve judges failure on the step's warmed strength, not the step-start
  // (pre-thermal) strength.
  mutable SharedStateStore pre_mech_states_;

  void snapshotStatesInto(SharedStateStore& out) const;
  void restoreStatesFrom(SharedStateStore const& in, bool keep_death) const;
  void snapshotSharedMeshStates() const;
  void snapshotMechStates() const;
  void restoreSharedMeshStates() const;
  // Rewind the mechanical physics to the post-thermal snapshot for an outer
  // death-iteration re-solve; keep_death preserves the element-death bookkeeping
  // so deaths committed so far survive the rewind.
  void restoreMechStates(bool keep_death) const;

  // Read cell_death into the mechanical app's death_status_vecs_ and capture the
  // fully-dead node DOFs for the hold-in-place Dirichlet. Called before each
  // mechanical solve (initial and each outer-death-iteration re-solve) so newly
  // fully-faded cells are scatter-skipped and pinned on the next solve.
  void populateDeathStatus(int subdomain) const;

  // True if any owned cell is mid-fade (0 < death_decay_old < 1); the outer
  // death iteration's convergence test. Collective (MAX reduce).
  bool anyCellMidFade(int subdomain) const;

  //! Global number of owned elements still in the active part; "alive"
  //! when no active-part machinery exists. Drives the all-elements-dead
  //! stop in the time loop.
  long long countActiveElements() const;

  mutable std::vector<bool> do_outputs_;
  mutable std::vector<bool> do_outputs_init_;

  //! Per-subdomain nodal-field projectors. Built only for mechanical
  //! subdomains, whose "Project IP to Nodal Field" response never runs (the
  //! TrapezoidRule solver hands the observer a MultiVector, whose overload
  //! omits observeResponse). Driven from doDynamicInitialOutput; reads saved
  //! quadrature-point states, so it cannot perturb the coupled trajectory.
  mutable std::vector<std::vector<Teuchos::RCP<Albany::NodalFieldProjector>>> projectors_;

  bool std_init_guess_{false};
  // Start the mechanical subproblem from static equilibrium under its
  // body force (solve K x = f once at t0; v = a = 0) instead of Piro's
  // initial-acceleration heuristic. See Piro::TrapezoidRuleSolver::
  // enableStaticInitSolve.
  bool static_equilibrium_init_{false};

  //! True when the mesh was populated from a restart file, so the solution,
  //! its time derivatives and the element states all carry real values and
  //! must not be re-derived from scratch. Set in createPersistentApps.
  bool restarted_{false};

  enum PROB_TYPE
  {
    THERMAL,
    MECHANICAL
  };

  enum class MechanicalSolver
  {
    Tempus,
    LOCA,
    TrapezoidRule
  };

  mutable MechanicalSolver mechanical_solver_{MechanicalSolver::TrapezoidRule};

  // std::vector mapping subdomain number to PROB_TYPE;
  mutable std::vector<PROB_TYPE> prob_types_;

  Teuchos::RCP<Teuchos::FancyOStream> fos_;

  Teuchos::RCP<Teuchos::ParameterList>   alt_system_params_;
  Teuchos::RCP<Teuchos::Comm<int> const> comm_;
  Teuchos::Array<std::string>            model_filenames_;
  // Min value of z-coordinate in initial mesh - needed for wave pressure NBC
  mutable double zmin_{0.0};

  // Storage for defining time intervals where the time step will be prescribed.
  // This is needed for time stepping through user-defined events such as
  // impact loads, for example, aircraft loads in the ACI project.
  std::vector<RealType> event_initial_times_;
  std::vector<RealType> event_final_times_;
  std::vector<RealType> event_time_steps_;
};

}  // namespace LCM

#endif  // LCM_ACEThermoMechanical_hpp
