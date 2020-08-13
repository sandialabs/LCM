
// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_ACEThermoMechanical_hpp)
#define LCM_ACEThermoMechanical_hpp

#include <functional>

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Albany_Application.hpp"
#include "Albany_MaterialDatabase.hpp"
#include "Albany_ModelEvaluator.hpp"
#include "Albany_SolverFactory.hpp"
#include "Piro_NOXSolver.hpp"
#include "StateVarUtils.hpp"
#include "Thyra_AdaptiveSolutionManager.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"
#include "Thyra_ModelEvaluatorDelegatorBase.hpp"
#include "Thyra_ResponseOnlyModelEvaluatorBase.hpp"

namespace LCM {

class ACEAdaptiveState : public Thyra::AdaptiveStateBase
{
 public:
  ACEAdaptiveState(Teuchos::RCP<Thyra::ModelEvaluator<ST>> const& model) : AdaptiveStateBase(model) {}
  ~ACEAdaptiveState() {}
  void
  buildSolutionGroup()
  {
  }
};

class ACEModelEvaluatorDelegator : public Thyra::ModelEvaluatorDelegatorBase<ST>
{
 public:
  ACEModelEvaluatorDelegator(Teuchos::RCP<Thyra::ModelEvaluator<ST>> const& model_rcp) : model_rcp_(model_rcp) {}
  Teuchos::RCP<Thyra::ModelEvaluator<ST>>
  getNonconstUnderlyingModel()
  {
    return model_rcp_;
  }

  ModelEvaluatorBase::OutArgs<ST>
  createOutArgsImpl() const
  {
    return ModelEvaluatorBase::OutArgs<ST>();
  }

  void
  evalModelImpl(ModelEvaluatorBase::InArgs<ST> const&, ModelEvaluatorBase::OutArgs<ST> const&) const
  {
  }

 private:
  Teuchos::RCP<Thyra::ModelEvaluator<ST>> model_rcp_;
};

///
/// ACEThermoMechanical coupling class
///
class ACEThermoMechanical : public Thyra::ResponseOnlyModelEvaluatorBase<ST>
{
 public:
  /// Constructor
  ACEThermoMechanical(
      Teuchos::RCP<Teuchos::ParameterList> const&   app_params,
      Teuchos::RCP<Teuchos::Comm<int> const> const& comm);

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
  evalModelImpl(
      Thyra::ModelEvaluatorBase::InArgs<ST> const&  in_args,
      Thyra::ModelEvaluatorBase::OutArgs<ST> const& out_args) const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgsImpl() const;

  /// Sequential thermo-mechanical coupling loops
  void
  ThermoMechanicalLoopQuasistatics() const;

  void
  ThermoMechanicalLoopDynamics() const;

  void
  AdvanceThermalDynamics(
      int const    subdomain,
      bool const   is_initial_state,
      double const current_time,
      double const next_time,
      double const time_step) const;

  void
  AdvanceMechanicalDynamics(
      int const    subdomain,
      bool const   is_initial_state,
      double const current_time,
      double const next_time,
      double const time_step) const;

  bool
  continueSolve() const;

  void
  createSolversAppsDiscsMEs(int const file_index, double const this_time = 0.0) const;

  void
  createThermalSolverAppDiscME(int const file_index, double const this_time) const;

  void
  createMechanicalSolverAppDiscME(int const file_index, double const this_time) const;

  void
  doQuasistaticOutput(ST const time) const;

  void
  doDynamicInitialOutput(ST const time, int const subdomain, int const stop) const;

  void 
  renamePrevWrittenExoFiles(const int subdomain, const int file_index) const; 

  void
  setExplicitUpdateInitialGuessForCoupling(ST const current_time, ST const time_step) const;

  void
  setICVecs(ST const time, int const subdomain) const;

  std::vector<Teuchos::RCP<Albany::SolverFactory>>                             solver_factories_;
  mutable std::vector<Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>> solvers_;
  mutable Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>                 apps_;
  mutable std::vector<Teuchos::RCP<Albany::AbstractSTKMeshStruct>>             stk_mesh_structs_;
  mutable std::vector<Teuchos::RCP<Albany::AbstractDiscretization>>            discs_;
  std::vector<Teuchos::RCP<Teuchos::ParameterList>>                            init_pls_;

  char const*  failure_message_{"No failure detected"};
  int          num_subdomains_{0};
  int          maximum_steps_{0};
  ST           initial_time_{0.0};
  ST           final_time_{0.0};
  ST           initial_time_step_{0.0};
  ST           min_time_step_{0.0};
  ST           max_time_step_{0.0};
  ST           reduction_factor_{0.0};
  ST           increase_factor_{0.0};
  int          output_interval_{1};
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

  // variable with previous thermal Exodus output file, for mechanical restarts
  mutable std::string prev_thermal_exo_outfile_name_;
  // variable with previous mechanical Exodus output file, for thermal restarts
  mutable std::string prev_mechanical_exo_outfile_name_;

  mutable std::vector<LCM::StateArrays> internal_states_;
  mutable std::vector<bool>             do_outputs_;
  mutable std::vector<bool>             do_outputs_init_;

  // Used if solving with loca or tempus
  mutable bool is_static_{false};
  mutable bool is_dynamic_{false};
  bool         std_init_guess_{false};

  enum PROB_TYPE
  {
    THERMAL,
    MECHANICAL
  };

  // std::vector mapping subdomain number to PROB_TYPE;
  mutable std::vector<PROB_TYPE> prob_types_;

  Teuchos::RCP<Teuchos::FancyOStream> fos_;

  Teuchos::RCP<Teuchos::ParameterList>   alt_system_params_;
  Teuchos::RCP<Teuchos::Comm<int> const> comm_;
  Teuchos::Array<std::string>            model_filenames_;
};

}  // namespace LCM

#endif  // LCM_ACEThermoMechanical_hpp
