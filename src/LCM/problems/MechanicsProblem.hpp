// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef MECHANICSPROBLEM_HPP
#define MECHANICSPROBLEM_HPP

#include "AAdapt_RC_Manager.hpp"
#include "Albany_AbstractProblem.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_MaterialDatabase.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_Utils.hpp"
#include "NOX_StatusTest_ModelEvaluatorFlag.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Workset.hpp"
#include "SolutionSniffer.hpp"

namespace Albany {

class MechanicsProblem : public AbstractProblem
{
 public:
  using FC = Kokkos::DynRankView<RealType, PHX::Device>;

  MechanicsProblem(
      Teuchos::RCP<Teuchos::ParameterList> const& params,
      Teuchos::RCP<ParamLib> const&               param_lib,
      int const                                   num_dims,
      Teuchos::RCP<AAdapt::rc::Manager> const&    rc_mgr,
      Teuchos::RCP<Teuchos::Comm<int> const>&     commT);

  virtual ~MechanicsProblem() {};

  MechanicsProblem(MechanicsProblem const&)            = delete;
  MechanicsProblem& operator=(MechanicsProblem const&) = delete;

  virtual int
  spatialDimension() const
  {
    return num_dims_;
  }

  virtual bool
  useSDBCs() const
  {
    return use_sdbcs_;
  }

  virtual bool
  haveAdaptation() const
  {
    return have_adaptation_;
  }

  virtual void
  buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct>> meshSpecs, StateManager& stateMgr);

  virtual Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
  buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>&      fm0,
      MeshSpecsStruct const&                      meshSpecs,
      StateManager&                               stateMgr,
      FieldManagerChoice                          fmchoice,
      Teuchos::RCP<Teuchos::ParameterList> const& responseList);

  Teuchos::RCP<Teuchos::ParameterList const>
  getValidProblemParameters() const;

  void
  getAllocatedStates(Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC>>> old_state, Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC>>> new_state) const;

  /// Add a custom NOX Status Test (for example, to trigger a global load step
  /// reduction)
  void
  applyProblemSpecificSolverSettings(Teuchos::RCP<Teuchos::ParameterList> params);

  /// Main problem setup routine. Not directly called, but indirectly by the
  /// buildEvaluators / constructDirichletEvaluators / constructNeumannEvaluators
  /// entry points.
  template <typename EvalT>
  Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>&      fm0,
      MeshSpecsStruct const&                      meshSpecs,
      StateManager&                               stateMgr,
      FieldManagerChoice                          fmchoice,
      Teuchos::RCP<Teuchos::ParameterList> const& responseList);

  void
  constructDirichletEvaluators(const MeshSpecsStruct& meshSpecs);

  void
  constructNeumannEvaluators(Teuchos::RCP<MeshSpecsStruct> const& meshSpecs);

 protected:
  /// How a variable appears in the problem.
  enum MECH_VAR_TYPE
  {
    MECH_VAR_TYPE_NONE,      //! Variable does not appear
    MECH_VAR_TYPE_CONSTANT,  //! Variable is a constant
    MECH_VAR_TYPE_DOF,       //! Variable is a degree-of-freedom
    MECH_VAR_TYPE_TIMEDEP    //! Variable is stepped by LOCA in time
  };

  /// Source function type.
  enum SOURCE_TYPE
  {
    SOURCE_TYPE_NONE,     //! No source
    SOURCE_TYPE_INPUT,    //! Source is specified in input file
    SOURCE_TYPE_MATERIAL  //! Source is specified in material database
  };

  void
  getVariableType(Teuchos::ParameterList& param_list, std::string const& default_type, MECH_VAR_TYPE& variable_type, bool& have_variable, bool& have_equation);

  std::string
  variableTypeToString(MECH_VAR_TYPE const variable_type);

  /// Boundary conditions on source term
  bool have_source_;

  /// Boolean marking whether SDBCs are used
  bool use_sdbcs_;

  /// Boolean marking whether adaptation is used
  bool have_adaptation_;

  /// Type of thermal source that is in effect
  SOURCE_TYPE thermal_source_;

  /// Has the thermal source been evaluated in this element block?
  bool thermal_source_evaluated_;

  int num_dims_;
  int num_pts_;
  int num_nodes_;
  int num_vertices_;

  /// Using composite tet elements
  bool composite_;

  /// Problem parameter list
  Teuchos::RCP<Teuchos::ParameterList> const params_;

  /// Type of mechanics variable (disp or acc)
  MECH_VAR_TYPE mech_type_;

  MECH_VAR_TYPE temperature_type_;
  MECH_VAR_TYPE pore_pressure_type_;
  MECH_VAR_TYPE transport_type_;
  MECH_VAR_TYPE hydrostress_type_;
  MECH_VAR_TYPE damage_type_;
  MECH_VAR_TYPE stab_pressure_type_;

  /// Mechanics
  bool have_mech_{true};
  bool have_mech_eq_{true};

  /// Temperature
  bool have_temperature_{false};

  /// Use default "classic" heat conduction equation
  bool have_temperature_eq_{false};

  /// Have ACE temperature (handling is different than temperature above)
  bool have_ace_temperature_{false};

  /// Use ACE heat conduction equation
  bool have_ace_temperature_eq_{false};

  /// Pore pressure
  bool have_pore_pressure_{false};
  bool have_pore_pressure_eq_{false};

  /// Transport
  bool have_transport_{false};
  bool have_transport_eq_{false};

  /// Projected hydrostatic stress term in transport equation
  bool have_hydrostress_{false};
  bool have_hydrostress_eq_{false};

  /// Damage
  bool have_damage_{false};
  bool have_damage_eq_{false};

  /// Stabilized pressure
  bool have_stab_pressure_{false};
  bool have_stab_pressure_eq_{false};

  /// Mesh adaptation: "Adaptation" sublist exists and the method is
  /// "RPI Albany Size".
  bool have_sizefield_adaptation_{false};

  /// Dynamic tempus solution method
  bool dynamic_tempus_{false};

  /// Have a Peridynamics block
  bool have_peridynamics_{false};

  /// Topology adaptation (adaptive insertion)
  bool have_topmod_adaptation_{false};

  /// Is a coupled sequential ACE thermo-mechanical problem
  bool is_ace_sequential_thermomechanical_{false};

  /// Data layouts
  Teuchos::RCP<Layouts> dl_;

  /// RCP to matDB object
  Teuchos::RCP<MaterialDatabase> material_db_;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC>>> old_state_;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC>>> new_state_;

  /// Reference configuration manager for mesh adaptation with ref config
  /// updating.
  Teuchos::RCP<AAdapt::rc::Manager> rc_mgr_;

  /// User-defined NOX Status Test that allows model evaluators to set the NOX
  /// status to "failed". This forces a global load step reduction.
  Teuchos::RCP<NOX::StatusTest::Generic> nox_status_test_;

  std::vector<std::string> variables_problem_ = {"Displacement"};

  std::vector<std::string> variables_auxiliary_ =
      {"Temperature", "ACE Temperature", "Pore Pressure", "Transport", "HydroStress", "Damage", "Stabilized Pressure"};
};

}  // namespace Albany

#endif
