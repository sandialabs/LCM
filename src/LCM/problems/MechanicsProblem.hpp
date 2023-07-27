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
#include "DislocationDensity.hpp"
#include "NOX_StatusTest_ModelEvaluatorFlag.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Workset.hpp"
#include "SolutionSniffer.hpp"

static int dir_count = 0;  // counter for registration of dirichlet_field

namespace Albany {

///
/// \brief Definition for the Mechanics Problem
///
class MechanicsProblem : public AbstractProblem
{
 public:
  using FC = typename Kokkos::DynRankView<RealType, PHX::Device>;

  ///
  /// Default constructor
  ///
  MechanicsProblem(
      Teuchos::RCP<Teuchos::ParameterList> const& params,
      Teuchos::RCP<ParamLib> const&               param_lib,
      int const                                   num_dims,
      Teuchos::RCP<AAdapt::rc::Manager> const&    rc_mgr,
      Teuchos::RCP<Teuchos::Comm<int> const>&     commT,
      double const                                init_time = 0.0);
  ///
  /// Destructor
  ///
  virtual ~MechanicsProblem(){};

  ///
  /// Return number of spatial dimensions
  ///
  virtual int
  spatialDimension() const
  {
    return num_dims_;
  }

  ///
  /// Get boolean telling code if SDBCs are utilized
  ///
  virtual bool
  useSDBCs() const
  {
    return use_sdbcs_;
  }

  ///
  /// Build the PDE instantiations, boundary conditions, initial solution
  ///
  virtual void
  buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct>> meshSpecs, StateManager& stateMgr);

  ///
  /// Build evaluators
  ///
  virtual Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
  buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>&      fm0,
      MeshSpecsStruct const&                      meshSpecs,
      StateManager&                               stateMgr,
      FieldManagerChoice                          fmchoice,
      Teuchos::RCP<Teuchos::ParameterList> const& responseList);

  ///
  /// Each problem must generate it's list of valid parameters
  ///
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidProblemParameters() const;

  ///
  /// Retrieve the state data
  ///
  void
  getAllocatedStates(Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC>>> old_state, Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC>>> new_state) const;

  ///
  /// Add a custom NOX Status Test (for example, to trigger a global load step
  /// reduction)
  ///
  void
  applyProblemSpecificSolverSettings(Teuchos::RCP<Teuchos::ParameterList> params);

  ///
  /// Main problem setup routine.
  /// Not directly called, but indirectly by following functions
  ///
  template <typename EvalT>
  Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>&      fm0,
      MeshSpecsStruct const&                      meshSpecs,
      StateManager&                               stateMgr,
      FieldManagerChoice                          fmchoice,
      Teuchos::RCP<Teuchos::ParameterList> const& responseList);

  ///
  /// Setup for the dirichlet BCs
  ///
  void
  constructDirichletEvaluators(const MeshSpecsStruct& meshSpecs);

  ///
  /// Setup for the traction BCs
  ///
  void
  constructNeumannEvaluators(Teuchos::RCP<MeshSpecsStruct> const& meshSpecs);

 private:
  ///
  /// Private to prohibit copying
  ///
  MechanicsProblem(const MechanicsProblem&);

  ///
  /// Private to prohibit copying
  ///
  MechanicsProblem&
  operator=(const MechanicsProblem&);

 protected:
  ///
  /// Enumerated type describing how a variable appears
  ///
  enum MECH_VAR_TYPE
  {
    MECH_VAR_TYPE_NONE,      //! Variable does not appear
    MECH_VAR_TYPE_CONSTANT,  //! Variable is a constant
    MECH_VAR_TYPE_DOF,       //! Variable is a degree-of-freedom
    MECH_VAR_TYPE_TIMEDEP    //! Variable is stepped by LOCA in time
  };

  // Source function type
  enum SOURCE_TYPE
  {
    SOURCE_TYPE_NONE,     //! No source
    SOURCE_TYPE_INPUT,    //! Source is specified in input file
    SOURCE_TYPE_MATERIAL  //! Source is specified in material database
  };

  ///
  /// Accessor for variable type
  ///
  void
  getVariableType(Teuchos::ParameterList& param_list, std::string const& default_type, MECH_VAR_TYPE& variable_type, bool& have_variable, bool& have_equation);

  ///
  /// Conversion from enum to string
  ///
  std::string
  variableTypeToString(MECH_VAR_TYPE const variable_type);

  ///
  /// Construct a string for consistent output with surface elements
  ///
  // std::string stateString(std::string, bool);

  /// Boundary conditions on source term
  bool have_source_;

  /// Boolean marking whether SDBCs are used
  bool use_sdbcs_;

  /// Type of thermal source that is in effect
  SOURCE_TYPE
  thermal_source_;

  /// Has the thermal source been evaluated in this element block?
  bool thermal_source_evaluated_;

  /// num of dimensions
  int num_dims_;

  /// number of integration points
  int num_pts_;

  /// number of element nodes
  int num_nodes_;

  /// number of element vertices
  int num_vertices_;

  /// boolean marking whether using composite tet
  bool composite_;

  /// Problem parameter list
  Teuchos::RCP<Teuchos::ParameterList> const params_;

  /// Type of mechanics variable (disp or acc)
  MECH_VAR_TYPE
  mech_type_;

  /// Variable types
  MECH_VAR_TYPE
  temperature_type_;

  MECH_VAR_TYPE
  dislocation_density_type_;

  MECH_VAR_TYPE
  pore_pressure_type_;

  MECH_VAR_TYPE
  transport_type_;

  MECH_VAR_TYPE
  hydrostress_type_;

  MECH_VAR_TYPE
  damage_type_;

  MECH_VAR_TYPE
  stab_pressure_type_;

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

  /// Dislocation transport physics
  bool have_dislocation_density_{false};
  bool have_dislocation_density_eq_{false};

  /// Have mesh adaptation - both the "Adaptation" sublist exists and the user
  /// has specified that the method
  ///    is "RPI Albany Size"
  ///
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

  /// old state data
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC>>> old_state_;

  /// new state data
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC>>> new_state_;

  /// Reference configuration manager for mesh adaptation with ref config
  /// updating.
  Teuchos::RCP<AAdapt::rc::Manager> rc_mgr_;

  /// User defined NOX Status Test that allows model evaluators to set the NOX
  /// status to "failed". This forces a global load step reduction.
  Teuchos::RCP<NOX::StatusTest::Generic> nox_status_test_;

  std::vector<std::string> variables_problem_ = {"Displacement"};

  std::vector<std::string> variables_auxiliary_ =
      {"Temperature", "ACE Temperature", "DislocationDensity", "Pore Pressure", "Transport", "HydroStress", "Damage", "Stabilized Pressure"};

  double init_time_; 

};  // class MechanicsProblem

}  // namespace Albany

#endif
