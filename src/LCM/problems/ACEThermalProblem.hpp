// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ACETHERMALPROBLEM_HPP
#define ACETHERMALPROBLEM_HPP

#include "Albany_AbstractProblem.hpp"
#include "Albany_MaterialDatabase.hpp"
#include "Albany_ProblemUtils.hpp"
#include "PHAL_ConvertFieldType.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_SaveStateField.hpp"
#include "PHAL_Workset.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {

/*!
 * \brief Abstract interface for representing finite element
 * problem.
 */
class ACEThermalProblem : public AbstractProblem
{
 public:
  //! Default constructor
  ACEThermalProblem(
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<ParamLib>&               paramLib,
      int const                                   num_dim,
      Teuchos::RCP<Teuchos::Comm<int> const>&     comm);

  //! Destructor
  ~ACEThermalProblem();

  //! Return number of spatial dimensions
  virtual int
  spatialDimension() const
  {
    return num_dim_;
  }

  //! Get boolean telling code if SDBCs are utilized
  virtual bool
  useSDBCs() const
  {
    return use_sdbcs_;
  }

  ///
  /// Get boolean telling code if Adaptation is utilized
  ///
  virtual bool
  haveAdaptation() const
  {
    return false;
  }

  //! Build the PDE instantiations, boundary conditions, and initial solution
  virtual void
  buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> mesh_specs, StateManager& state_mgr);

  // Build evaluators
  virtual Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
  buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>&      fm0,
      Albany::MeshSpecsStruct const&              mesh_specs,
      Albany::StateManager&                       state_mgr,
      Albany::FieldManagerChoice                  fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& response_list);

  //! Each problem must generate it's list of valide parameters
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidProblemParameters() const;

 private:
  //! Private to prohibit copying
  ACEThermalProblem(const ACEThermalProblem&);

  //! Private to prohibit copying
  ACEThermalProblem&
  operator=(const ACEThermalProblem&);

 public:
  //! Main problem setup routine. Not directly called, but indirectly by
  //! following functions
  template <typename EvalT>
  Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>&      fm0,
      Albany::MeshSpecsStruct const&              mesh_specs,
      Albany::StateManager&                       state_mgr,
      Albany::FieldManagerChoice                  fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& response_list);

  void
  constructDirichletEvaluators(std::vector<std::string> const& node_set_ids);
  void
  constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& mesh_specs);

 protected:
  int num_dim_;  // number spatial dimensions

  Teuchos::RCP<Albany::MaterialDatabase> material_db_;

  Teuchos::ArrayRCP<std::string> eb_names_;

  const Teuchos::RCP<Teuchos::ParameterList> params_;

  Teuchos::RCP<Teuchos::Comm<int> const> comm_;

  Teuchos::RCP<Albany::Layouts> dl_;

  /// Boolean marking whether SDBCs are used
  bool use_sdbcs_;

  /// Boolean marking whether adaptation is used
  bool have_adaptation_;

  // Stabilization-related parameters
  bool        use_stab_{false};
  double      stab_value_{1.0};
  double      x_max_{0.0};
  double      z_max_{0.0};
  double      max_time_stab_{1.0e10};
  std::string tau_type_;
  std::string stab_type_;
};

}  // namespace Albany

#include "ACETempStabilization.hpp"
#include "ACETempStandAloneResid.hpp"
#include "ACEThermalParameters.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_Utils.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::ACEThermalProblem::constructEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits>&      fm0,
    Albany::MeshSpecsStruct const&              mesh_specs,
    Albany::StateManager&                       state_mgr,
    Albany::FieldManagerChoice                  field_manager_choice,
    const Teuchos::RCP<Teuchos::ParameterList>& response_list)
{
  using PHAL::AlbanyTraits;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::string;
  using std::vector;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  Albany::StateStruct::MeshFieldEntity entity;

  // Collect problem-specific response parameters
  Teuchos::RCP<Teuchos::ParameterList> params_from_prob = Teuchos::rcp(new Teuchos::ParameterList("Response Parameters from Problem"));

  const CellTopologyData* const elem_top = &mesh_specs.ctd;

  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepid_basis = Albany::getIntrepid2Basis(*elem_top);
  RCP<shards::CellTopology>                              cell_type      = rcp(new shards::CellTopology(elem_top));

  int const num_nodes    = intrepid_basis->getCardinality();
  int const workset_size = mesh_specs.worksetSize;

  Intrepid2::DefaultCubatureFactory     cub_factory;
  RCP<Intrepid2::Cubature<PHX::Device>> cell_cubature = cub_factory.create<PHX::Device, RealType, RealType>(*cell_type, mesh_specs.cubatureDegree);

  int const num_qps_cell = cell_cubature->getNumPoints();
  int const num_vertices = cell_type->getNodeCount();

  // Problem is steady or transient
  ALBANY_PANIC(
      number_of_time_deriv != 1,
      "ACETempStandAloneProblem must be defined as transient "
      "calculation.");

  *out << "Field Dimensions: Workset=" << workset_size << ", Vertices= " << num_vertices << ", Nodes= " << num_nodes << ", QuadPts= " << num_qps_cell
       << ", Dim= " << num_dim_ << "\n";

  dl_ = rcp(new Albany::Layouts(workset_size, num_vertices, num_nodes, num_qps_cell, num_dim_));
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl_);

  // Have to register boundary_indicators in the mesh before the
  // discretization is built
  auto find_cell_boundary_indicator = std::find(this->requirements.begin(), this->requirements.end(), "cell_boundary_indicator");

  auto find_node_boundary_indicator = std::find(this->requirements.begin(), this->requirements.end(), "node_boundary_indicator");

  if (find_cell_boundary_indicator != this->requirements.end()) {
    auto entity = StateStruct::ElemData;
    state_mgr.registerStateVariable("cell_boundary_indicator", dl_->cell_scalar2, mesh_specs.ebName, false, &entity);
  }
  if (find_node_boundary_indicator != this->requirements.end()) {
    auto entity = StateStruct::ElemData;
    state_mgr.registerStateVariable("node_boundary_indicator", dl_->node_scalar, mesh_specs.ebName, false, &entity);
  }

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits>> ev;

  Teuchos::ArrayRCP<string> dof_names(neq);
  dof_names[0] = "Temperature";
  Teuchos::ArrayRCP<string> dof_names_dot(neq);
  dof_names_dot[0] = "Temperature_dot";
  Teuchos::ArrayRCP<string> resid_names(neq);
  resid_names[0] = "Temperature Residual";

  fm0.template registerEvaluator<EvalT>(evalUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot));

  fm0.template registerEvaluator<EvalT>(evalUtils.constructScatterResidualEvaluator(false, resid_names));

  fm0.template registerEvaluator<EvalT>(evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>(evalUtils.constructMapToPhysicalFrameEvaluator(cell_type, cell_cubature));

  fm0.template registerEvaluator<EvalT>(evalUtils.constructComputeBasisFunctionsEvaluator(cell_type, intrepid_basis, cell_cubature));

  for (unsigned int i = 0; i < neq; i++) {
    fm0.template registerEvaluator<EvalT>(evalUtils.constructDOFInterpolationEvaluator(dof_names[i]));

    fm0.template registerEvaluator<EvalT>(evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[i]));

    fm0.template registerEvaluator<EvalT>(evalUtils.constructDOFGradInterpolationEvaluator(dof_names[i]));
  }

  // Register ACE_Bluff_Salinity
  {
    Teuchos::RCP<Teuchos::ParameterList> p         = Teuchos::rcp(new Teuchos::ParameterList);
    std::string                          stateName = "ACE_Bluff_Salinity";
    Albany::StateStruct::MeshFieldEntity entity    = Albany::StateStruct::QuadPoint;
    p                                              = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
    // Load parameter using its field name
    std::string fieldName = "ACE_Bluff_SalinityRead";
    p->set<std::string>("Field Name", fieldName);
    p->set<std::string>("State Name", stateName);
    p->set<Teuchos::RCP<PHX::DataLayout>>("State Field Layout", dl_->qp_scalar);
    using LoadStateFieldST = PHAL::LoadStateFieldBase<EvalT, PHAL::AlbanyTraits, typename EvalT::ScalarT>;
    ev                     = Teuchos::rcp(new LoadStateFieldST(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // IKT, 11/7/2022: the following logic is for determining whether we're in the initial
  // timestep or not within ACE::ThermalParameters, which tells the code where to get
  // the bluff_salinity_ field from.
  double current_time = 0.0;
  if (params->isParameter("ACE Sequential Thermomechanical")) {
    if (params->isParameter("ACE Thermomechanical Problem Current Time")) current_time = params->get<double>("ACE Thermomechanical Problem Current Time");
  }

  // ACE thermal parameters
  {
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("ACE_Therm_Cond QP Variable Name", "ACE_Therm_Cond");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("BF Name", "BF");
    p->set<string>("ACE_Thermal_Inertia QP Variable Name", "ACE_Thermal_Inertia");
    p->set<string>("ACE_Bluff_Salinity QP Variable Name", "ACE_Bluff_Salinity");
    p->set<double>("Current Time", current_time);
    p->set<string>("ACE_Bluff_SalinityRead QP Variable Name", "ACE_Bluff_SalinityRead");
    p->set<string>("ACE_Ice_Saturation QP Variable Name", "ACE_Ice_Saturation");
    p->set<string>("ACE_Freezing_Curve QP Variable Name", "ACE_Freezing_Curve");
    p->set<string>("ACE_Density QP Variable Name", "ACE_Density");
    p->set<string>("ACE_Heat_Capacity QP Variable Name", "ACE_Heat_Capacity");
    p->set<string>("ACE_Water_Saturation QP Variable Name", "ACE_Water_Saturation");
    p->set<string>("ACE_Porosity QP Variable Name", "ACE_Porosity");
    p->set<string>("ACE Temperature QP Variable Name", "Temperature");
    p->set<string>("ACE_Therm_Cond Gradient Node Variable Name", "ACE_Therm_Cond Gradient Node");
    p->set<string>("ACE_Therm_Cond Gradient QP Variable Name", "ACE_Therm_Cond Gradient QP");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<RCP<DataLayout>>("Node Data Layout", dl_->node_scalar);
    p->set<RCP<DataLayout>>("QP Scalar Data Layout", dl_->qp_scalar);
    p->set<RCP<DataLayout>>("QP Vector Data Layout", dl_->qp_vector);
    p->set<RCP<DataLayout>>("Node Vector Data Layout", dl_->node_vector);
    p->set<RCP<DataLayout>>("Node QP Vector Data Layout", dl_->node_qp_vector);

    p->set<RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("ACE Thermal Parameters");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    p->set<Teuchos::ArrayRCP<string>>("Element Block Names", eb_names_);

    if (material_db_ != Teuchos::null) p->set<RCP<Albany::MaterialDatabase>>("MaterialDB", material_db_);

    ev = rcp(new LCM::ACEThermalParameters<EvalT, AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
    // Save ACE_Bluff_Salinity to the output Exodus file
    {
      std::string stateName = "ACE_Bluff_Salinity";
      entity                = Albany::StateStruct::QuadPoint;
      p                     = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE_Bluff_Salinity");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0)) fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
    // Save ACE_Ice_Saturation to the output Exodus file
    {
      std::string stateName = "ACE_Ice_Saturation";
      entity                = Albany::StateStruct::QuadPoint;
      p                     = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE_Ice_Saturation");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0)) fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
    // Save ACE_Freezing_Curve to the output Exodus file
    {
      std::string stateName = "ACE_Freezing_Curve";
      entity                = Albany::StateStruct::QuadPoint;
      p                     = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE_Freezing_Curve");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0)) fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }

    // Save ACE_Density to the output Exodus file
    {
      std::string stateName = "ACE_Density";
      entity                = Albany::StateStruct::QuadPoint;
      p                     = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE_Density");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0)) fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
    // Save ACE_Heat_Capacity to the output Exodus file
    {
      std::string stateName = "ACE_Heat_Capacity";
      entity                = Albany::StateStruct::QuadPoint;
      p                     = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE_Heat_Capacity");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0)) fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
    // Save ACE_Therm_Cond to the output Exodus file
    {
      std::string stateName = "ACE_Thermal_Cond";
      entity                = Albany::StateStruct::QuadPoint;
      p                     = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE_Therm_Cond");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0)) fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
    // Save ACE_Thermal_Inertia to the output Exodus file
    {
      std::string stateName = "ACE_Thermal_Inertia";
      entity                = Albany::StateStruct::QuadPoint;
      p                     = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE_Thermal_Inertia");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0)) fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
    // Save ACE_Water_Saturation to the output Exodus file
    {
      std::string stateName = "ACE_Water_Saturation";
      entity                = Albany::StateStruct::QuadPoint;
      p                     = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE_Water_Saturation");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0)) fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
    // Save ACE_Porosity to the output Exodus file
    {
      std::string stateName = "ACE_Porosity";
      entity                = Albany::StateStruct::QuadPoint;
      p                     = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE_Porosity");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0)) fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
  }

  {  // Temperature stabilization
    RCP<ParameterList> p = rcp(new ParameterList("Temperature Stabilization"));

    // Input
    p->set<RCP<DataLayout>>("Node QP Scalar Data Layout", dl_->node_qp_scalar);
    p->set<RCP<DataLayout>>("QP Scalar Data Layout", dl_->qp_scalar);
    p->set<RCP<DataLayout>>("QP Vector Data Layout", dl_->qp_vector);
    p->set<RCP<DataLayout>>("Node QP Vector Data Layout", dl_->node_qp_vector);
    p->set<double>("Stabilization Parameter Value", stab_value_);
    p->set<std::string>("Jacobian Det Name", "Jacobian Det");
    p->set<string>("ACE_Therm_Cond Gradient QP Variable Name", "ACE_Therm_Cond Gradient QP");
    p->set<string>("Tau Type", tau_type_);

    // Output
    p->set<string>("Tau Name", "ACE Thermal Stabilization Parameter Tau");
    p->set<RCP<DataLayout>>("Node Scalar Data Layout", dl_->node_scalar);

    ev = rcp(new LCM::ACETempStabilization<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  {  // Temperature Resid
    RCP<ParameterList> p = rcp(new ParameterList("Temperature Resid"));

    // Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("QP Time Derivative Variable Name", "Temperature_dot");
    p->set<string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");

    p->set<RCP<DataLayout>>("Node QP Scalar Data Layout", dl_->node_qp_scalar);
    p->set<RCP<DataLayout>>("QP Scalar Data Layout", dl_->qp_scalar);
    p->set<RCP<DataLayout>>("QP Vector Data Layout", dl_->qp_vector);
    p->set<RCP<DataLayout>>("Node QP Vector Data Layout", dl_->node_qp_vector);

    p->set<string>("ACE_Therm_Cond QP Variable Name", "ACE_Therm_Cond");
    p->set<string>("ACE_Thermal_Inertia QP Variable Name", "ACE_Thermal_Inertia");
    p->set<RCP<DataLayout>>("QP Scalar Data Layout", dl_->qp_scalar);
    p->set<string>("ACE_Therm_Cond Gradient QP Variable Name", "ACE_Therm_Cond Gradient QP");
    p->set<std::string>("Jacobian Det Name", "Jacobian Det");
    p->set<bool>("Use Stabilization", use_stab_);
    p->set<double>("Max Value of x-Coord", x_max_);
    p->set<double>("Max Value of z-Coord", z_max_);
    p->set<double>("Max Stabilization Time", max_time_stab_);
    p->set<string>("Tau Name", "ACE Thermal Stabilization Parameter Tau");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<string>("Stabilization Type", stab_type_);

    // Output
    p->set<string>("Residual Name", "Temperature Residual");
    p->set<RCP<DataLayout>>("Node Scalar Data Layout", dl_->node_scalar);

    ev = rcp(new LCM::ACETempStandAloneResid<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (field_manager_choice == Albany::BUILD_RESID_FM) {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl_->dummy);
    fm0.requireField<EvalT>(res_tag);
    return res_tag.clone();
  }

  else if (field_manager_choice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> resp_utils(dl_);
    return resp_utils.constructResponses(fm0, *response_list, params_from_prob, state_mgr, &mesh_specs);
  }

  return Teuchos::null;
}

#endif  // ALBANY_THERMALNONLINEARSOURCEPROBLEM_HPP
