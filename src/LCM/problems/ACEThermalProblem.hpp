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
  Teuchos::RCP<Teuchos::ParameterList> params_from_prob =
      Teuchos::rcp(new Teuchos::ParameterList("Response Parameters from Problem"));

  const CellTopologyData* const elem_top = &mesh_specs.ctd;

  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepid_basis = Albany::getIntrepid2Basis(*elem_top);
  RCP<shards::CellTopology>                              cell_type      = rcp(new shards::CellTopology(elem_top));

  int const num_nodes    = intrepid_basis->getCardinality();
  int const workset_size = mesh_specs.worksetSize;

  Intrepid2::DefaultCubatureFactory     cub_factory;
  RCP<Intrepid2::Cubature<PHX::Device>> cell_cubature =
      cub_factory.create<PHX::Device, RealType, RealType>(*cell_type, mesh_specs.cubatureDegree);

  int const num_qps_cell = cell_cubature->getNumPoints();
  int const num_vertices = cell_type->getNodeCount();

  // Problem is steady or transient
  ALBANY_PANIC(
      number_of_time_deriv != 1,
      "ACETempStandAloneProblem must be defined as transient "
      "calculation.");

  *out << "Field Dimensions: Workset=" << workset_size << ", Vertices= " << num_vertices << ", Nodes= " << num_nodes
       << ", QuadPts= " << num_qps_cell << ", Dim= " << num_dim_ << "\n";

  dl_ = rcp(new Albany::Layouts(workset_size, num_vertices, num_nodes, num_qps_cell, num_dim_));
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl_);

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

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructComputeBasisFunctionsEvaluator(cell_type, intrepid_basis, cell_cubature));

  for (unsigned int i = 0; i < neq; i++) {
    fm0.template registerEvaluator<EvalT>(evalUtils.constructDOFInterpolationEvaluator(dof_names[i]));

    fm0.template registerEvaluator<EvalT>(evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[i]));

    fm0.template registerEvaluator<EvalT>(evalUtils.constructDOFGradInterpolationEvaluator(dof_names[i]));
  }

  // ACE thermal parameters
  {
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("ACE Thermal Conductivity QP Variable Name", "ACE Thermal Conductivity");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("BF Name", "BF");
    p->set<string>("ACE Thermal Inertia QP Variable Name", "ACE Thermal Inertia");
    p->set<string>("ACE Bluff Salinity QP Variable Name", "ACE Bluff Salinity");
    p->set<string>("ACE Ice Saturation QP Variable Name", "ACE Ice Saturation");
    p->set<string>("ACE Density QP Variable Name", "ACE Density");
    p->set<string>("ACE Heat Capacity QP Variable Name", "ACE Heat Capacity");
    p->set<string>("ACE Water Saturation QP Variable Name", "ACE Water Saturation");
    p->set<string>("ACE Porosity QP Variable Name", "ACE Porosity");
    p->set<string>("ACE Temperature QP Variable Name", "Temperature");
    p->set<string>("ACE Thermal Conductivity Gradient Node Variable Name", "ACE Thermal Conductivity Gradient Node");
    p->set<string>("ACE Thermal Conductivity Gradient QP Variable Name", "ACE Thermal Conductivity Gradient QP");
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
    // Save ACE Bluff Salinity to the output Exodus file
    {
      std::string stateName = "ACE_Bluff_Salinity";
      entity                = Albany::StateStruct::QuadPoint;
      p = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE Bluff Salinity");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0))
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
    // Save ACE Ice Saturation to the output Exodus file
    {
      std::string stateName = "ACE_Ice_Saturation";
      entity                = Albany::StateStruct::QuadPoint;
      p = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE Ice Saturation");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0))
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
    // Save ACE Density to the output Exodus file
    {
      std::string stateName = "ACE_Density";
      entity                = Albany::StateStruct::QuadPoint;
      p = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE Density");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0))
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
    // Save ACE Heat Capacity to the output Exodus file
    {
      std::string stateName = "ACE_Heat_Capacity";
      entity                = Albany::StateStruct::QuadPoint;
      p = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE Heat Capacity");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0))
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
    // Save ACE Thermal Conductivity to the output Exodus file
    {
      std::string stateName = "ACE_Thermal_Cond";
      entity                = Albany::StateStruct::QuadPoint;
      p = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE Thermal Conductivity");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0))
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
    // Save ACE Thermal Inertia to the output Exodus file
    {
      std::string stateName = "ACE_Thermal_Inertia";
      entity                = Albany::StateStruct::QuadPoint;
      p = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE Thermal Inertia");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0))
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
    // Save ACE Water Saturation to the output Exodus file
    {
      std::string stateName = "ACE_Water_Saturation";
      entity                = Albany::StateStruct::QuadPoint;
      p = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE Water Saturation");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0))
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
    // Save ACE Porosity to the output Exodus file
    {
      std::string stateName = "ACE_Porosity";
      entity                = Albany::StateStruct::QuadPoint;
      p = state_mgr.registerStateVariable(stateName, dl_->qp_scalar, mesh_specs.ebName, true, &entity, "");
      p->set<std::string>("Field Name", "ACE Porosity");
      p->set("Field Layout", dl_->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((field_manager_choice == Albany::BUILD_RESID_FM) && (ev->evaluatedFields().size() > 0))
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
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
    p->set<string>("ACE Thermal Conductivity Gradient QP Variable Name", "ACE Thermal Conductivity Gradient QP");
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

    p->set<string>("ACE Thermal Conductivity QP Variable Name", "ACE Thermal Conductivity");
    p->set<string>("ACE Thermal Inertia QP Variable Name", "ACE Thermal Inertia");
    p->set<RCP<DataLayout>>("QP Scalar Data Layout", dl_->qp_scalar);
    p->set<string>("ACE Thermal Conductivity Gradient QP Variable Name", "ACE Thermal Conductivity Gradient QP");
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
