// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <MiniTensor.h>

#include <Phalanx_DataLayout.hpp>
#include <Sacado_ParameterRegistration.hpp>

#include "Albany_Application.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_ThyraUtils.hpp"
#include "SchwarzBC.hpp"

// Generic Template Code for Constructor and PostRegistrationSetup

namespace LCM {

template <typename EvalT, typename Traits>
SchwarzBC_Base<EvalT, Traits>::SchwarzBC_Base(Teuchos::ParameterList& p)
    : PHAL::DirichletBase<EvalT, Traits>(p),
      app_(p.get<Teuchos::RCP<Albany::Application>>("Application", Teuchos::null)),
      coupled_apps_(app_->getApplications()),
      coupled_app_name_(p.get<std::string>("Coupled Application", "SELF")),
      coupled_block_name_(p.get<std::string>("Coupled Block", "NONE"))
{
  std::string const& nodeset_name = this->nodeSetID;
  app_->setCoupledAppBlockNodeset(coupled_app_name_, coupled_block_name_, nodeset_name);
  std::string const& this_app_name      = app_->getAppName();
  auto const&        app_name_index_map = *(app_->getAppNameIndexMap());
  auto               it                 = app_name_index_map.find(this_app_name);
  ALBANY_EXPECT(it != app_name_index_map.end());
  auto const this_app_index = it->second;
  setThisAppIndex(this_app_index);
  it = app_name_index_map.find(coupled_app_name_);
  ALBANY_EXPECT(it != app_name_index_map.end());
  auto const coupled_app_index = it->second;
  setCoupledAppIndex(coupled_app_index);
}

template <typename EvalT, typename Traits>
template <typename T>
void
SchwarzBC_Base<EvalT, Traits>::computeBCs(size_t const ns_node, T& x_val, T& y_val, T& z_val)
{
  auto const coupled_app_index = getCoupledAppIndex();

  Albany::Application const& coupled_app = getApplication(coupled_app_index);

  Teuchos::RCP<Thyra_Vector const> coupled_solution = coupled_app.getX();

  if (coupled_solution == Teuchos::null) {
    x_val = 0.0;
    y_val = 0.0;
    z_val = 0.0;
    return;
  }

  auto const this_app_index = getThisAppIndex();

  Albany::Application const& this_app = getApplication(this_app_index);

  Teuchos::RCP<Albany::AbstractDiscretization> this_disc = this_app.getDiscretization();

  auto* this_stk_disc = static_cast<Albany::STKDiscretization*>(this_disc.get());

  Teuchos::RCP<Albany::AbstractDiscretization> coupled_disc = coupled_app.getDiscretization();

  auto* coupled_stk_disc = static_cast<Albany::STKDiscretization*>(coupled_disc.get());

  auto& coupled_gms = dynamic_cast<Albany::GenericSTKMeshStruct&>(*(coupled_stk_disc->getSTKMeshStruct()));

  auto const& coupled_ws_eb_names = coupled_disc->getWsEBNames();

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> coupled_mesh_specs = coupled_gms.getMeshSpecs();

  // Get cell topology of the application and block to which this node set
  // is coupled.
  std::string const& this_app_name      = this_app.getAppName();
  std::string const& coupled_app_name   = coupled_app.getAppName();
  std::string const  coupled_block_name = this_app.getCoupledBlockName(coupled_app_index);

  bool const use_block = coupled_block_name != "NONE";

  std::map<std::string, int> const& coupled_block_name_to_index = coupled_gms.getMeshSpecs()[0]->ebNameToIndex;

  auto       it            = coupled_block_name_to_index.find(coupled_block_name);
  bool const missing_block = it == coupled_block_name_to_index.end();

  if (use_block == true && missing_block == true) {
    std::cerr << "\nERROR: " << __PRETTY_FUNCTION__ << '\n';
    std::cerr << "Unknown coupled block: " << coupled_block_name << '\n';
    std::cerr << "Coupling application : " << this_app_name << '\n';
    std::cerr << "To application       : " << coupled_app_name << '\n';
    exit(1);
  }

  // When ignoring the block, set the index to zero to get defaults
  // corresponding to the first block.
  auto const coupled_block_index = use_block == true ? it->second : 0;

  CellTopologyData const coupled_cell_topology_data = coupled_mesh_specs[coupled_block_index]->ctd;

  shards::CellTopology coupled_cell_topology(&coupled_cell_topology_data);
  auto const           coupled_dimension  = coupled_cell_topology_data.dimension;
  auto const           coupled_node_count = coupled_cell_topology_data.node_count;

  std::string const& coupled_nodeset_name = this_app.getNodesetName(coupled_app_index);

  std::vector<double*> const& ns_coord = this_stk_disc->getNodeSetCoords().find(coupled_nodeset_name)->second;

  auto const& ws_elem_to_node_id = coupled_stk_disc->getWsElNodeID();

  std::vector<minitensor::Vector<double>> coupled_element_nodes(coupled_node_count);

  std::vector<minitensor::Vector<double>> coupled_element_solution(coupled_node_count);

  for (unsigned i = 0; i < coupled_node_count; ++i) {
    coupled_element_nodes[i].set_dimension(coupled_dimension);
    coupled_element_solution[i].set_dimension(coupled_dimension);
  }

  // This tolerance is used for geometric approximations. It will be used
  // to determine whether a node of this_app is inside an element of
  // coupled_app within that tolerance.
  double const tolerance = 5.0e-2;

  auto const parametric_dimension = coupled_dimension;
  auto const coupled_vertex_count = coupled_cell_topology_data.vertex_count;
  auto const coupled_element_type = minitensor::find_type(coupled_dimension, coupled_vertex_count);

  minitensor::Vector<double> lo(parametric_dimension, minitensor::Filler::ONES);
  minitensor::Vector<double> hi(parametric_dimension, minitensor::Filler::ONES);

  hi = hi * (1.0 + tolerance);

  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> basis;

  switch (coupled_element_type) {
    default: MT_ERROR_EXIT("Unknown element type"); break;

    case minitensor::ELEMENT::TETRAHEDRAL:
      basis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_TET_C1_FEM<PHX::Device>());
      lo    = -tolerance * lo;
      break;

    case minitensor::ELEMENT::HEXAHEDRAL:
      basis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_HEX_C1_FEM<PHX::Device>());
      lo    = -lo * (1.0 + tolerance);
      break;
  }

  double* const              coord = ns_coord[ns_node];
  minitensor::Vector<double> point;
  point.set_dimension(coupled_dimension);
  point.fill(coord);

  // Determine the element that contains this point.
  Teuchos::ArrayRCP<double> const& coupled_coordinates = coupled_stk_disc->getCoordinates();

  Teuchos::ArrayRCP<ST const> coupled_solution_view = Albany::getLocalData(coupled_solution);

  Teuchos::RCP<Thyra_VectorSpace const> coupled_overlap_node_vs = coupled_stk_disc->getOverlapNodeVectorSpace();

  // We do this element by element
  auto const number_cells = 1;

  // We do this point by point
  auto const number_points = 1;

  // Container for the parametric coordinates
  Kokkos::DynRankView<RealType, PHX::Device> parametric_point("par_point", number_cells, number_points, parametric_dimension);

  for (unsigned j = 0; j < parametric_dimension; ++j) {
    parametric_point(0, 0, j) = 0.0;
  }

  // Container for the physical point
  Kokkos::DynRankView<RealType, PHX::Device> physical_coordinates("phys_point", number_cells, number_points, coupled_dimension);

  for (unsigned i = 0; i < coupled_dimension; ++i) {
    physical_coordinates(0, 0, i) = point(i);
  }

  // Container for the physical nodal coordinates
  Kokkos::DynRankView<RealType, PHX::Device> nodal_coordinates("coords", number_cells, coupled_node_count, coupled_dimension);

  bool found = false;

  auto coupled_ov_node_vs_indexer = Albany::createGlobalLocalIndexer(coupled_overlap_node_vs);
  for (auto workset = 0; workset < ws_elem_to_node_id.size(); ++workset) {
    std::string const& coupled_element_block = coupled_ws_eb_names[workset];

    bool const block_names_differ = coupled_element_block != coupled_block_name;
    if (use_block == true && block_names_differ == true) continue;
    auto const elements_per_workset = ws_elem_to_node_id[workset].size();

    for (auto element = 0; element < elements_per_workset; ++element) {
      for (unsigned node = 0; node < coupled_node_count; ++node) {
        auto const global_node_id = ws_elem_to_node_id[workset][element][node];

        auto const local_node_id = coupled_ov_node_vs_indexer->getLocalElement(global_node_id);

        double* const pcoord = &(coupled_coordinates[coupled_dimension * local_node_id]);

        coupled_element_nodes[node].fill(pcoord);

        for (unsigned i = 0; i < coupled_dimension; ++i) {
          coupled_element_solution[node](i) = coupled_solution_view[coupled_dimension * local_node_id + i];
        }  // dimension loop

      }  // node loop

      for (unsigned i = 0; i < coupled_node_count; ++i) {
        for (unsigned j = 0; j < coupled_dimension; ++j) {
          nodal_coordinates(0, i, j) = coupled_element_nodes[i](j);
        }
      }

      // Get parametric coordinates
      Intrepid2::CellTools<PHX::Device>::mapToReferenceFrame(parametric_point, physical_coordinates, nodal_coordinates, coupled_cell_topology);

      bool in_element = true;

      for (unsigned i = 0; i < parametric_dimension; ++i) {
        auto const xi = parametric_point(0, 0, i);
        in_element    = in_element && lo(i) <= xi && xi <= hi(i);
      }

      if (in_element == true) {
        found = true;
        break;
      }

    }  // element loop

    if (found == true) {
      break;
    }

  }  // workset loop

  ALBANY_EXPECT(found == true);

  // Evaluate shape functions at parametric point.
  Kokkos::DynRankView<RealType, PHX::Device> basis_values("basis", coupled_node_count, number_points);

  // Another container for the parametric coordinates. Needed because above
  // it is required that parametric_points has rank 3 for mapToReferenceFrame
  // but here basis->getValues requires a rank 2 view :(
  Kokkos::DynRankView<RealType, PHX::Device> pp_reduced("par_point", number_points, parametric_dimension);

  for (unsigned j = 0; j < parametric_dimension; ++j) {
    pp_reduced(0, j) = parametric_point(0, 0, j);
  }
  basis->getValues(basis_values, pp_reduced, Intrepid2::OPERATOR_VALUE);

  // Evaluate solution at parametric point using values of shape
  // functions just computed.
  minitensor::Vector<double> value(coupled_dimension, minitensor::Filler::ZEROS);

  for (unsigned i = 0; i < coupled_node_count; ++i) {
    value += basis_values(i, 0) * coupled_element_solution[i];
  }

  x_val = value(0);
  y_val = value(1);
  z_val = value(2);
}

#if defined(ALBANY_DTK)
template <typename EvalT, typename Traits>
Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>
SchwarzBC_Base<EvalT, Traits>::computeBCsDTK()
{
  auto const                 this_app_index    = getThisAppIndex();
  auto const                 coupled_app_index = getCoupledAppIndex();
  Albany::Application const& this_app          = getApplication(this_app_index);
  Albany::Application const& coupled_app       = getApplication(coupled_app_index);

  // neq should be the same for this_app and coupled_app.
  ALBANY_EXPECT(this_app.getNumEquations() == coupled_app.getNumEquations());

  // Get number of equations from this_app
  int neq = this_app.getNumEquations();

  // this_disc = target mesh
  Teuchos::RCP<Albany::AbstractDiscretization> this_disc = this_app.getDiscretization();

  auto* this_stk_disc = static_cast<Albany::STKDiscretization*>(this_disc.get());

  // coupled_disc = source mesh
  Teuchos::RCP<Albany::AbstractDiscretization> coupled_disc = coupled_app.getDiscretization();

  auto* coupled_stk_disc = static_cast<Albany::STKDiscretization*>(coupled_disc.get());

  // Source Mesh
  Teuchos::RCP<Albany::AbstractSTKMeshStruct> const coupled_stk_mesh_struct = coupled_stk_disc->getSTKMeshStruct();

  // get pointer to metadata from coupled_stk_disc
  Teuchos::RCP<stk::mesh::MetaData const> const coupled_meta_data = Teuchos::rcpFromRef(coupled_stk_disc->getSTKMetaData());

  // Get coupled_app parameter list
  Teuchos::RCP<Teuchos::ParameterList const> coupled_app_params = coupled_app.getAppPL();

  // Get discretization sublist from coupled_app parameter list
  Teuchos::ParameterList dtk_params = coupled_app_params->sublist("DataTransferKit");

  // Get solution name from Discretization sublist
  std::string map_name = dtk_params.get("Map Type", "Consistent Interpolation");

  Albany::AbstractSTKFieldContainer::STKFieldType* coupled_field =
      Teuchos::rcp_dynamic_cast<Albany::OrdinarySTKFieldContainer<true>>(coupled_stk_disc->getSTKMeshStruct()->getFieldContainer())->getSolutionField();

  stk::mesh::Selector coupled_stk_selector = stk::mesh::Selector(coupled_meta_data->universal_part());

  Teuchos::RCP<stk::mesh::BulkData> coupled_bulk_data = Teuchos::rcpFromRef(coupled_field->get_mesh());

  // Target Mesh

  // get pointer to metadata from this_stk_disc
  Teuchos::RCP<stk::mesh::MetaData const> this_meta_data = Teuchos::rcpFromRef(this_stk_disc->getSTKMetaData());

  Albany::AbstractSTKFieldContainer::STKFieldType* this_field =
      Teuchos::rcp_dynamic_cast<Albany::OrdinarySTKFieldContainer<true>>(this_stk_disc->getSTKMeshStruct()->getFieldContainer())->getSolutionFieldDTK();

  // Get the part corresponding to this nodeset.
  std::string const& nodeset_name = this->nodeSetID;

  stk::mesh::Part* this_part = this_meta_data->get_part(nodeset_name);

  Teuchos::RCP<stk::mesh::BulkData> this_bulk_data = Teuchos::rcpFromRef(this_field->get_mesh());

  // Solution Transfer Setup

  // Create a manager for the source part elements.
  DataTransferKit::STKMeshManager coupled_manager(coupled_bulk_data, coupled_stk_selector);

  // Create a manager for the target part nodes.
  stk::mesh::Selector this_stk_selector(*this_part);

  DataTransferKit::STKMeshManager this_manager(this_bulk_data, this_stk_selector);

  // Create a solution vector for the source.
  Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>> coupled_vector =
      coupled_manager.createFieldMultiVector<Albany::AbstractSTKFieldContainer::STKFieldType>(Teuchos::ptr(coupled_field), neq);

  // Create a solution vector for the target.
  Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>> this_vector =
      this_manager.createFieldMultiVector<Albany::AbstractSTKFieldContainer::STKFieldType>(Teuchos::ptr(this_field), neq);

  // Solution transfer

  DataTransferKit::MapOperatorFactory op_factory;

  Teuchos::RCP<DataTransferKit::MapOperator> map_op = op_factory.create(coupled_vector->getMap(), this_vector->getMap(), dtk_params);

  // Setup the map operator. This creates the underlying linear operators.
  map_op->setup(coupled_manager.functionSpace(), this_manager.functionSpace());

  // Apply the map operator. This interpolates the data from one STK field
  // to the other.
  map_op->apply(*coupled_vector, *this_vector);

  return this_vector;
}
#endif  // ALBANY_DTK

// Fill residual, used in both residual and Jacobian
template <typename SchwarzBC, typename Traits>
void
fillResidual(SchwarzBC& sbc, typename Traits::EvalData workset)
{
  // Solution
  Teuchos::RCP<Thyra_Vector const> x            = workset.x;
  Teuchos::ArrayRCP<ST const>      x_const_view = Albany::getLocalData(x);

  // Residual
  Teuchos::RCP<Thyra_Vector>           f      = workset.f;
  Teuchos::ArrayRCP<ST>                f_view = Albany::getNonconstLocalData(f);
  std::vector<std::vector<int>> const& ns_dof = workset.nodeSets->find(sbc.nodeSetID)->second;

  auto const ns_number_nodes = ns_dof.size();

#if defined(ALBANY_DTK)

  Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>> const schwarz_bcs = sbc.computeBCsDTK();

  Teuchos::RCP<Teuchos::Comm<int> const> commT = schwarz_bcs->getMap()->getComm();

  Teuchos::ArrayRCP<ST const> schwarz_bcs_const_view_x = schwarz_bcs->getData(0);

  Teuchos::ArrayRCP<ST const> schwarz_bcs_const_view_y = schwarz_bcs->getData(1);

  Teuchos::ArrayRCP<ST const> schwarz_bcs_const_view_z = schwarz_bcs->getData(2);

  for (auto ns_node = 0; ns_node < ns_number_nodes; ++ns_node) {
    auto const x_dof = ns_dof[ns_node][0];
    auto const y_dof = ns_dof[ns_node][1];
    auto const z_dof = ns_dof[ns_node][2];
    auto const dof   = x_dof / 3;

    std::set<int> const& fixed_dofs = workset.fixed_dofs_;

    if (fixed_dofs.find(x_dof) == fixed_dofs.end()) {
      f_view[x_dof] = x_const_view[x_dof] - schwarz_bcs_const_view_x[dof];
    }
    if (fixed_dofs.find(y_dof) == fixed_dofs.end()) {
      f_view[y_dof] = x_const_view[y_dof] - schwarz_bcs_const_view_y[dof];
    }
    if (fixed_dofs.find(z_dof) == fixed_dofs.end()) {
      f_view[z_dof] = x_const_view[z_dof] - schwarz_bcs_const_view_z[dof];
    }
  }
#else   // ALBANY_DTK
  for (auto ns_node = 0; ns_node < ns_number_nodes; ++ns_node) {
    ST x_val, y_val, z_val;
    sbc.computeBCs(ns_node, x_val, y_val, z_val);
    auto const           x_dof      = ns_dof[ns_node][0];
    auto const           y_dof      = ns_dof[ns_node][1];
    auto const           z_dof      = ns_dof[ns_node][2];
    std::set<int> const& fixed_dofs = workset.fixed_dofs_;

    if (fixed_dofs.find(x_dof) == fixed_dofs.end()) {
      f_view[x_dof] = x_const_view[x_dof] - x_val;
    }
    if (fixed_dofs.find(y_dof) == fixed_dofs.end()) {
      f_view[y_dof] = x_const_view[y_dof] - y_val;
    }
    if (fixed_dofs.find(z_dof) == fixed_dofs.end()) {
      f_view[z_dof] = x_const_view[z_dof] - z_val;
    }

  }  // node in node set loop
#endif  // ALBANY_DTK
  return;
}

// Specialization: Residual
template <typename Traits>
SchwarzBC<PHAL::AlbanyTraits::Residual, Traits>::SchwarzBC(Teuchos::ParameterList& p) : SchwarzBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

template <typename Traits>
void
SchwarzBC<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  fillResidual<SchwarzBC<PHAL::AlbanyTraits::Residual, Traits>, Traits>(*this, workset);
  return;
}

// Specialization: Jacobian
template <typename Traits>
SchwarzBC<PHAL::AlbanyTraits::Jacobian, Traits>::SchwarzBC(Teuchos::ParameterList& p) : SchwarzBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}

template <typename Traits>
void
SchwarzBC<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Thyra_Vector const> x   = workset.x;
  Teuchos::RCP<Thyra_Vector>       f   = workset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = workset.Jac;

  Teuchos::ArrayRCP<ST const> x_const_view = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_view;

  RealType const j_coeff = workset.j_coeff;

  std::vector<std::vector<int>> const& ns_nodes = workset.nodeSets->find(this->nodeSetID)->second;

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);

  value[0] = j_coeff;

  Teuchos::Array<ST> matrixEntries;
  Teuchos::Array<LO> matrixIndices;

  bool const fill_residual = (f != Teuchos::null);

  if (fill_residual == true) {
    f_view = Albany::getNonconstLocalData(f);
  }

  for (auto ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    auto const x_dof = ns_nodes[ns_node][0];
    auto const y_dof = ns_nodes[ns_node][1];
    auto const z_dof = ns_nodes[ns_node][2];

    std::set<int> const& fixed_dofs = workset.fixed_dofs_;

    if (fixed_dofs.find(x_dof) == fixed_dofs.end()) {
      // replace jac values for the X dof
      Albany::getLocalRowValues(jac, x_dof, matrixIndices, matrixEntries);
      for (auto& val : matrixEntries) {
        val = 0.0;
      }
      Albany::setLocalRowValues(jac, x_dof, matrixIndices(), matrixEntries());
      index[0] = x_dof;
      Albany::setLocalRowValues(jac, x_dof, index(), value());
    }

    if (fixed_dofs.find(y_dof) == fixed_dofs.end()) {
      // replace jac values for the y dof
      Albany::getLocalRowValues(jac, y_dof, matrixIndices, matrixEntries);
      for (auto& val : matrixEntries) {
        val = 0.0;
      }
      Albany::setLocalRowValues(jac, y_dof, matrixIndices(), matrixEntries());
      index[0] = y_dof;
      Albany::setLocalRowValues(jac, y_dof, index(), value());
    }

    if (fixed_dofs.find(z_dof) == fixed_dofs.end()) {
      // replace jac values for the z dof
      Albany::getLocalRowValues(jac, z_dof, matrixIndices, matrixEntries);
      for (auto& val : matrixEntries) {
        val = 0.0;
      }
      Albany::setLocalRowValues(jac, z_dof, matrixIndices(), matrixEntries());
      index[0] = z_dof;
      Albany::setLocalRowValues(jac, z_dof, index(), value());
    }
  }

  if (fill_residual == true) {
    fillResidual<SchwarzBC<PHAL::AlbanyTraits::Jacobian, Traits>, Traits>(*this, workset);
  }
}

}  // namespace LCM
