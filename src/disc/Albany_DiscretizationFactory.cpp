// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_DiscretizationFactory.hpp"

#include "Albany_AsciiSTKMesh2D.hpp"
#include "Albany_AsciiSTKMeshStruct.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_GmshSTKMeshStruct.hpp"
#include "Albany_IossSTKMeshStruct.hpp"
#include "Albany_Macros.hpp"
#include "Albany_STK3DPointStruct.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_SideSetSTKMeshStruct.hpp"
#include "Albany_TmplSTKMeshStruct.hpp"
#include "Topology_Utils.hpp"

Albany::DiscretizationFactory::DiscretizationFactory(
    const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
    const Teuchos::RCP<Teuchos_Comm const>&     commT_)
    : commT(commT_)
{
  discParams = Teuchos::sublist(topLevelParams, "Discretization", true);

  if (topLevelParams->isSublist("Piro")) {
    piroParams = Teuchos::sublist(topLevelParams, "Piro", true);
  }

  if (topLevelParams->isSublist("Problem")) {
    Teuchos::RCP<Teuchos::ParameterList> problemParams =
        Teuchos::sublist(topLevelParams, "Problem", true);

    if (problemParams->isSublist("Adaptation")) {
      adaptParams = Teuchos::sublist(problemParams, "Adaptation", true);
    }
    if (problemParams->isSublist("Catalyst")) {
      catalystParams = Teuchos::sublist(problemParams, "Catalyst", true);
    }
  }
}

namespace {

void
createInterfaceParts(
    Teuchos::RCP<Teuchos::ParameterList> const& adapt_params,
    Teuchos::RCP<Albany::AbstractMeshStruct>&   mesh_struct)
{
  // Top mod uses BGL
  bool const do_adaptation = adapt_params.is_null() == false;

  if (do_adaptation == false) return;

  std::string const& adaptation_method_name =
      adapt_params->get<std::string>("Method");

  bool const is_topology_modification = adaptation_method_name == "Topmod";

  if (is_topology_modification == false) return;

  std::string const& bulk_part_name =
      adapt_params->get<std::string>("Bulk Block Name");

  Albany::AbstractSTKMeshStruct& stk_mesh_struct =
      dynamic_cast<Albany::AbstractSTKMeshStruct&>(*mesh_struct);

  stk::mesh::MetaData& meta_data = *(stk_mesh_struct.metaData);

  stk::mesh::Part& bulk_part = *(meta_data.get_part(bulk_part_name));

  stk::topology               stk_topo_data = meta_data.get_topology(bulk_part);
  shards::CellTopology const& bulk_cell_topology =
      stk::mesh::get_cell_topology(stk_topo_data);

  std::string const& interface_part_name(
      adapt_params->get<std::string>("Interface Block Name"));

  shards::CellTopology const interface_cell_topology =
      LCM::interfaceCellTopogyFromBulkCellTopogy(bulk_cell_topology);

  stk::mesh::EntityRank const interface_dimension =
      static_cast<stk::mesh::EntityRank>(
          interface_cell_topology.getDimension());

  stk::mesh::Part& interface_part =
      meta_data.declare_part(interface_part_name, interface_dimension);

  stk::topology stk_interface_topo =
      stk::mesh::get_topology(interface_cell_topology);
  stk::mesh::set_topology(interface_part, stk_interface_topo);

  stk::io::put_io_part_attribute(interface_part);

  // Augment the MeshSpecsStruct array with one additional entry for
  // the interface block. Essentially copy the last entry from the array
  // and modify some of its fields as needed.
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>& mesh_specs_struct =
      stk_mesh_struct.getMeshSpecs();

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>::size_type
      number_blocks = mesh_specs_struct.size();

  Albany::MeshSpecsStruct& last_mesh_specs_struct =
      *(mesh_specs_struct[number_blocks - 1]);

  CellTopologyData const& interface_cell_topology_data =
      *(interface_cell_topology.getCellTopologyData());

  int const dimension = interface_cell_topology.getDimension();

  int const cubature_degree = last_mesh_specs_struct.cubatureDegree;

  std::vector<std::string> node_sets, side_sets;

  int const workset_size = last_mesh_specs_struct.worksetSize;

  std::string const& element_block_name = interface_part_name;

  std::map<std::string, int>& eb_name_to_index_map =
      last_mesh_specs_struct.ebNameToIndex;

  // Add entry to the map for this block
  eb_name_to_index_map.insert(
      std::make_pair(element_block_name, number_blocks));

  bool const is_interleaved = last_mesh_specs_struct.interleavedOrdering;

  Intrepid2::EPolyType const cubature_rule =
      last_mesh_specs_struct.cubatureRule;

  mesh_specs_struct.resize(number_blocks + 1);

  mesh_specs_struct[number_blocks] = Teuchos::rcp(new Albany::MeshSpecsStruct(
      interface_cell_topology_data,
      dimension,
      cubature_degree,
      node_sets,
      side_sets,
      workset_size,
      element_block_name,
      eb_name_to_index_map,
      is_interleaved,
      number_blocks > 1,
      cubature_rule));
  return;
}

}  // anonymous namespace

Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>
Albany::DiscretizationFactory::createMeshSpecs()
{
  // First, create the mesh struct
  meshStruct = createMeshStruct(discParams, adaptParams, commT);

  // Add an interface block. For now relies on STK, so we force a cast that
  // will fail if the underlying meshStruct is not based on STK.
  createInterfaceParts(adaptParams, meshStruct);
  return meshStruct->getMeshSpecs();
}

Teuchos::RCP<Albany::AbstractMeshStruct>
Albany::DiscretizationFactory::createMeshStruct(
    Teuchos::RCP<Teuchos::ParameterList> disc_params,
    Teuchos::RCP<Teuchos::ParameterList> adapt_params,
    Teuchos::RCP<Teuchos_Comm const>     comm)
{
  std::string& method = disc_params->get("Method", "STK1D");
  if (method == "STK1D") {
    return Teuchos::rcp(
        new Albany::TmplSTKMeshStruct<1>(disc_params, adapt_params, comm));
  } else if (method == "STK0D") {
    return Teuchos::rcp(
        new Albany::TmplSTKMeshStruct<0>(disc_params, adapt_params, comm));
  } else if (method == "STK2D") {
    return Teuchos::rcp(
        new Albany::TmplSTKMeshStruct<2>(disc_params, adapt_params, comm));
  } else if (method == "STK3D") {
    return Teuchos::rcp(
        new Albany::TmplSTKMeshStruct<3>(disc_params, adapt_params, comm));
  } else if (method == "STK3DPoint") {
    return Teuchos::rcp(new Albany::STK3DPointStruct(disc_params, comm));
  } else if (method == "Ioss" || method == "Exodus" || method == "Pamgen") {
    return Teuchos::rcp(
        new Albany::IossSTKMeshStruct(disc_params, adapt_params, comm));
  } else if (method == "Ascii") {
    return Teuchos::rcp(new Albany::AsciiSTKMeshStruct(disc_params, comm));
  } else if (method == "Ascii2D") {
    return Teuchos::rcp(new Albany::AsciiSTKMesh2D(disc_params, comm));
  } else if (method == "Hacky Ascii2D") {
    // FixME very hacky! needed for printing 2d mesh
    Teuchos::RCP<Albany::GenericSTKMeshStruct> meshStruct2D;
    meshStruct2D = Teuchos::rcp(new Albany::AsciiSTKMesh2D(disc_params, comm));
    Teuchos::RCP<Albany::StateInfoStruct> sis =
        Teuchos::rcp(new Albany::StateInfoStruct);
    Albany::AbstractFieldContainer::FieldContainerRequirements req;
    int                                                        neq = 2;
    meshStruct2D->setFieldAndBulkData(
        comm,
        disc_params,
        neq,
        req,
        sis,
        meshStruct2D->getMeshSpecs()[0]->worksetSize);
    Ioss::Init::Initializer                io;
    Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data =
        Teuchos::rcp(new stk::io::StkMeshIoBroker(MPI_COMM_WORLD));
    mesh_data->set_bulk_data(*meshStruct2D->bulkData);
    std::string const& output_filename =
        disc_params->get("Exodus Output File Name", "ice_mesh.2d.exo");
    size_t idx =
        mesh_data->create_output_mesh(output_filename, stk::io::WRITE_RESULTS);
    mesh_data->process_output_request(idx, 0.0);
  } else if (method == "Gmsh") {
    return Teuchos::rcp(new Albany::GmshSTKMeshStruct(disc_params, comm));
  } else

    ALBANY_ABORT(
        std::endl
        << "Error!  Unknown discretization method in "
           "DiscretizationFactory: "
        << method << "!" << std::endl
        << "Supplied parameter list is " << std::endl
        << *disc_params
        << "\nValid Methods are: STK1D, STK2D, STK3D, STK3DPoint, Ioss, "
        << " Exodus, Sim, Ascii,"
        << " Ascii2D, Extruded" << std::endl);
  return Teuchos::null;
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretization(
    unsigned int                                              neq,
    const Teuchos::RCP<Albany::StateInfoStruct>&              sis,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    const Teuchos::RCP<Albany::RigidBodyModes>&               rigidBodyModes)
{
  return createDiscretization(
      neq,
      empty_side_set_equations,
      sis,
      empty_side_set_sis,
      req,
      empty_side_set_req,
      rigidBodyModes);
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretization(
    unsigned int                                   neq,
    std::map<int, std::vector<std::string>> const& sideSetEquations,
    const Teuchos::RCP<Albany::StateInfoStruct>&   sis,
    std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> const&
                                                              side_set_sis,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    std::map<
        std::string,
        AbstractFieldContainer::FieldContainerRequirements> const& side_set_req,
    const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes)
{
  ALBANY_PANIC(
      meshStruct == Teuchos::null,
      "meshStruct accessed, but it has not been constructed" << std::endl);

  setupInternalMeshStruct(neq, sis, side_set_sis, req, side_set_req);
  Teuchos::RCP<Albany::AbstractDiscretization> result =
      createDiscretizationFromInternalMeshStruct(
          sideSetEquations, rigidBodyModes);

  // Wrap the discretization in the catalyst decorator if needed.

  return result;
}

Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>
Albany::DiscretizationFactory::createMeshSpecs(
    Teuchos::RCP<Albany::AbstractMeshStruct> mesh)
{
  meshStruct = mesh;
  return meshStruct->getMeshSpecs();
}

void
Albany::DiscretizationFactory::setupInternalMeshStruct(
    unsigned int                                              neq,
    const Teuchos::RCP<Albany::StateInfoStruct>&              sis,
    const AbstractFieldContainer::FieldContainerRequirements& req)
{
  setupInternalMeshStruct(
      neq, sis, empty_side_set_sis, req, empty_side_set_req);
}

void
Albany::DiscretizationFactory::setupInternalMeshStruct(
    unsigned int                                 neq,
    const Teuchos::RCP<Albany::StateInfoStruct>& sis,
    std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> const&
                                                              side_set_sis,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    std::map<
        std::string,
        AbstractFieldContainer::FieldContainerRequirements> const& side_set_req)
{
  meshStruct->setFieldAndBulkData(
      commT,
      discParams,
      neq,
      req,
      sis,
      meshStruct->getMeshSpecs()[0]->worksetSize,
      side_set_sis,
      side_set_req);
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretizationFromInternalMeshStruct(
    const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes)
{
  return createDiscretizationFromInternalMeshStruct(
      empty_side_set_equations, rigidBodyModes);
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretizationFromInternalMeshStruct(
    std::map<int, std::vector<std::string>> const& sideSetEquations,
    const Teuchos::RCP<Albany::RigidBodyModes>&    rigidBodyModes)
{
  if (!piroParams.is_null() && !rigidBodyModes.is_null()) {
    rigidBodyModes->setPiroPL(piroParams);
  }
  auto ms =
      Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(meshStruct);
  auto disc = Teuchos::rcp(new Albany::STKDiscretization(
      discParams, ms, commT, rigidBodyModes, sideSetEquations));
  disc->updateMesh();
  return disc;
}

/* This function overwrite previous discretization parameter list */
void
Albany::DiscretizationFactory::setDiscretizationParameters(
    Teuchos::RCP<Teuchos::ParameterList> disc_params)
{
  discParams = disc_params;
}
