// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include "Topology.hpp"

#include <Albany_CommUtils.hpp>
#include <Albany_STKNodeSharing.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/SkinMesh.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>

#include "Subgraph.hpp"
#include "Topology_FailureCriterion.hpp"
#include "Topology_Utils.hpp"

namespace LCM {

// Default constructor
Topology::Topology() : discretization_(Teuchos::null), stk_mesh_struct_(Teuchos::null), failure_criterion_(Teuchos::null), output_type_(UNIDIRECTIONAL_UNILEVEL)
{
  return;
}

// Constructor with input and output files.
Topology::Topology(std::string const& input_file, std::string const& output_file)
    : discretization_(Teuchos::null), stk_mesh_struct_(Teuchos::null), failure_criterion_(Teuchos::null), output_type_(UNIDIRECTIONAL_UNILEVEL)
{
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList("params"));

  // Create discretization object
  Teuchos::RCP<Teuchos::ParameterList> disc_params = Teuchos::sublist(params, "Discretization");

  // Set Method to Exodus and set input file name
  disc_params->set<std::string>("Method", "Exodus");
  disc_params->set<std::string>("Exodus Input File Name", input_file);
  disc_params->set<std::string>("Exodus Output File Name", output_file);
  disc_params->set<int>("Number Of Time Derivatives", 0);

  Teuchos::RCP<Teuchos::ParameterList> problem_params = Teuchos::sublist(params, "Problem");

  // Make adaptation list to force Albany::DiscretizationFactory
  // to create interface block.
  Teuchos::RCP<Teuchos::ParameterList> adapt_params = Teuchos::sublist(problem_params, "Adaptation");

  Teuchos::RCP<Teuchos_Comm const> communicator = Albany::getDefaultComm();
  adapt_params->set<std::string>("Method", "Topmod");
  std::string const bulk_block_name = "Bulk Element";
  adapt_params->set<std::string>("Bulk Block Name", bulk_block_name);
  std::string const interface_block_name = "Surface Element";
  adapt_params->set<std::string>("Interface Block Name", interface_block_name);
  set_bulk_block_name(bulk_block_name);
  set_interface_block_name(interface_block_name);
  Albany::DiscretizationFactory disc_factory(params, communicator);

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> mesh_specs = disc_factory.createMeshSpecs();

  Teuchos::RCP<Albany::StateInfoStruct> state_info = Teuchos::rcp(new Albany::StateInfoStruct());

  // The default fields
  Albany::AbstractFieldContainer::FieldContainerRequirements req;

  Teuchos::RCP<Albany::AbstractDiscretization> const& abstract_disc = disc_factory.createDiscretization(3, state_info, req);

  set_discretization(abstract_disc);

  Albany::STKDiscretization& stk_disc = static_cast<Albany::STKDiscretization&>(*abstract_disc);

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> stk_mesh_struct = stk_disc.getSTKMeshStruct();

  set_stk_mesh_struct(stk_mesh_struct);
  Topology::graphInitialization();
}

void
Topology::computeExtrema()
{
  auto                    stk_mesh_struct = get_stk_mesh_struct();
  auto&                   coord_field     = *(stk_mesh_struct->getCoordinatesField());
  auto&                   bulk_data       = get_bulk_data();
  auto const              node_rank       = stk::topology::NODE_RANK;
  stk::mesh::EntityVector nodes;
  stk::mesh::get_entities(bulk_data, node_rank, nodes);
  auto  first_node = nodes[0];
  auto* pcfn       = stk::mesh::field_data(coord_field, first_node);
  auto  xmin       = pcfn[0];
  auto  ymin       = pcfn[1];
  auto  zmin       = pcfn[2];
  auto  xmax       = xmin;
  auto  ymax       = ymin;
  auto  zmax       = zmin;

  for (auto node : nodes) {
    auto*      pc = stk::mesh::field_data(coord_field, node);
    auto const x  = pc[0];
    auto const y  = pc[1];
    auto const z  = pc[2];
    xmin          = std::min(x, xmin);
    ymin          = std::min(y, ymin);
    zmin          = std::min(z, zmin);
    xmax          = std::max(x, xmax);
    ymax          = std::max(y, ymax);
    zmax          = std::max(z, zmax);
  }
  auto global_xmin = xmin;
  auto global_ymin = ymin;
  auto global_zmin = zmin;
  auto global_xmax = xmax;
  auto global_ymax = ymax;
  auto global_zmax = zmax;
  auto comm        = static_cast<stk::ParallelMachine>(MPI_COMM_WORLD);
  stk::all_reduce_min(comm, &xmin, &global_xmin, 1);
  stk::all_reduce_min(comm, &ymin, &global_ymin, 1);
  stk::all_reduce_min(comm, &zmin, &global_zmin, 1);
  stk::all_reduce_max(comm, &xmax, &global_xmax, 1);
  stk::all_reduce_max(comm, &ymax, &global_ymax, 1);
  stk::all_reduce_max(comm, &zmax, &global_zmax, 1);
  xm_ = xmin;
  ym_ = ymin;
  zm_ = zmin;
  xp_ = xmax;
  yp_ = ymax;
  zp_ = zmax;
}

// Construct by using given discretization.
Topology::Topology(Teuchos::RCP<Albany::AbstractDiscretization>& abstract_disc, std::string const& bulk_block_name, std::string const& interface_block_name)
    : discretization_(Teuchos::null), stk_mesh_struct_(Teuchos::null), failure_criterion_(Teuchos::null), output_type_(UNIDIRECTIONAL_UNILEVEL)
{
  auto& stk_disc        = static_cast<Albany::STKDiscretization&>(*abstract_disc);
  auto  stk_mesh_struct = stk_disc.getSTKMeshStruct();
  set_discretization(abstract_disc);
  set_stk_mesh_struct(stk_mesh_struct);
  set_bulk_block_name(bulk_block_name);
  set_interface_block_name(interface_block_name);
  computeExtrema();
  graphInitialization();
}

stk::mesh::EntityId
Topology::get_entity_id(stk::mesh::Entity const entity)
{
  size_t const                space_dimension = get_space_dimension();
  int const                   parallel_rank   = get_bulk_data().parallel_rank();
  stk::mesh::EntityRank const rank            = get_bulk_data().entity_rank(entity);
  stk::mesh::EntityId const   high_id         = get_bulk_data().identifier(entity);
  stk::mesh::EntityId const   low_id          = low_id_from_high_id(space_dimension, parallel_rank, rank, high_id);
  return low_id;
}

// Check fracture criterion
bool
Topology::checkOpen(stk::mesh::Entity e)
{
  return failure_criterion_->check(get_bulk_data(), e);
}

// Initialize failure state field
// It exists for all entities
void
Topology::initializeFailureState()
{
  auto&      bulk_data = get_bulk_data();
  auto const node_rank = stk::topology::NODE_RANK;
  auto const cell_rank = stk::topology::ELEMENT_RANK;

  for (auto rank = node_rank; rank <= cell_rank; ++rank) {
    stk::mesh::EntityVector entities;
    stk::mesh::get_entities(bulk_data, rank, entities);
    for (auto entity : entities) {
      set_failure_state(entity, INTACT);
    }
  }
}

// Initialize cell failure state field
void
Topology::initializeCellFailureState()
{
  auto&                   bulk_data = get_bulk_data();
  auto const              cell_rank = stk::topology::ELEMENT_RANK;
  stk::mesh::EntityVector cells;
  stk::mesh::get_entities(bulk_data, cell_rank, cells);
  for (auto cell : cells) {
    set_failure_state(cell, INTACT);
  }
}

// Set boundary indicators
void
Topology::setCellBoundaryIndicator()
{
  auto&                   bulk_data = get_bulk_data();
  auto const              cell_rank = stk::topology::ELEMENT_RANK;
  stk::mesh::EntityVector cells;
  stk::mesh::get_entities(bulk_data, cell_rank, cells);

  for (auto cell : cells) {
    auto const bi = is_erodible_cell(cell) == true ? ERODIBLE : (is_boundary_cell(cell) == true ? EXTERIOR : INTERIOR);
    set_cell_boundary_indicator(cell, bi);
  }
}

void
Topology::setFaceBoundaryIndicator()
{
  auto&                   bulk_data = get_bulk_data();
  auto const              face_rank = stk::topology::FACE_RANK;
  stk::mesh::EntityVector faces;
  stk::mesh::get_entities(bulk_data, face_rank, faces);

  for (auto face : faces) {
    auto const bi = is_erodible_face(face) == true ? ERODIBLE : (is_boundary_face(face) == true ? EXTERIOR : INTERIOR);
    set_face_boundary_indicator(face, bi);
  }
}

void
Topology::setEdgeBoundaryIndicator()
{
  auto&                   bulk_data = get_bulk_data();
  auto const              edge_rank = stk::topology::EDGE_RANK;
  stk::mesh::EntityVector edges;
  stk::mesh::get_entities(bulk_data, edge_rank, edges);

  for (auto edge : edges) {
    auto const bi = is_erodible_edge(edge) == true ? ERODIBLE : (is_boundary_edge(edge) == true ? EXTERIOR : INTERIOR);
    set_edge_boundary_indicator(edge, bi);
  }
}

void
Topology::setNodeBoundaryIndicator()
{
  auto&                   bulk_data = get_bulk_data();
  auto const              node_rank = stk::topology::NODE_RANK;
  stk::mesh::EntityVector nodes;
  stk::mesh::get_entities(bulk_data, node_rank, nodes);

#if defined(DEBUG)
  auto& stk_mesh_struct = *(get_stk_mesh_struct());
  auto& coord_field     = *(stk_mesh_struct.getCoordinatesField());
  std::cout << "*** BUILD BOUNDARY INDICATOR ***\n";
#endif

  for (auto node : nodes) {
    auto const bi = is_erodible_node(node) == true ? ERODIBLE : (is_boundary_node(node) == true ? EXTERIOR : INTERIOR);
    set_node_boundary_indicator(node, bi);
#if defined(DEBUG)
    double*    pc = stk::mesh::field_data(coord_field, node);
    auto const x  = pc[0];
    auto const y  = pc[1];
    auto const z  = pc[2];
    std::cout << "NODE : " << std::setw(4) << node << ", BI : " << std::setw(2) << bi << ", ";
    std::cout << "X : " << std::setw(24) << std::setprecision(16) << x << ", ";
    std::cout << "Y : " << std::setw(24) << std::setprecision(16) << y << ", ";
    std::cout << "Z : " << std::setw(24) << std::setprecision(16) << z << "\n";
#endif
  }
}

stk::mesh::EntityVector
Topology::getErodibleCells()
{
  auto&                   bulk_data = get_bulk_data();
  auto const              cell_rank = stk::topology::ELEMENT_RANK;
  stk::mesh::EntityVector cells;
  stk::mesh::EntityVector erodible_cells;
  stk::mesh::get_entities(bulk_data, cell_rank, cells);

  for (auto cell : cells) {
    if (is_erodible_cell(cell) == true) erodible_cells.emplace_back(cell);
  }
  return erodible_cells;
}

std::vector<stk::mesh::EntityId>
Topology::getEntityGIDs(stk::mesh::EntityVector const& entities)
{
  std::vector<stk::mesh::EntityId> gids;
  for (auto entity : entities) {
    gids.emplace_back(get_gid(entity));
  }
  return gids;
}

// Compute volume of given cell
double
Topology::getCellVolume(stk::mesh::Entity const cell)
{
  stk::mesh::EntityRank const cell_rank       = stk::topology::ELEMENT_RANK;
  stk::mesh::EntityRank const node_rank       = stk::topology::NODE_RANK;
  auto&                       bulk_data       = get_bulk_data();
  auto&                       stk_mesh_struct = *(get_stk_mesh_struct());
  auto&                       coord_field     = *(stk_mesh_struct.getCoordinatesField());

  assert(bulk_data.entity_rank(cell) == cell_rank);
  stk::mesh::Entity const*                   relations     = bulk_data.begin(cell, node_rank);
  auto const                                 num_relations = bulk_data.num_connectivity(cell, node_rank);
  std::vector<minitensor::Vector<double, 3>> points;
  for (auto i = 0; i < num_relations; ++i) {
    stk::mesh::Entity node = relations[i];
    double*           pc   = stk::mesh::field_data(coord_field, node);

    minitensor::Vector<double, 3> point(pc[0], pc[1], pc[2]);
    points.emplace_back(point);
  }

  double volume{0.0};
  switch (num_relations) {
    default: ALBANY_ASSERT(num_relations == 4 || num_relations == 8); break;
    case 4: volume = minitensor::volume(points[0], points[1], points[2], points[3]); break;
    case 8: volume = minitensor::volume(points[0], points[1], points[2], points[3], points[4], points[5], points[6], points[7]); break;
  }

  return volume;
}

// Create boundary
void
Topology::createBoundary()
{
  stk::mesh::EntityRank const cell_rank     = stk::topology::ELEMENT_RANK;
  stk::mesh::EntityRank const face_rank     = stk::topology::FACE_RANK;
  auto&                       bulk_data     = get_bulk_data();
  auto&                       meta_data     = get_meta_data();
  stk::mesh::Part&            del_part      = meta_data.declare_part("Boundary", face_rank);
  stk::mesh::Part&            locally_owned = meta_data.locally_owned_part();
  stk::mesh::PartVector       add_parts(1, &del_part);
  stk::mesh::skin_mesh(bulk_data, add_parts);
  stk::mesh::Selector            del_selector = del_part & locally_owned;
  stk::mesh::BucketVector const& face_buckets = bulk_data.buckets(face_rank);
  stk::mesh::EntityVector        boundary_faces;
  stk::mesh::get_selected_entities(del_selector, face_buckets, boundary_faces);
  boundary_.clear();
  for (auto face : boundary_faces) {
    boundary_.emplace(face);
  }
}

// Create the full mesh representation. This must be done prior to
// the adaptation query.
void
Topology::graphInitialization()
{
  // Create boundary first
  createBoundary();
  auto& bulk_data = get_bulk_data();
  set_output_type(UNIDIRECTIONAL_MULTILEVEL);
  createAllLevelsRelations();
  initializeCellFailureState();
  setCellBoundaryIndicator();
  setNodeBoundaryIndicator();
  Albany::fix_node_sharing(bulk_data);
  get_stk_discretization().updateMesh();
  initializeTopologies();
  initializeHighestIds();
}

void
Topology::createAllLevelsRelations()
{
  stk::mesh::PartVector add_parts;
  auto&                 bulk_data = get_bulk_data();
  stk::mesh::create_adjacent_entities(bulk_data, add_parts);
}

// Creates temporary nodal connectivity for the elements and removes
// the relationships between the elements and nodes.
void
Topology::removeNodeRelations()
{
  // Create the temporary connectivity array
  stk::mesh::EntityVector elements;

  stk::mesh::get_entities(get_bulk_data(), stk::topology::ELEMENT_RANK, elements);

  modification_begin();

  for (size_t i = 0; i < elements.size(); ++i) {
    stk::mesh::Entity const* relations     = get_bulk_data().begin_nodes(elements[i]);
    size_t const             num_relations = get_bulk_data().num_nodes(elements[i]);

    stk::mesh::EntityVector nodes(relations, relations + num_relations);

    connectivity_.push_back(nodes);

    for (size_t j = 0; j < nodes.size(); ++j) {
      get_bulk_data().destroy_relation(elements[i], nodes[j], j);
    }
  }

  Albany::fix_node_sharing(get_bulk_data());
  modification_end();

  return;
}

// Removes multilevel relations.
void
Topology::removeMultiLevelRelations()
{
  typedef std::vector<EdgeId>   EdgeIdList;
  typedef EdgeIdList::size_type EdgeIdListIndex;
  auto const                    cell_rank = stk::topology::ELEMENT_RANK;
  auto const                    node_rank = stk::topology::NODE_RANK;
  auto&                         bulk_data = get_bulk_data();

  modification_begin();
  // Go from points to cells
  for (auto source_rank = node_rank; source_rank <= cell_rank; ++source_rank) {
    stk::mesh::EntityVector source_entities;
    stk::mesh::get_entities(bulk_data, source_rank, source_entities);

    for (auto source_entity : source_entities) {
      stk::mesh::EntityVector target_entities;
      EdgeIdList              multilevel_relation_ids;

      for (auto target_rank = node_rank; target_rank <= cell_rank; ++target_rank) {
        stk::mesh::Entity const* relations = bulk_data.begin(source_entity, target_rank);

        size_t const num_relations = bulk_data.num_connectivity(source_entity, target_rank);

        stk::mesh::ConnectivityOrdinal const* ordinals = bulk_data.begin_ordinals(source_entity, target_rank);

        // Collect relations to delete
        for (size_t r = 0; r < num_relations; ++r) {
          auto const rank_distance       = source_rank - target_rank;
          bool const is_invalid_relation = rank_distance > 1;

          if (is_invalid_relation == true) {
            target_entities.push_back(relations[r]);
            multilevel_relation_ids.push_back(ordinals[r]);
          }
        }
      }

      // Delete them
      for (EdgeIdListIndex i = 0; i < multilevel_relation_ids.size(); ++i) {
        stk::mesh::Entity target_entity = target_entities[i];
        EdgeId const      id            = multilevel_relation_ids[i];
        bulk_data.destroy_relation(source_entity, target_entity, id);
      }
    }
  }
  modification_end();
}

// Remove all entities but the ones representing elements and nodes.
// Only 3D for now.
void
Topology::removeMidLevelEntities()
{
  auto&                   bulk_data = get_bulk_data();
  auto const              edge_rank = stk::topology::EDGE_RANK;
  auto const              face_rank = stk::topology::FACE_RANK;
  auto const              cell_rank = stk::topology::ELEMENT_RANK;
  stk::mesh::EntityVector edges;
  stk::mesh::EntityVector faces;
  stk::mesh::EntityVector cells;
  stk::mesh::get_entities(bulk_data, edge_rank, edges);
  stk::mesh::get_entities(bulk_data, face_rank, faces);
  stk::mesh::get_entities(bulk_data, cell_rank, cells);

  modification_begin();

  for (auto cell : cells) {
    stk::mesh::Entity const* relations     = bulk_data.begin_faces(cell);
    size_t const             num_relations = bulk_data.num_faces(cell);
    stk::mesh::EntityVector  cell_faces(relations, relations + num_relations);
    for (size_t j = 0; j < cell_faces.size(); ++j) {
      assert(bulk_data.destroy_relation(cell, cell_faces[j], j) == true);
    }
  }

  for (auto face : faces) {
    stk::mesh::Entity const* relations     = bulk_data.begin_edges(face);
    size_t const             num_relations = bulk_data.num_edges(face);
    stk::mesh::EntityVector  face_edges(relations, relations + num_relations);
    for (size_t j = 0; j < face_edges.size(); ++j) {
      assert(bulk_data.destroy_relation(face, face_edges[j], j) == true);
    }
    assert(num_connectivity(face) == 0);
    remove_entity(face);
  }

  for (auto edge : edges) {
    stk::mesh::Entity const* relations     = bulk_data.begin_nodes(edge);
    size_t const             num_relations = bulk_data.num_nodes(edge);
    stk::mesh::EntityVector  edge_nodes(relations, relations + num_relations);
    for (size_t j = 0; j < edge_nodes.size(); ++j) {
      assert(bulk_data.destroy_relation(edge, edge_nodes[j], j) == true);
    }
    assert(num_connectivity(edge) == 0);
    remove_entity(edge);
  }

  modification_end();
}

// After mesh manipulations are complete, need to recreate a stk
// mesh understood by Albany::STKDiscretization.
void
Topology::restoreElementToNodeConnectivity()
{
  stk::mesh::EntityVector elements;

  stk::mesh::get_entities(get_bulk_data(), stk::topology::ELEMENT_RANK, elements);

  modification_begin();

  // Add relations from element to nodes
  for (size_t i = 0; i < elements.size(); ++i) {
    stk::mesh::Entity element = elements[i];

    stk::mesh::EntityVector element_connectivity = connectivity_[i];

    for (size_t j = 0; j < element_connectivity.size(); ++j) {
      stk::mesh::Entity node = element_connectivity[j];

      get_bulk_data().declare_relation(element, node, j);
    }
  }
  Albany::fix_node_sharing(get_bulk_data());
  modification_end();

  return;
}

// Determine nodes associated with a boundary entity
stk::mesh::EntityVector
Topology::getBoundaryEntityNodes(stk::mesh::Entity boundary_entity)
{
  stk::mesh::EntityRank const boundary_rank = get_bulk_data().entity_rank(boundary_entity);

  assert(boundary_rank == stk::topology::ELEMENT_RANK - 1);

  stk::mesh::Entity const* relations = get_bulk_data().begin_elements(boundary_entity);

  stk::mesh::Entity first_cell = relations[0];

  size_t const num_cell_nodes = get_bulk_data().num_nodes(first_cell);

  stk::mesh::Entity const* node_relations = get_bulk_data().begin_nodes(first_cell);

  stk::mesh::ConnectivityOrdinal const* node_ords = get_bulk_data().begin_node_ordinals(first_cell);

  stk::mesh::ConnectivityOrdinal const* ords = get_bulk_data().begin_element_ordinals(boundary_entity);

  EdgeId const face_order = ords[0];

  shards::CellTopology const cell_topology = get_cell_topology();

  RelationVectorIndex const number_face_nodes = cell_topology.getNodeCount(boundary_rank, face_order);

  stk::mesh::EntityVector nodes;

  for (RelationVectorIndex i = 0; i < number_face_nodes; ++i) {
    EdgeId const cell_order = cell_topology.getNodeMap(boundary_rank, face_order, i);

    // Brute force approach. Maybe there is a better way to do this?
    for (size_t j = 0; j < num_cell_nodes; ++j) {
      if (node_ords[j] == cell_order) {
        nodes.push_back(node_relations[j]);
      }
    }
  }

  return nodes;
}

// Get nodal coordinates
std::vector<minitensor::Vector<double>>
Topology::getNodalCoordinates()
{
  stk::mesh::Selector local_selector = get_meta_data().locally_owned_part();

  std::vector<stk::mesh::Bucket*> const& buckets = get_bulk_data().buckets(stk::topology::NODE_RANK);

  stk::mesh::EntityVector entities;

  stk::mesh::get_selected_entities(local_selector, buckets, entities);

  EntityVectorIndex const number_nodes = entities.size();

  std::vector<minitensor::Vector<double>> coordinates(number_nodes);

  size_t const dimension = get_space_dimension();

  minitensor::Vector<double> X(dimension);

  VectorFieldType& node_coordinates = *(get_stk_mesh_struct()->getCoordinatesField());

  for (EntityVectorIndex i = 0; i < number_nodes; ++i) {
    stk::mesh::Entity node = entities[i];

    double const* const pointer_coordinates = stk::mesh::field_data(node_coordinates, node);

    for (size_t j = 0; j < dimension; ++j) {
      X(j) = pointer_coordinates[j];
    }

    coordinates[i] = X;
  }

  return coordinates;
}

// Output of boundary
void
Topology::outputBoundary(std::string const& output_filename)
{
  // Open output file
  std::ofstream ofs;

  ofs.open(output_filename.c_str(), std::ios::out);

  if (ofs.is_open() == false) {
    std::cout << "Unable to open boundary output file: ";
    std::cout << output_filename << '\n';
    return;
  }

  std::cout << "Write boundary file: ";
  std::cout << output_filename << '\n';

  // Header
  ofs << "# vtk DataFile Version 3.0\n";
  ofs << "LCM/LCM\n";
  ofs << "ASCII\n";
  ofs << "DATASET UNSTRUCTURED_GRID\n";

  // Coordinates
  Coordinates const coordinates = getNodalCoordinates();

  CoordinatesIndex const number_nodes = coordinates.size();

  ofs << "POINTS " << number_nodes << " double\n";

  for (CoordinatesIndex i = 0; i < number_nodes; ++i) {
    minitensor::Vector<double> const& X = coordinates[i];

    for (minitensor::Index j = 0; j < X.get_dimension(); ++j) {
      ofs << std::setw(24) << std::scientific << std::setprecision(16) << X(j);
    }
    ofs << '\n';
  }

  Connectivity const connectivity = getBoundary();

  ConnectivityIndex const number_cells = connectivity.size();

  size_t cell_list_size = 0;

  for (size_t i = 0; i < number_cells; ++i) {
    cell_list_size += connectivity[i].size() + 1;
  }

  // Boundary cell connectivity
  ofs << "CELLS " << number_cells << " " << cell_list_size << '\n';
  for (size_t i = 0; i < number_cells; ++i) {
    size_t const number_cell_nodes = connectivity[i].size();

    ofs << number_cell_nodes;

    for (size_t j = 0; j < number_cell_nodes; ++j) {
      ofs << ' ' << connectivity[i][j] - 1;
    }
    ofs << '\n';
  }

  ofs << "CELL_TYPES " << number_cells << '\n';
  for (size_t i = 0; i < number_cells; ++i) {
    size_t const number_cell_nodes = connectivity[i].size();

    VTKCellType cell_type = INVALID;

    switch (number_cell_nodes) {
      default:
        std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
        std::cerr << '\n';
        std::cerr << "Invalid number of nodes in boundary cell: ";
        std::cerr << number_cell_nodes;
        std::cerr << '\n';
        exit(1);
        break;

      case 1: cell_type = VERTEX; break;

      case 2: cell_type = LINE; break;

      case 3: cell_type = TRIANGLE; break;

      case 4: cell_type = QUAD; break;
    }
    ofs << cell_type << '\n';
  }

  ofs.close();
  return;
}

// Create boundary mesh
Connectivity
Topology::getBoundary()
{
  stk::mesh::EntityRank const boundary_rank = get_boundary_rank();

  stk::mesh::EntityRank const cell_rank = static_cast<stk::mesh::EntityRank>(boundary_rank + 1);

  stk::mesh::EntityVector faces;

  stk::mesh::get_entities(get_bulk_data(), boundary_rank, faces);

  Connectivity connectivity;

  EntityVectorIndex const number_faces = faces.size();

  for (EntityVectorIndex i = 0; i < number_faces; ++i) {
    stk::mesh::Entity const face = faces[i];

    stk::mesh::Entity const* cell_relations = get_bulk_data().begin(face, cell_rank);

    size_t const number_connected_cells = get_bulk_data().num_elements(face);

    switch (number_connected_cells) {
      default:
        std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
        std::cerr << '\n';
        std::cerr << "Invalid number of connected cells: ";
        std::cerr << number_connected_cells;
        std::cerr << '\n';
        exit(1);
        break;

      case 1:
        // Make sure connected cell is in bulk and continue.
        assert(is_bulk_cell(cell_relations[0]) == true);
        break;

      case 2:
        // Two connected faces. Determine 1f one of them is a surface element.
        {
          stk::mesh::Entity const cell_0 = cell_relations[0];

          stk::mesh::Entity const cell_1 = cell_relations[1];

          bool const is_in_bulk_0 = is_in_bulk(cell_0);

          bool const is_in_bulk_1 = is_in_bulk(cell_1);

          bool const both_bulk = is_in_bulk_0 && is_in_bulk_1;

          // If internal face do nothing.
          if (both_bulk == true) continue;

          bool const is_in_interface_0 = is_in_interface(cell_0);

          bool const is_in_interface_1 = is_in_interface(cell_1);

          bool const both_interface = is_in_interface_0 && is_in_interface_1;

          if (both_interface == true) {
            std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
            std::cerr << '\n';
            std::cerr << "Cannot be connected to two surface elements: ";
            std::cerr << "Face: " << get_entity_id(face);
            std::cerr << '\n';
            exit(1);
          }

          // One element is bulk and one interface, so it's an internal face.
        }
        break;
    }

    stk::mesh::EntityVector const nodes = getBoundaryEntityNodes(face);

    EntityVectorIndex const number_nodes = nodes.size();

    std::vector<stk::mesh::EntityId> node_ids(number_nodes);

    for (EntityVectorIndex i = 0; i < number_nodes; ++i) {
      node_ids[i] = get_entity_id(nodes[i]);
    }

    connectivity.push_back(node_ids);
  }

  return connectivity;
}

///
/// Compute normal using first 3 nodes of boundary entity.
///
minitensor::Vector<double>
Topology::get_normal(stk::mesh::Entity boundary_entity)
{
  shards::CellTopology const cell_topology = get_cell_topology();

  stk::mesh::EntityRank const boundary_rank = get_bulk_data().entity_rank(boundary_entity);

  stk::mesh::ConnectivityOrdinal const* ords = get_bulk_data().begin_element_ordinals(boundary_entity);

  EdgeId const face_order = ords[0];

  RelationVectorIndex const num_corner_nodes = cell_topology.getVertexCount(boundary_rank, face_order);

  assert(num_corner_nodes >= 3);

  size_t const dimension = get_space_dimension();

  std::vector<minitensor::Vector<double>> nodal_coords(num_corner_nodes);

  stk::mesh::EntityVector nodes = getBoundaryEntityNodes(boundary_entity);

  VectorFieldType& coordinates = *(get_stk_mesh_struct()->getCoordinatesField());

  for (EntityVectorIndex i = 0; i < num_corner_nodes; ++i) {
    stk::mesh::Entity node = nodes[i];

    double const* const pointer_coordinates = stk::mesh::field_data(coordinates, node);

    minitensor::Vector<double>& X = nodal_coords[i];

    X.set_dimension(dimension);

    for (size_t j = 0; j < dimension; ++j) {
      X(j) = pointer_coordinates[j];
    }
  }

  minitensor::Vector<double> const v1 = nodal_coords[1] - nodal_coords[0];

  minitensor::Vector<double> const v2 = nodal_coords[2] - nodal_coords[0];

  minitensor::Vector<double> normal = minitensor::unit(minitensor::cross(v1, v2));

  return normal;
}

// Create cohesive connectivity
stk::mesh::EntityVector
Topology::createSurfaceElementConnectivity(stk::mesh::Entity face_top, stk::mesh::Entity face_bottom)
{
  // Check first if normals point in the same direction, just in case.
  minitensor::Vector<double> const normal_top = get_normal(face_top);

  minitensor::Vector<double> const normal_bottom = get_normal(face_bottom);

  if (minitensor::dot(normal_top, normal_bottom) > 0.0) {
    std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
    std::cerr << '\n';
    std::cerr << "Face normals have the same instead of opposite directions:\n";
    std::cerr << "Normal top    :" << normal_top << '\n';
    std::cerr << "Normal bottom :" << normal_bottom << '\n';
    std::cerr << '\n';
    exit(1);
  }

  stk::mesh::EntityVector nodes_top = getBoundaryEntityNodes(face_top);

  EntityVectorIndex const num_top = nodes_top.size();

  stk::mesh::EntityVector nodes_bottom = getBoundaryEntityNodes(face_bottom);

  EntityVectorIndex const num_bottom = nodes_bottom.size();

  assert(num_top == num_bottom);

  stk::mesh::EntityVector reordered;

  // Ensure that the order of the bottom nodes is the same as the top ones.
  VectorFieldType& coordinates = *(get_stk_mesh_struct()->getCoordinatesField());

  size_t const dimension = get_space_dimension();

  for (EntityVectorIndex i = 0; i < num_top; ++i) {
    stk::mesh::Entity node_top = nodes_top[i];

    double const* const p_top = stk::mesh::field_data(coordinates, node_top);

    minitensor::Vector<double> X(dimension);

    for (size_t n = 0; n < dimension; ++n) {
      X(n) = p_top[n];
    }

    bool found = false;

    for (EntityVectorIndex j = 0; j < num_bottom; ++j) {
      stk::mesh::Entity node_bottom = nodes_bottom[j];

      double const* const p_bottom = stk::mesh::field_data(coordinates, node_bottom);

      minitensor::Vector<double> Y(dimension);

      for (size_t n = 0; n < dimension; ++n) {
        Y(n) = p_bottom[n];
      }

      if (X == Y) {
        reordered.push_back(node_bottom);
        found = true;
        break;
      }
    }

    if (found == false) {
      std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
      std::cerr << '\n';
      std::cerr << "Nodes on top and bottom faces do not match.";
      std::cerr << '\n';
      exit(1);
    }
  }

  stk::mesh::EntityVector nodes;

  nodes.reserve(nodes_top.size() + nodes_bottom.size());

  nodes.insert(nodes.end(), nodes_top.begin(), nodes_top.end());
  nodes.insert(nodes.end(), reordered.begin(), reordered.end());

  return nodes;
}

// Create vectors describing the vertices and edges of the star of
// an entity in the stk mesh.
void
Topology::createStar(stk::mesh::Entity entity, std::set<stk::mesh::Entity>& subgraph_entities, std::set<STKEdge, EdgeLessThan>& subgraph_edges)
{
  subgraph_entities.insert(entity);

  assert(get_space_dimension() == 3);

  stk::mesh::EntityRank const rank = get_bulk_data().entity_rank(entity);

  stk::mesh::EntityRank const one_up = static_cast<stk::mesh::EntityRank>(rank + 1);

  stk::mesh::Entity const* relations = get_bulk_data().begin(entity, one_up);

  size_t const num_relations = get_bulk_data().num_connectivity(entity, one_up);

  stk::mesh::ConnectivityOrdinal const* ords = get_bulk_data().begin_ordinals(entity, one_up);

  for (size_t i = 0; i < num_relations; ++i) {
    stk::mesh::Entity source = relations[i];

    if (is_interface_cell(source) == true) continue;

    STKEdge edge;

    edge.source   = source;
    edge.target   = entity;
    edge.local_id = ords[i];

    subgraph_edges.insert(edge);
    createStar(source, subgraph_entities, subgraph_edges);
  }

  return;
}

// Fractures all open boundary entities of the mesh.
void
Topology::splitOpenFaces()
{
  // 3D only for now.
  assert(get_space_dimension() == stk::topology::ELEMENT_RANK);

  stk::mesh::EntityVector points;

  stk::mesh::EntityVector open_points;

  stk::mesh::Selector local_bulk = get_local_bulk_selector();

  std::set<EntityPair> fractured_faces;

  stk::mesh::BulkData& bulk_data = get_bulk_data();

  stk::mesh::BucketVector const& point_buckets = bulk_data.buckets(stk::topology::NODE_RANK);

  stk::mesh::get_selected_entities(local_bulk, point_buckets, points);

  // Collect open points
  for (stk::mesh::EntityVector::iterator i = points.begin(); i != points.end(); ++i) {
    stk::mesh::Entity point = *i;

    if (get_entity_failure_state(point) == FAILED) {
      open_points.push_back(point);
    }
  }

  modification_begin();

  // Iterate over open points and fracture them.
  for (stk::mesh::EntityVector::iterator i = open_points.begin(); i != open_points.end(); ++i) {
    stk::mesh::Entity point = *i;

    stk::mesh::Entity const* segment_relations = bulk_data.begin_edges(point);

    size_t const num_segments = bulk_data.num_edges(point);

    stk::mesh::EntityVector open_segments;

    // Collect open segments.
    for (size_t j = 0; j < num_segments; ++j) {
      stk::mesh::Entity segment = segment_relations[j];

      bool const is_local_segment = is_local_entity(segment) == true;

      bool const is_open_segment = get_entity_failure_state(segment) == FAILED;

      bool const is_local_and_open_segment = is_local_segment == true && is_open_segment == true;

      if (is_local_and_open_segment == true) {
        open_segments.push_back(segment);
      }
    }

    // Iterate over open segments and fracture them.
    for (stk::mesh::EntityVector::iterator j = open_segments.begin(); j != open_segments.end(); ++j) {
      stk::mesh::Entity segment = *j;

      // Create star of segment
      std::set<stk::mesh::Entity> star_entities;

      std::set<STKEdge, EdgeLessThan> star_edges;

      createStar(segment, star_entities, star_edges);

      // Iterators
      std::set<stk::mesh::Entity>::iterator first_entity = star_entities.begin();

      std::set<stk::mesh::Entity>::iterator last_entity = star_entities.end();

      std::set<STKEdge>::iterator first_edge = star_edges.begin();

      std::set<STKEdge>::iterator last_edge = star_edges.end();

      Subgraph segment_star(*this, first_entity, last_entity, first_edge, last_edge);

      // Collect open faces
      stk::mesh::Entity const* face_relations = bulk_data.begin_faces(segment);

      size_t const num_faces = bulk_data.num_faces(segment);

      stk::mesh::EntityVector open_faces;

      for (size_t k = 0; k < num_faces; ++k) {
        stk::mesh::Entity face = face_relations[k];

        bool const is_local_face = is_local_entity(face);

        bool const is_open_face = is_internal_and_open_face(face) == true;

        bool const is_local_and_open_face = is_local_face == true && is_open_face == true;

        if (is_local_and_open_face == true) {
          open_faces.push_back(face);
        }
      }

      // Iterate over the open faces
      for (stk::mesh::EntityVector::iterator k = open_faces.begin(); k != open_faces.end(); ++k) {
        stk::mesh::Entity face = *k;

        Vertex face_vertex = segment_star.vertexFromEntity(face);

        Vertex new_face_vertex = segment_star.cloneBoundaryVertex(face_vertex);

        stk::mesh::Entity new_face = segment_star.entityFromVertex(new_face_vertex);

        // Reset fracture state for both old and new faces
        set_failure_state(face, INTACT);
        set_failure_state(new_face, INTACT);

        EntityPair face_pair = std::make_pair(face, new_face);

        fractured_faces.insert(face_pair);
      }

      // Split the articulation point (current segment)
      Vertex segment_vertex = segment_star.vertexFromEntity(segment);

      segment_star.splitArticulation(segment_vertex);

      // Reset segment fracture state
      set_failure_state(segment, INTACT);
    }

    // All open faces and segments have been dealt with.
    // Split the node articulation point
    // Create star of node
    std::set<stk::mesh::Entity> star_entities;

    std::set<STKEdge, EdgeLessThan> star_edges;

    createStar(point, star_entities, star_edges);

    // Iterators
    std::set<stk::mesh::Entity>::iterator first_entity = star_entities.begin();

    std::set<stk::mesh::Entity>::iterator last_entity = star_entities.end();

    std::set<STKEdge>::iterator first_edge = star_edges.begin();

    std::set<STKEdge>::iterator last_edge = star_edges.end();

    Subgraph point_star(*this, first_entity, last_entity, first_edge, last_edge);

    Vertex point_vertex = point_star.vertexFromEntity(point);

    EntityEntityMap new_connectivity = point_star.splitArticulation(point_vertex);

    // Reset fracture state of point
    set_failure_state(point, INTACT);

    // Update the connectivity
    for (EntityEntityMap::iterator j = new_connectivity.begin(); j != new_connectivity.end(); ++j) {
      stk::mesh::Entity new_point = j->second;

      bulk_data.copy_entity_fields(point, new_point);
    }
  }

  Albany::fix_node_sharing(bulk_data);
  modification_end();

  bool const insert_surface_elements = false;

  if (insert_surface_elements == true) {
    insertSurfaceElements(fractured_faces);
  }

  return;
}

void
Topology::printFailureState()
{
  stk::mesh::BulkData&        bulk_data = get_bulk_data();
  stk::mesh::EntityRank const cell_rank = stk::topology::ELEMENT_RANK;
  stk::mesh::EntityVector     cells;
  auto&                       fos = *Teuchos::VerboseObjectBase::getDefaultOStream();
  stk::mesh::get_entities(bulk_data, cell_rank, cells);
  for (auto cell : cells) {
    auto const fs        = get_entity_failure_state(cell);
    auto       proc_rank = get_proc_rank();
    auto       gid       = get_gid(cell) - 1;
    std::cout << "**** DEBUG TOPO PROC GID FAILED: " << proc_rank << " " << std::setw(3) << std::setfill('0') << gid << " " << fs << "\n";
  }
}

// Destroy upward relations of an entity
void
Topology::remove_entity_and_up_relations(stk::mesh::Entity entity)
{
  auto&      bulk_data   = get_bulk_data();
  auto const cell_rank   = stk::topology::ELEMENT_RANK;
  auto const entity_rank = bulk_data.entity_rank(entity);
  ALBANY_ASSERT(bulk_data.is_valid(entity) == true);
  for (auto rank = cell_rank; rank != entity_rank; --rank) {
    auto const num_conn = bulk_data.num_connectivity(entity, rank);
    if (num_conn == 0) continue;
    auto const* rel_entities = bulk_data.begin(entity, rank);
    auto const* rel_ordinals = bulk_data.begin_ordinals(entity, rank);
    // Must delete relation from higher-to-lower ranked entity
    // DO NOT use auto here. Need a signed int.
    for (int j = num_conn - 1; j >= 0; --j) {
      ALBANY_ASSERT(bulk_data.is_valid(rel_entities[j]) == true);
      bulk_data.destroy_relation(rel_entities[j], entity, rel_ordinals[j]);
    }
  }
  bool successfully_destroyed = bulk_data.destroy_entity(entity);
  ALBANY_ASSERT(successfully_destroyed == true);
}

int
Topology::numberCells()
{
  auto const              cell_rank     = stk::topology::ELEMENT_RANK;
  auto&                   bulk_data     = get_bulk_data();
  auto&                   meta_data     = get_meta_data();
  auto&                   locally_owned = meta_data.locally_owned_part();
  auto const&             cell_buckets  = bulk_data.buckets(cell_rank);
  stk::mesh::EntityVector cells;
  stk::mesh::get_selected_entities(locally_owned, cell_buckets, cells);
  return cells.size();
}

// Remove failed elements from the mesh
double
Topology::erodeFailedElements()
{
  auto const cell_rank     = stk::topology::ELEMENT_RANK;
  auto const face_rank     = stk::topology::FACE_RANK;
  auto const edge_rank     = stk::topology::EDGE_RANK;
  auto const node_rank     = stk::topology::NODE_RANK;
  auto&      bulk_data     = get_bulk_data();
  auto&      meta_data     = get_meta_data();
  auto&      locally_owned = meta_data.locally_owned_part();
  double     eroded_volume = 0.0;

  assert(get_space_dimension() == cell_rank);
  modification_begin();

  // Collect and remove failed cells
  auto const&             cell_buckets = bulk_data.buckets(cell_rank);
  stk::mesh::EntityVector cells;
  stk::mesh::get_selected_entities(locally_owned, cell_buckets, cells);
  auto& bulk_failure_criterion      = static_cast<BulkFailureCriterion&>(*failure_criterion_);
  bulk_failure_criterion.accumulate = true;
  for (auto cell : cells) {
    if (failure_criterion_->check(bulk_data, cell) == true) {
      auto cell_volume = getCellVolume(cell);
      eroded_volume += cell_volume;
      set_failure_state(cell, INTACT);
      remove_entity_and_up_relations(cell);
    }
  }
  bulk_failure_criterion.accumulate = false;

  auto const&             face_buckets = bulk_data.buckets(face_rank);
  stk::mesh::EntityVector faces;
  stk::mesh::get_selected_entities(locally_owned, face_buckets, faces);
  for (auto face : faces) {
    auto const num_elems = bulk_data.num_elements(face);
    if (num_elems == 0) {
      remove_entity_and_up_relations(face);
    }
  }
  auto const&             edge_buckets = bulk_data.buckets(edge_rank);
  stk::mesh::EntityVector edges;
  stk::mesh::get_selected_entities(locally_owned, edge_buckets, edges);
  for (auto edge : edges) {
    auto const num_elems = bulk_data.num_elements(edge);
    auto const num_faces = bulk_data.num_faces(edge);
    if (num_elems == 0 || num_faces == 0) {
      remove_entity_and_up_relations(edge);
    }
  }
  auto const&             node_buckets = bulk_data.buckets(node_rank);
  stk::mesh::EntityVector nodes;
  stk::mesh::get_selected_entities(locally_owned, node_buckets, nodes);
  for (auto node : nodes) {
    auto const num_elems = bulk_data.num_elements(node);
    auto const num_faces = bulk_data.num_faces(node);
    auto const num_edges = bulk_data.num_edges(node);
    if (num_elems == 0 || num_faces == 0 || num_edges == 0) {
      remove_entity_and_up_relations(node);
    }
  }
  modification_end();
  Albany::fix_node_sharing(bulk_data);
  initializeCellFailureState();
  createBoundary();
  setCellBoundaryIndicator();
  setNodeBoundaryIndicator();

  return eroded_volume;
}

bool
Topology::isIsolatedNode(stk::mesh::Entity entity)
{
  stk::mesh::EntityRank const cell_rank   = stk::topology::ELEMENT_RANK;
  stk::mesh::EntityRank const node_rank   = stk::topology::NODE_RANK;
  stk::mesh::BulkData&        bulk_data   = get_bulk_data();
  auto const                  entity_rank = bulk_data.entity_rank(entity);
  ALBANY_ASSERT(entity_rank == node_rank);
  stk::mesh::EntityVector cells;
  stk::mesh::get_entities(bulk_data, cell_rank, cells);
  for (RelationVectorIndex i = 0; i < cells.size(); ++i) {
    stk::mesh::Entity        cell          = cells[i];
    stk::mesh::Entity const* relations     = bulk_data.begin_nodes(cell);
    auto const               num_relations = get_bulk_data().num_nodes(cell);
    stk::mesh::EntityVector  nodes(relations, relations + num_relations);
    for (RelationVectorIndex j = 0; j < nodes.size(); ++j) {
      stk::mesh::Entity node = nodes[j];
      if (node == entity) return false;
    }
  }
  return true;
}

void
Topology::insertSurfaceElements(std::set<EntityPair> const& fractured_faces)
{
  stk::mesh::BulkData& bulk_data = get_bulk_data();

  modification_begin();

  // Same rank as bulk cells!
  stk::mesh::EntityRank const interface_rank = stk::topology::ELEMENT_RANK;

  stk::mesh::Part& interface_part = failure_criterion_->get_interface_part();

  stk::mesh::PartVector interface_parts;

  interface_parts.push_back(&interface_part);

  increase_highest_id(interface_rank);

  stk::mesh::EntityId new_id = get_highest_id(interface_rank);

  // Create the interface connectivity
  for (std::set<EntityPair>::iterator i = fractured_faces.begin(); i != fractured_faces.end(); ++i) {
    stk::mesh::Entity face1 = i->first;

    stk::mesh::Entity face2 = i->second;

    stk::mesh::EntityVector interface_points = createSurfaceElementConnectivity(face1, face2);

    // Insert the surface element
    stk::mesh::Entity new_surface = bulk_data.declare_entity(interface_rank, new_id, interface_parts);

    // Connect to faces
    bulk_data.declare_relation(new_surface, face1, 0);
    bulk_data.declare_relation(new_surface, face2, 1);

    // Connect to points
    for (EntityVectorIndex j = 0; j < interface_points.size(); ++j) {
      stk::mesh::Entity interface_point = interface_points[j];

      bulk_data.declare_relation(new_surface, interface_point, j);
    }

    ++new_id;
  }

  Albany::fix_node_sharing(bulk_data);
  modification_end();

  return;
}

size_t
Topology::setEntitiesOpen()
{
  stk::mesh::EntityVector boundary_entities;

  stk::mesh::Selector local_bulk = get_local_bulk_selector();

  stk::mesh::get_selected_entities(local_bulk, get_bulk_data().buckets(get_boundary_rank()), boundary_entities);

  size_t counter = 0;

  // Iterate over the boundary entities
  for (EntityVectorIndex i = 0; i < boundary_entities.size(); ++i) {
    stk::mesh::Entity entity = boundary_entities[i];

    if (is_internal_face(entity) == false) continue;

    if (checkOpen(entity) == false) continue;

    set_failure_state(entity, FAILED);
    ++counter;

    switch (get_space_dimension()) {
      default:
        std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
        std::cerr << '\n';
        std::cerr << "Invalid cells rank in fracture: ";
        std::cerr << stk::topology::ELEMENT_RANK;
        std::cerr << '\n';
        exit(1);
        break;

      case stk::topology::ELEMENT_RANK: {
        stk::mesh::Entity const* segments = get_bulk_data().begin_edges(entity);

        size_t const num_segments = get_bulk_data().num_edges(entity);

        for (size_t j = 0; j < num_segments; ++j) {
          stk::mesh::Entity segment = segments[j];

          set_failure_state(segment, FAILED);

          stk::mesh::Entity const* points = get_bulk_data().begin_nodes(segment);

          size_t const num_points = get_bulk_data().num_nodes(segment);

          for (size_t k = 0; k < num_points; ++k) {
            stk::mesh::Entity point = points[k];

            set_failure_state(point, FAILED);
          }
        }
      } break;

      case stk::topology::EDGE_RANK: {
        stk::mesh::Entity const* points = get_bulk_data().begin_nodes(entity);

        size_t const num_points = get_bulk_data().num_nodes(entity);

        for (size_t j = 0; j < num_points; ++j) {
          stk::mesh::Entity point = points[j];

          set_failure_state(point, FAILED);
        }
      } break;
    }
  }

  return counter;
}

// Output the graph associated with the mesh to graphviz .dot
// file for visualization purposes.
void
Topology::outputToGraphviz(std::string const& output_filename)
{
  // Open output file
  std::ofstream gviz_out;

  gviz_out.open(output_filename.c_str(), std::ios::out);

  if (gviz_out.is_open() == false) {
    std::cout << "Unable to open graphviz output file: ";
    std::cout << output_filename << '\n';
    return;
  }

  std::cout << "Write graph to graphviz dot file: ";
  std::cout << output_filename << '\n';

  // Write beginning of file
  gviz_out << dot_header();

  typedef std::vector<EntityPair> RelationList;

  RelationList relation_list;

  std::vector<EdgeId> relation_local_id;

  // Entities (graph vertices)
  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank <= stk::topology::ELEMENT_RANK; ++rank) {
    stk::mesh::EntityVector entities;

    stk::mesh::get_entities(get_bulk_data(), rank, entities);

    for (EntityVectorIndex i = 0; i < entities.size(); ++i) {
      stk::mesh::Entity source_entity = entities[i];

      FailureState const failure_state = get_entity_failure_state(source_entity);

      stk::mesh::EntityId const source_id = get_entity_id(source_entity);

      gviz_out << dot_entity(get_space_dimension(), get_parallel_rank(), source_entity, source_id, rank, failure_state);

      for (stk::mesh::EntityRank target_rank = stk::topology::NODE_RANK; target_rank < get_meta_data().entity_rank_count(); ++target_rank) {
        unsigned const num_valid_conn = get_bulk_data().count_valid_connectivity(source_entity, target_rank);

        if (num_valid_conn > 0) {
          stk::mesh::Entity const* relations = get_bulk_data().begin(source_entity, target_rank);

          size_t const num_relations = get_bulk_data().num_connectivity(source_entity, target_rank);

          stk::mesh::ConnectivityOrdinal const* ords = get_bulk_data().begin_ordinals(source_entity, target_rank);

          for (size_t j = 0; j < num_relations; ++j) {
            stk::mesh::Entity target_entity = relations[j];

            bool is_valid_target_rank = false;

            OutputType const output_type = get_output_type();

            switch (output_type) {
              default:
                std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
                std::cerr << '\n';
                std::cerr << "Invalid output type: ";
                std::cerr << output_type;
                std::cerr << '\n';
                exit(1);
                break;

              case UNIDIRECTIONAL_UNILEVEL: is_valid_target_rank = target_rank + 1 == rank; break;

              case UNIDIRECTIONAL_MULTILEVEL: is_valid_target_rank = target_rank < rank; break;

              case BIDIRECTIONAL_UNILEVEL: is_valid_target_rank = (target_rank == rank + 1) || (target_rank + 1 == rank); break;

              case BIDIRECTIONAL_MULTILEVEL: is_valid_target_rank = target_rank != rank; break;
            }

            if (is_valid_target_rank == false) continue;

            EntityPair entity_pair = std::make_pair(source_entity, target_entity);

            EdgeId const edge_id = ords[j];

            relation_list.push_back(entity_pair);
            relation_local_id.push_back(edge_id);
          }
        }
      }
    }
  }

  // Relations (graph edges)
  for (RelationList::size_type i = 0; i < relation_list.size(); ++i) {
    EntityPair entity_pair = relation_list[i];

    stk::mesh::Entity source = entity_pair.first;

    stk::mesh::Entity target = entity_pair.second;

    gviz_out << dot_relation(
        get_entity_id(source), get_bulk_data().entity_rank(source), get_entity_id(target), get_bulk_data().entity_rank(target), relation_local_id[i]);
  }

  // File end
  gviz_out << dot_footer();

  gviz_out.close();

  return;
}

void
Topology::initializeTopologies()
{
  auto const dimension = get_space_dimension();
  auto const cell_rank = stk::topology::ELEMENT_RANK;
  auto const node_rank = stk::topology::NODE_RANK;

  for (auto rank = node_rank; rank <= cell_rank; ++rank) {
    if (rank > dimension) break;
    std::vector<stk::mesh::Bucket*> buckets = get_bulk_data().buckets(rank);
    stk::mesh::Bucket const&        bucket  = *(buckets[0]);
    topologies_.push_back(bucket.topology());
  }
  return;
}

// Place the entity in the root part that has the stk::topology
// associated with the given rank.
void
Topology::AssignTopology(stk::mesh::EntityRank const rank, stk::mesh::Entity const entity)
{
  stk::topology stk_topology = get_topology().get_rank_topology(rank);

  shards::CellTopology cell_topology = stk::mesh::get_cell_topology(stk_topology);

  stk::mesh::Part& part = get_meta_data().get_topology_root_part(stk_topology);

  stk::mesh::PartVector add_parts;

  add_parts.push_back(&part);

  get_bulk_data().change_entity_parts(entity, add_parts);

  return;
}

// \brief This returns the number of entities of a given rank
EntityVectorIndex
Topology::get_num_entities(stk::mesh::EntityRank const entity_rank)
{
  std::vector<stk::mesh::Bucket*> buckets = get_bulk_data().buckets(entity_rank);

  EntityVectorIndex number_entities = 0;

  for (EntityVectorIndex i = 0; i < buckets.size(); ++i) {
    number_entities += buckets[i]->size();
  }

  return number_entities;
}

// \brief Determine highest id number for each entity rank.
// Used to assign unique ids to newly created entities
void
Topology::initializeHighestIds()
{
  size_t const dimension = get_space_dimension();

  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank <= stk::topology::ELEMENT_RANK; ++rank) {
    if (rank > dimension) break;

    highest_ids_.push_back(get_num_entities(rank));
  }

  return;
}

stk::mesh::EntityId
Topology::get_highest_id(stk::mesh::EntityRank const rank)
{
  return highest_ids_[rank];
}

// Set failure state.
void
Topology::set_failure_state_0(stk::mesh::Entity e, FailureState const fs)
{
  auto&      bulk_data                       = get_bulk_data();
  auto const rank                            = bulk_data.entity_rank(e);
  auto&      failure_field                   = get_failure_state_field(rank);
  *(stk::mesh::field_data(failure_field, e)) = static_cast<int>(fs);
}

void
Topology::set_failure_state(stk::mesh::Entity e, FailureState const fs)
{
  auto&                bulk_data             = get_bulk_data();
  auto const           rank                  = bulk_data.entity_rank(e);
  stk::mesh::MetaData& meta_data             = get_meta_data();
  ScalarFieldType&     failure_field         = *meta_data.get_field<ScalarFieldType>(rank, "failure_state");
  *(stk::mesh::field_data(failure_field, e)) = static_cast<int>(fs);
}

// Get failure state.
FailureState
Topology::get_failure_state_0(stk::mesh::Entity e)
{
  auto& bulk_data     = get_bulk_data();
  auto  rank          = bulk_data.entity_rank(e);
  auto& failure_field = get_failure_state_field(rank);
  return static_cast<FailureState>(*(stk::mesh::field_data(failure_field, e)));
}

FailureState
Topology::get_entity_failure_state(stk::mesh::Entity e)
{
  auto&                bulk_data     = get_bulk_data();
  auto const           rank          = bulk_data.entity_rank(e);
  stk::mesh::MetaData& meta_data     = get_meta_data();
  ScalarFieldType&     failure_field = *meta_data.get_field<ScalarFieldType>(rank, "failure_state");
  return static_cast<FailureState>(*(stk::mesh::field_data(failure_field, e)));
}

// Set boundary indicators.
void
Topology::set_cell_boundary_indicator(stk::mesh::Entity e, BoundaryIndicator const bi)
{
  auto& bulk_data               = get_bulk_data();
  auto& boundary_field          = get_cell_boundary_indicator_field();
  auto* pbi                     = stk::mesh::field_data(boundary_field, e);
  auto* cell_boundary_indicator = static_cast<double*>(pbi);
  ALBANY_ASSERT(cell_boundary_indicator != nullptr);
  *cell_boundary_indicator = static_cast<double>(bi);
}

void
Topology::set_face_boundary_indicator(stk::mesh::Entity e, BoundaryIndicator const bi)
{
  auto& bulk_data               = get_bulk_data();
  auto& boundary_field          = get_face_boundary_indicator_field();
  auto* pbi                     = stk::mesh::field_data(boundary_field, e);
  auto* face_boundary_indicator = static_cast<double*>(pbi);
  ALBANY_ASSERT(face_boundary_indicator != nullptr);
  *face_boundary_indicator = static_cast<double>(bi);
}

void
Topology::set_edge_boundary_indicator(stk::mesh::Entity e, BoundaryIndicator const bi)
{
  auto& bulk_data               = get_bulk_data();
  auto& boundary_field          = get_edge_boundary_indicator_field();
  auto* pbi                     = stk::mesh::field_data(boundary_field, e);
  auto* edge_boundary_indicator = static_cast<double*>(pbi);
  ALBANY_ASSERT(edge_boundary_indicator != nullptr);
  *edge_boundary_indicator = static_cast<double>(bi);
}

void
Topology::set_node_boundary_indicator(stk::mesh::Entity e, BoundaryIndicator const bi)
{
  auto& bulk_data               = get_bulk_data();
  auto& boundary_field          = get_node_boundary_indicator_field();
  auto* pbi                     = stk::mesh::field_data(boundary_field, e);
  auto* node_boundary_indicator = static_cast<double*>(pbi);
  ALBANY_ASSERT(node_boundary_indicator != nullptr);
  *node_boundary_indicator = static_cast<double>(bi);
}

// Get boundary indicators.
BoundaryIndicator
Topology::get_cell_boundary_indicator(stk::mesh::Entity e)
{
  auto& bulk_data               = get_bulk_data();
  auto& boundary_field          = get_cell_boundary_indicator_field();
  auto* pbi                     = stk::mesh::field_data(boundary_field, e);
  auto* cell_boundary_indicator = static_cast<double*>(pbi);
  ALBANY_ASSERT(cell_boundary_indicator != nullptr);
  return static_cast<BoundaryIndicator>(*cell_boundary_indicator);
}

BoundaryIndicator
Topology::get_face_boundary_indicator(stk::mesh::Entity e)
{
  auto& bulk_data               = get_bulk_data();
  auto& boundary_field          = get_face_boundary_indicator_field();
  auto* pbi                     = stk::mesh::field_data(boundary_field, e);
  auto* face_boundary_indicator = static_cast<double*>(pbi);
  ALBANY_ASSERT(face_boundary_indicator != nullptr);
  return static_cast<BoundaryIndicator>(*face_boundary_indicator);
}

BoundaryIndicator
Topology::get_edge_boundary_indicator(stk::mesh::Entity e)
{
  auto& bulk_data               = get_bulk_data();
  auto& boundary_field          = get_edge_boundary_indicator_field();
  auto* pbi                     = stk::mesh::field_data(boundary_field, e);
  auto* edge_boundary_indicator = static_cast<double*>(pbi);
  ALBANY_ASSERT(edge_boundary_indicator != nullptr);
  return static_cast<BoundaryIndicator>(*edge_boundary_indicator);
}

BoundaryIndicator
Topology::get_node_boundary_indicator(stk::mesh::Entity e)
{
  auto& bulk_data               = get_bulk_data();
  auto& boundary_field          = get_node_boundary_indicator_field();
  auto* pbi                     = stk::mesh::field_data(boundary_field, e);
  auto* node_boundary_indicator = static_cast<double*>(pbi);
  ALBANY_ASSERT(node_boundary_indicator != nullptr);
  return static_cast<BoundaryIndicator>(*node_boundary_indicator);
}

bool
Topology::is_internal_face(stk::mesh::Entity e)
{
  assert(get_bulk_data().entity_rank(e) == get_boundary_rank());
  auto iter = boundary_.find(e);
  return iter == boundary_.end();
}

bool
Topology::is_external_face(stk::mesh::Entity e)
{
  return is_internal_face(e) == false;
}

bool
Topology::is_boundary_cell(stk::mesh::Entity e)
{
  stk::mesh::EntityRank const cell_rank = stk::topology::ELEMENT_RANK;
  stk::mesh::EntityRank const face_rank = get_boundary_rank();
  auto&                       bulk_data = get_bulk_data();
  assert(bulk_data.entity_rank(e) == cell_rank);
  stk::mesh::Entity const* relations     = bulk_data.begin(e, face_rank);
  size_t const             num_relations = bulk_data.num_connectivity(e, face_rank);
  for (size_t i = 0; i < num_relations; ++i) {
    stk::mesh::Entity face_entity = relations[i];
    if (is_boundary_face(face_entity) == true) return true;
  }
  return false;
}

bool
Topology::is_boundary_face(stk::mesh::Entity e)
{
  stk::mesh::EntityRank const face_rank = stk::topology::FACE_RANK;
  auto&                       bulk_data = get_bulk_data();
  assert(bulk_data.entity_rank(e) == face_rank);
  return is_external_face(e);
}

bool
Topology::is_boundary_edge(stk::mesh::Entity e)
{
  stk::mesh::EntityRank const edge_rank = stk::topology::EDGE_RANK;
  stk::mesh::EntityRank const face_rank = get_boundary_rank();
  auto&                       bulk_data = get_bulk_data();
  assert(bulk_data.entity_rank(e) == edge_rank);
  stk::mesh::Entity const* relations     = bulk_data.begin(e, face_rank);
  size_t const             num_relations = bulk_data.num_connectivity(e, face_rank);
  for (size_t i = 0; i < num_relations; ++i) {
    stk::mesh::Entity face_entity = relations[i];
    if (is_boundary_face(face_entity) == true) return true;
  }
  return false;
}

bool
Topology::is_boundary_node(stk::mesh::Entity e)
{
  stk::mesh::EntityRank const node_rank = stk::topology::NODE_RANK;
  stk::mesh::EntityRank const face_rank = get_boundary_rank();
  auto&                       bulk_data = get_bulk_data();
  assert(bulk_data.entity_rank(e) == node_rank);
  stk::mesh::Entity const* relations     = bulk_data.begin(e, face_rank);
  size_t const             num_relations = bulk_data.num_connectivity(e, face_rank);
  for (size_t i = 0; i < num_relations; ++i) {
    stk::mesh::Entity face_entity = relations[i];
    if (is_boundary_face(face_entity) == true) return true;
  }
  return false;
}

bool
Topology::is_erodible_cell(stk::mesh::Entity e)
{
  stk::mesh::EntityRank const cell_rank = stk::topology::ELEMENT_RANK;
  stk::mesh::EntityRank const face_rank = get_boundary_rank();
  auto&                       bulk_data = get_bulk_data();
  assert(bulk_data.entity_rank(e) == cell_rank);
  stk::mesh::Entity const* relations     = bulk_data.begin(e, face_rank);
  size_t const             num_relations = bulk_data.num_connectivity(e, face_rank);
  for (size_t i = 0; i < num_relations; ++i) {
    stk::mesh::Entity face_entity = relations[i];
    if (is_erodible_face(face_entity) == true) return true;
  }
  return false;
}

namespace {

bool
all_aligned(std::vector<minitensor::Vector<double, 3>> const& points, double const plane, int const component)
{
  double const eps = 1e-4;
  for (auto&& point : points) {
    if (std::abs(point(component) - plane) > eps) return false;
  }
  return true;
}

}  // anonymous namespace

bool
Topology::is_erodible_face(stk::mesh::Entity face)
{
  if (is_internal_face(face) == true) return false;

  auto const face_rank       = stk::topology::FACE_RANK;
  auto const node_rank       = stk::topology::NODE_RANK;
  auto&      bulk_data       = get_bulk_data();
  auto&      stk_mesh_struct = *(get_stk_mesh_struct());
  auto&      coord_field     = *(stk_mesh_struct.getCoordinatesField());

  ALBANY_ASSERT(bulk_data.entity_rank(face) == face_rank);
  auto const* relations     = bulk_data.begin(face, node_rank);
  auto const  num_relations = bulk_data.num_connectivity(face, node_rank);
  ALBANY_ASSERT(num_relations > 0);
  std::vector<minitensor::Vector<double, 3>> points;
  for (auto i = 0; i < num_relations; ++i) {
    auto  node  = relations[i];
    auto* pc    = stk::mesh::field_data(coord_field, node);
    auto  point = minitensor::Vector<double, 3>(pc[0], pc[1], pc[2]);
    points.emplace_back(point);
  }
  if (all_aligned(points, xm_, 0) == true) return false;
  if (all_aligned(points, ym_, 1) == true) return false;
  if (all_aligned(points, yp_, 1) == true) return false;
  if (all_aligned(points, zm_, 2) == true) return false;
  if (all_aligned(points, zp_, 2) == true) return false;
#if defined(DEBUG)
  std::cout << "ERODIBLE FACE ENTITY ID: " << face << ", NODE ENTITY IDS, GIDS, COORDS :\n";
  for (auto i = 0; i < num_relations; ++i) {
    auto node  = relations[i];
    auto point = points[i];
    std::cout << node << ", " << get_gid(node) << ", " << point << "\n";
  }
#endif
  return true;
}

bool
Topology::is_erodible_edge(stk::mesh::Entity e)
{
  stk::mesh::EntityRank const edge_rank = stk::topology::EDGE_RANK;
  stk::mesh::EntityRank const face_rank = get_boundary_rank();
  auto&                       bulk_data = get_bulk_data();
  assert(bulk_data.entity_rank(e) == edge_rank);
  stk::mesh::Entity const* relations     = bulk_data.begin(e, face_rank);
  size_t const             num_relations = bulk_data.num_connectivity(e, face_rank);
  for (size_t i = 0; i < num_relations; ++i) {
    stk::mesh::Entity face_entity = relations[i];
    if (is_erodible_face(face_entity) == true) return true;
  }
  return false;
}

bool
Topology::is_erodible_node(stk::mesh::Entity e)
{
  stk::mesh::EntityRank const node_rank = stk::topology::NODE_RANK;
  stk::mesh::EntityRank const face_rank = get_boundary_rank();
  auto&                       bulk_data = get_bulk_data();
  assert(bulk_data.entity_rank(e) == node_rank);
  stk::mesh::Entity const* relations     = bulk_data.begin(e, face_rank);
  size_t const             num_relations = bulk_data.num_connectivity(e, face_rank);
  for (size_t i = 0; i < num_relations; ++i) {
    stk::mesh::Entity face_entity = relations[i];
    if (is_erodible_face(face_entity) == true) return true;
  }
  return false;
}

bool
Topology::there_are_failed_cells_global()
{
  int        global_count = 0;
  bool const failed_local = there_are_failed_cells_local() == true;
  int const  local_count  = failed_local == true ? 1 : 0;
  auto       comm         = static_cast<stk::ParallelMachine>(MPI_COMM_WORLD);
  stk::all_reduce_sum(comm, &local_count, &global_count, 1);
  return global_count > 0;
}

bool
Topology::there_are_failed_cells_local()
{
  stk::mesh::EntityRank const cell_rank = stk::topology::ELEMENT_RANK;
  stk::mesh::EntityVector     cells;
  stk::mesh::get_entities(get_bulk_data(), cell_rank, cells);
  for (auto cell : cells) {
    bool const have_failed = failure_criterion_->check(get_bulk_data(), cell);
    if (have_failed == true) return true;
  }
  return false;
}

bool
Topology::there_are_failed_boundary_cells()
{
  stk::mesh::EntityRank const cell_rank = stk::topology::ELEMENT_RANK;
  stk::mesh::EntityVector     cells;
  stk::mesh::get_entities(get_bulk_data(), cell_rank, cells);
  for (auto cell : cells) {
    bool const have_failed = is_failed_boundary_cell(cell);
    if (have_failed == true) return true;
  }
  return false;
}

size_t
Topology::num_connectivity(stk::mesh::Entity e)
{
  size_t                      num{0};
  stk::mesh::EntityRank const node_rank = stk::topology::NODE_RANK;
  stk::mesh::EntityRank const cell_rank = stk::topology::ELEMENT_RANK;
  auto&                       bulk_data = get_bulk_data();
  for (auto rank = node_rank; rank <= cell_rank; ++rank) {
    auto const n = bulk_data.num_connectivity(e, rank);
    num += n;
  }
  return num;
}

double
Topology::erodeElements()
{
  auto const              cell_rank     = stk::topology::ELEMENT_RANK;
  auto&                   bulk_data     = get_bulk_data();
  auto&                   meta_data     = get_meta_data();
  auto&                   locally_owned = meta_data.locally_owned_part();
  double                  eroded_volume = 0.0;
  auto const&             cell_buckets  = bulk_data.buckets(cell_rank);
  stk::mesh::EntityVector cells;
  stk::mesh::EntityVector failed_cells;
  stk::mesh::get_selected_entities(locally_owned, cell_buckets, cells);
  for (auto cell : cells) {
    if (failure_criterion_->check(bulk_data, cell) == true) {
      auto cell_volume = getCellVolume(cell);
      eroded_volume += cell_volume;
      set_failure_state(cell, INTACT);
      failed_cells.emplace_back(cell);
    }
  }
  execute_entity_deletion_operations(failed_cells);
  Albany::fix_node_sharing(bulk_data);
  initializeCellFailureState();
  createBoundary();
  setCellBoundaryIndicator();
  setNodeBoundaryIndicator();
  return eroded_volume;
}

void
Topology::execute_entity_deletion_operations(stk::mesh::EntityVector& entities)
{
  auto&            bulk_data = get_bulk_data();
  auto&            meta_data = get_meta_data();
  std::vector<int> num_local_changes(stk::topology::NUM_RANKS, 0);
  std::vector<int> num_global_changes(stk::topology::NUM_RANKS, 0);
  modification_begin();
  for (auto i = 0; i < entities.size(); ++i) {
    auto entity = entities[i];
    auto rank   = bulk_data.entity_rank(entity);
    ++num_local_changes[rank];
  }
  auto comm = static_cast<stk::ParallelMachine>(MPI_COMM_WORLD);
  stk::all_reduce_sum(comm, num_local_changes.data(), num_global_changes.data(), stk::topology::NUM_RANKS);
  // Must destroy upward relations before destory_entity() will remove
  // the entity.
  for (auto i = 0; i < entities.size(); ++i) {
    auto entity = entities[i];
    ALBANY_ASSERT(bulk_data.is_valid(entity) == true);
    auto const                                  entity_rank = bulk_data.entity_rank(entity);
    stk::mesh::EntityVector                     temp_entities;
    std::vector<stk::mesh::ConnectivityOrdinal> temp_ordinals;
    stk::mesh::Entity const*                    rel_entities = nullptr;
    stk::mesh::ConnectivityOrdinal const*       rel_ordinals;
    int                                         num_conn   = 0;
    auto const                                  end_rank   = static_cast<stk::mesh::EntityRank>(meta_data.entity_rank_count() - 1);
    auto const                                  begin_rank = static_cast<stk::mesh::EntityRank>(entity_rank);
    for (stk::mesh::EntityRank irank = end_rank; irank != begin_rank; --irank) {
      num_conn     = bulk_data.num_connectivity(entity, irank);
      rel_entities = bulk_data.begin(entity, irank);
      rel_ordinals = bulk_data.begin_ordinals(entity, irank);
      for (auto j = num_conn - 1; j >= 0; --j) {
        ALBANY_ASSERT(bulk_data.is_valid(rel_entities[j]) == true);
        bulk_data.destroy_relation(rel_entities[j], entity, rel_ordinals[j]);
      }
    }
    bool successfully_destroyed = bulk_data.destroy_entity(entity);
    ALBANY_ASSERT(successfully_destroyed == true);
  }
  modification_end();
}

}  // namespace LCM
