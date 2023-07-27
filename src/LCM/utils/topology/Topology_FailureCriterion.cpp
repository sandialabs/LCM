// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Topology_FailureCriterion.hpp"

#include "Albany_GlobalLocalIndexer.hpp"
#include "Topology.hpp"

namespace LCM {

FractureCriterionTraction::FractureCriterionTraction(Topology& topology, std::string const& stress_name, double const critical_traction, double const beta)
    : AbstractFailureCriterion(topology),
      stress_field_(get_meta_data().get_field<TensorFieldType>(stk::topology::NODE_RANK, stress_name)),
      critical_traction_(critical_traction),
      beta_(beta)
{
  if (stress_field_ == NULL) {
    std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
    std::cerr << '\n';
    std::cerr << "Cannot find field for traction criterion: ";
    std::cerr << stress_name;
    std::cerr << '\n';
    exit(1);
  }
  computeNormals();
}

bool
FractureCriterionTraction::check(stk::mesh::BulkData& bulk_data, stk::mesh::Entity interface)
{
  // Check the adjacent bulk elements. Proceed only
  // if both elements belong to the bulk part.
  stk::mesh::EntityRank const rank = bulk_data.entity_rank(interface);

  stk::mesh::EntityRank const rank_up = static_cast<stk::mesh::EntityRank>(rank + 1);

  stk::mesh::Entity const* relations_up = bulk_data.begin(interface, rank_up);

  assert(bulk_data.num_connectivity(interface, rank_up) == 2);

  stk::mesh::Entity element_0 = relations_up[0];

  stk::mesh::Entity element_1 = relations_up[1];

  stk::mesh::Bucket const& bucket_0 = bulk_data.bucket(element_0);

  stk::mesh::Bucket const& bucket_1 = bulk_data.bucket(element_1);

  bool const is_embedded = bucket_0.member(get_bulk_part()) && bucket_1.member(get_bulk_part());

  if (is_embedded == false) return false;

  // Now traction check
  stk::mesh::EntityVector nodes = get_topology().getBoundaryEntityNodes(interface);

  EntityVectorIndex const number_nodes = nodes.size();

  minitensor::Tensor<double> stress(get_space_dimension(), minitensor::Filler::ZEROS);

  minitensor::Tensor<double> nodal_stress(get_space_dimension(), minitensor::Filler::ZEROS);

  // The traction is evaluated at centroid of face, so a simple
  // average yields the value.
  for (EntityVectorIndex i = 0; i < number_nodes; ++i) {
    stk::mesh::Entity node = nodes[i];

    double* const pstress = stk::mesh::field_data(*stress_field_, node);

    nodal_stress.fill(pstress);

    stress += nodal_stress;
  }

  stress = stress / static_cast<double>(number_nodes);

  // Use low level id functions from BulkData instead of the mapping
  // functions for entity ids from the Topology class as the local
  // element mapping functions expect the former.
  stk::mesh::EntityId const face_id = get_bulk_data().identifier(interface);

  minitensor::Vector<double> const& normal = getNormal(face_id);

  minitensor::Vector<double> const traction = stress * normal;

  double t_n = minitensor::dot(traction, normal);

  minitensor::Vector<double> const traction_normal = t_n * normal;

  minitensor::Vector<double> const traction_shear = traction - traction_normal;

  double const t_s = minitensor::norm(traction_shear);

  // Ignore compression
  t_n = std::max(t_n, 0.0);

  double const effective_traction = std::sqrt(t_s * t_s / beta_ / beta_ + t_n * t_n);

  return effective_traction >= critical_traction_;
}

minitensor::Vector<double> const&
FractureCriterionTraction::getNormal(stk::mesh::EntityId const entity_id)
{
  std::map<stk::mesh::EntityId, minitensor::Vector<double>>::const_iterator it = normals_.find(entity_id);

  assert(it != normals_.end());

  return it->second;
}

void
FractureCriterionTraction::computeNormals()
{
  stk::mesh::Selector local_selector = get_meta_data().locally_owned_part();

  std::vector<stk::mesh::Bucket*> const& node_buckets = get_bulk_data().buckets(stk::topology::NODE_RANK);

  stk::mesh::EntityVector nodes;

  stk::mesh::get_selected_entities(local_selector, node_buckets, nodes);

  EntityVectorIndex const number_nodes = nodes.size();

  std::vector<minitensor::Vector<double>> coordinates(number_nodes);

  const Teuchos::ArrayRCP<double>& node_coordinates = get_stk_discretization().getCoordinates();

  for (EntityVectorIndex i = 0; i < number_nodes; ++i) {
    double const* const pointer_coordinates = &(node_coordinates[get_space_dimension() * i]);

    coordinates[i].set_dimension(get_space_dimension());
    coordinates[i].fill(pointer_coordinates);
  }

  std::vector<stk::mesh::Bucket*> const& face_buckets = get_bulk_data().buckets(get_meta_data().side_rank());

  stk::mesh::EntityVector faces;

  stk::mesh::get_selected_entities(local_selector, face_buckets, faces);

  EntityVectorIndex const number_normals = faces.size();

  // Use low level id functions from BulkData instead of the mapping
  // functions for entity ids from the Topology class as the local
  // element mapping functions expect the former.

  auto node_vs_indexer = Albany::createGlobalLocalIndexer(get_stk_discretization().getNodeVectorSpace());
  for (EntityVectorIndex i = 0; i < number_normals; ++i) {
    stk::mesh::Entity face = faces[i];

    stk::mesh::EntityId face_id = get_bulk_data().identifier(face);

    nodes = get_topology().getBoundaryEntityNodes(face);

    minitensor::Vector<double> normal(get_space_dimension());

    // Depending on the dimension is how the normal is computed.
    // TODO: generalize this for all topologies.
    switch (get_space_dimension()) {
      default:
        std::cerr << "ERROR: " << __PRETTY_FUNCTION__ << '\n';
        std::cerr << "Wrong dimension: " << get_space_dimension() << '\n';
        exit(1);
        break;

      case 2: {
        stk::mesh::EntityId const gid0 = get_bulk_data().identifier(nodes[0]) - 1;

        stk::mesh::EntityId const lid0 = node_vs_indexer->getLocalElement(gid0);

        assert(lid0 < number_nodes);

        stk::mesh::EntityId gid1 = get_bulk_data().identifier(nodes[1]) - 1;

        stk::mesh::EntityId const lid1 = node_vs_indexer->getLocalElement(gid1);

        assert(lid1 < number_nodes);

        minitensor::Vector<double> v = coordinates[lid1] - coordinates[lid0];

        normal(0) = -v(1);
        normal(1) = v(0);

        normal = minitensor::unit(normal);
      } break;

      case 3: {
        stk::mesh::EntityId const gid0 = get_bulk_data().identifier(nodes[0]) - 1;

        stk::mesh::EntityId const lid0 = node_vs_indexer->getLocalElement(gid0);

        assert(lid0 < number_nodes);

        stk::mesh::EntityId gid1 = get_bulk_data().identifier(nodes[1]) - 1;

        stk::mesh::EntityId const lid1 = node_vs_indexer->getLocalElement(gid1);

        assert(lid1 < number_nodes);

        stk::mesh::EntityId gid2 = get_bulk_data().identifier(nodes[2]) - 1;

        stk::mesh::EntityId const lid2 = node_vs_indexer->getLocalElement(gid2);

        assert(lid2 < number_nodes);

        minitensor::Vector<double> v1 = coordinates[lid1] - coordinates[lid0];

        minitensor::Vector<double> v2 = coordinates[lid2] - coordinates[lid0];

        normal = minitensor::cross(v1, v2);

        normal = minitensor::unit(normal);
      } break;
    }

    normals_.insert(std::make_pair(face_id, normal));
  }
}

BulkFailureCriterion::BulkFailureCriterion(Topology& topology, std::string const& failure_state_name)
    : AbstractFailureCriterion(topology), failure_state_name_(failure_state_name)
{
}

bool
BulkFailureCriterion::check(stk::mesh::BulkData& /* bulk_data */, stk::mesh::Entity element)
{
  failure_state_ = get_meta_data().get_field<ScalarFieldType>(stk::topology::ELEMENT_RANK, failure_state_name_);
  ALBANY_ASSERT(failure_state_ != nullptr);
  auto const* const pfs              = reinterpret_cast<double const* const>(stk::mesh::field_data(*failure_state_, element));
  auto const        failure_modes    = static_cast<int>(pfs[0]);
  auto              failure_state    = failure_modes;
  auto const        num_displacement = failure_state / 10000;
  failure_state -= 10000 * num_displacement;
  auto const num_angle = failure_state / 1000;
  failure_state -= 1000 * num_angle;
  auto const num_yield = failure_state / 100;
  failure_state -= 100 * num_yield;
  auto const num_strain = failure_state / 10;
  failure_state -= 10 * num_strain;
  auto const num_tension = failure_state;
  auto const num_failed  = num_displacement + num_angle + num_yield + num_strain + num_tension;
  if (accumulate == true) {
    count_displacement += num_displacement;
    count_angle += num_angle;
    count_yield += num_yield;
    count_strain += num_strain;
    count_tension += num_tension;
  }
  return num_failed >= failed_threshold;  // # of integration points that must fail (max is 8 per element)
}

}  // namespace LCM
