// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

///
/// Fracture criteria classes are required to have a method
/// called check that takes as argument an entity and returns a bool.
///

#if !defined(LCM_Topology_FailureCriterion_hpp)
#define LCM_Topology_FailureCriterion_hpp

#include <cassert>
#include <stk_mesh/base/FieldBase.hpp>

#include "Teuchos_ScalarTraits.hpp"
#include "Topology.hpp"
#include "Topology_Types.hpp"
#include "Topology_Utils.hpp"

namespace LCM {

///
/// Useful to distinguish among different partitioning schemes.
///
namespace fracture {

enum Criterion
{
  UNKNOWN,
  ONE,
  RANDOM,
  TRACTION
};

}

///
/// Base class for fracture criteria
///
class AbstractFailureCriterion
{
 public:
  AbstractFailureCriterion(Topology& topology) : topology_(topology) {}

  virtual bool
  check(stk::mesh::BulkData& mesh, stk::mesh::Entity interface) = 0;

  virtual ~AbstractFailureCriterion() {}

  Topology&
  get_topology()
  {
    return topology_;
  }

  std::string const&
  get_bulk_block_name()
  {
    return get_topology().get_bulk_block_name();
  }

  std::string const&
  get_interface_block_name()
  {
    return get_topology().get_interface_block_name();
  }

  Albany::STKDiscretization&
  get_stk_discretization()
  {
    return get_topology().get_stk_discretization();
  }

  Albany::AbstractSTKMeshStruct const&
  get_stk_mesh_struct()
  {
    return *(get_topology().get_stk_mesh_struct());
  }

  stk::mesh::BulkData const&
  get_bulk_data()
  {
    return get_topology().get_bulk_data();
  }

  stk::mesh::MetaData const&
  get_meta_data()
  {
    return get_topology().get_meta_data();
  }

  minitensor::Index
  get_space_dimension()
  {
    return get_topology().get_space_dimension();
  }

  stk::mesh::Part&
  get_bulk_part()
  {
    return get_topology().get_bulk_part();
  }

  stk::mesh::Part&
  get_interface_part()
  {
    return get_topology().get_interface_part();
  }

  shards::CellTopology
  get_cell_topology()
  {
    return get_topology().get_cell_topology();
  }

  AbstractFailureCriterion()                                = delete;
  AbstractFailureCriterion(const AbstractFailureCriterion&) = delete;
  AbstractFailureCriterion&
  operator=(const AbstractFailureCriterion&) = delete;

 protected:
  Topology& topology_;
};

///
/// Random fracture criterion given a probability of failure
///
class FractureCriterionRandom : public AbstractFailureCriterion
{
 public:
  FractureCriterionRandom(Topology& topology, double const probability) : AbstractFailureCriterion(topology), probability_(probability) {}

  bool
  check(stk::mesh::BulkData& bulk_data, stk::mesh::Entity interface)
  {
    stk::mesh::EntityRank const rank = bulk_data.entity_rank(interface);

    stk::mesh::EntityRank const rank_up = static_cast<stk::mesh::EntityRank>(rank + 1);

    size_t const num_connected = bulk_data.num_connectivity(interface, rank_up);

    assert(num_connected == 2);

    double const random = 0.5 * Teuchos::ScalarTraits<double>::random() + 0.5;

    return random < probability_;
  }

  FractureCriterionRandom()                               = delete;
  FractureCriterionRandom(FractureCriterionRandom const&) = delete;
  FractureCriterionRandom&
  operator=(FractureCriterionRandom const&) = delete;

 private:
  double probability_;
};

///
/// Fracture criterion that open only once (for debugging)
///
class FractureCriterionOnce : public AbstractFailureCriterion
{
 public:
  FractureCriterionOnce(Topology& topology, double const probability) : AbstractFailureCriterion(topology), probability_(probability), open_(true) {}

  bool
  check(stk::mesh::BulkData& bulk_data, stk::mesh::Entity interface)
  {
    stk::mesh::EntityRank const rank = bulk_data.entity_rank(interface);

    stk::mesh::EntityRank const rank_up = static_cast<stk::mesh::EntityRank>(rank + 1);

    size_t const num_connected = bulk_data.num_connectivity(interface, rank_up);

    assert(num_connected == 2);

    double const random = 0.5 * Teuchos::ScalarTraits<double>::random() + 0.5;

    bool const is_open = random < probability_ && open_;

    if (is_open == true) open_ = false;

    return is_open;
  }

  FractureCriterionOnce()                             = delete;
  FractureCriterionOnce(FractureCriterionOnce const&) = delete;
  FractureCriterionOnce&
  operator=(FractureCriterionOnce const&) = delete;

 private:
  double probability_;

  bool open_;
};

///
/// Traction fracture criterion
///
class FractureCriterionTraction : public AbstractFailureCriterion
{
 public:
  FractureCriterionTraction(Topology& topology, std::string const& stress_name, double const critical_traction, double const beta);

  bool
  check(stk::mesh::BulkData& bulk_data, stk::mesh::Entity interface);

  FractureCriterionTraction()                                 = delete;
  FractureCriterionTraction(FractureCriterionTraction const&) = delete;
  FractureCriterionTraction&
  operator=(FractureCriterionTraction const&) = delete;

 private:
  minitensor::Vector<double> const&
  getNormal(stk::mesh::EntityId const entity_id);

  void
  computeNormals();

 private:
  TensorFieldType const* const stress_field_;

  double critical_traction_;

  double beta_;

  std::map<stk::mesh::EntityId, minitensor::Vector<double>> normals_;
};

///
/// Bulk fracture criterion
///
class BulkFailureCriterion : public AbstractFailureCriterion
{
 public:
  BulkFailureCriterion(Topology& topology, std::string const& failure_state_name);

  bool
  check(stk::mesh::BulkData& bulk_data, stk::mesh::Entity element);

  BulkFailureCriterion()                            = delete;
  BulkFailureCriterion(BulkFailureCriterion const&) = delete;
  BulkFailureCriterion&
  operator=(BulkFailureCriterion const&) = delete;

  bool      accumulate{false};
  int const failed_threshold{8};
  int       count_displacement{0};
  int       count_angle{0};
  int       count_yield{0};
  int       count_strain{0};
  int       count_tension{0};

 private:
  ScalarFieldType const* failure_state_{nullptr};
  std::string            failure_state_name_{""};
};
}  // namespace LCM

#endif  // LCM_Topology_FailureCriterion_hpp
