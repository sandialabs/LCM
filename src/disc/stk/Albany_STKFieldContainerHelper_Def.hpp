// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_Macros.hpp"
#include "Albany_STKFieldContainerHelper.hpp"
#include "Albany_ThyraUtils.hpp"

#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/EntityValues.hpp>

namespace Albany {

// Fill the result vector
// Create a view of the solution field data for this bucket of nodes.
// Uses the STK Field data API, which applies whatever component stride the
// active memory layout uses, so these loops are correct under both
// Layout::Right and the Layout::Left of a unified-memory build.

void
STKFieldContainerHelper::fillVector(
    Thyra_Vector&                                 field_thyra,
    const FieldType&                              field_stk,
    const Teuchos::RCP<const GlobalLocalIndexer>& indexer,
    const stk::mesh::Bucket&                      bucket,
    const NodalDOFManager&                        nodalDofManager,
    int const                                     offset)
{
  const int num_nodes_in_bucket = bucket.size();
  const int scalars_per_entity = stk::mesh::field_scalars_per_entity(field_stk, bucket);
  const int num_vec_components = (scalars_per_entity <= 1) ? 1 : nodalDofManager.numComponents();

  const stk::mesh::BulkData& mesh = field_stk.get_mesh();
  auto                       data = getNonconstLocalData(field_thyra);

  auto field_data = field_stk.data();

  for (int i = 0; i < num_nodes_in_bucket; ++i) {
    const GO node_gid = mesh.identifier(bucket[i]) - 1;
    const LO node_lid = indexer->getLocalElement(node_gid);
    auto     values   = field_data.entity_values(bucket[i]);

    for (int j = 0; j < num_vec_components; ++j) {
      data[nodalDofManager.getLocalDOF(node_lid, offset + j)] =
          values(stk::mesh::ComponentIdx(j));
    }
  }
}

void
STKFieldContainerHelper::saveVector(
    Thyra_Vector const&                           field_thyra,
    FieldType&                                    field_stk,
    const Teuchos::RCP<const GlobalLocalIndexer>& indexer,
    const stk::mesh::Bucket&                      bucket,
    const NodalDOFManager&                        nodalDofManager,
    int const                                     offset)
{
  const int num_nodes_in_bucket = bucket.size();
  const int scalars_per_entity = stk::mesh::field_scalars_per_entity(field_stk, bucket);
  const int num_vec_components = (scalars_per_entity <= 1) ? 1 : nodalDofManager.numComponents();

  const stk::mesh::BulkData& mesh = field_stk.get_mesh();
  auto                       data = getLocalData(field_thyra);

  auto field_data = field_stk.data<stk::mesh::ReadWrite>();

  for (int i = 0; i < num_nodes_in_bucket; ++i) {
    const GO node_gid = mesh.identifier(bucket[i]) - 1;
    const LO node_lid = indexer->getLocalElement(node_gid);
    auto     values   = field_data.entity_values(bucket[i]);

    for (int j = 0; j < num_vec_components; ++j) {
      values(stk::mesh::ComponentIdx(j)) =
          data[nodalDofManager.getLocalDOF(node_lid, offset + j)];
    }
  }
}

void
STKFieldContainerHelper::copySTKField(const FieldType& source, FieldType& target)
{
  const stk::mesh::BulkData&     mesh = source.get_mesh();
  const stk::mesh::BucketVector& bv   = mesh.buckets(stk::topology::NODE_RANK);

  // Acquired once, outside the bucket loop: these are temporary handles and
  // re-acquiring them per bucket costs a lookup for nothing. Source and target
  // are always distinct fields here, so the ReadOnly and ReadWrite lifetimes
  // may overlap.
  auto src_data = source.data();
  auto tgt_data = target.data<stk::mesh::ReadWrite>();

  for (auto it = bv.begin(); it != bv.end(); ++it) {
    const stk::mesh::Bucket& bucket = **it;

    const int num_nodes_in_bucket   = bucket.size();
    const int num_source_components = stk::mesh::field_scalars_per_entity(source, bucket);
    const int num_target_components = stk::mesh::field_scalars_per_entity(target, bucket);

    const int uneven_downsampling = num_source_components % num_target_components;

    ALBANY_PANIC(
        uneven_downsampling,
        "Error in stk fields: specification of coordinate vector vs. solution "
        "layout is incorrect."
            << std::endl);

    for (int i = 0; i < num_nodes_in_bucket; ++i) {
      auto src_values = src_data.entity_values(bucket[i]);
      auto tgt_values = tgt_data.entity_values(bucket[i]);

      for (int j = 0; j < num_target_components; ++j) {
        tgt_values(stk::mesh::ComponentIdx(j)) =
            src_values(stk::mesh::ComponentIdx(j));
      }
    }
  }
}

}  // namespace Albany
