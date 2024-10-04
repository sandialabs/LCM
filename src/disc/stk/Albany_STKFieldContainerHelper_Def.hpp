// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <stk_mesh/base/GetBuckets.hpp>

#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Albany_BucketArray.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_Macros.hpp"
#include "Albany_STKFieldContainerHelper.hpp"
#include "Albany_ThyraUtils.hpp"

namespace Albany {

// Get the rank of a field
template <typename FieldType>
constexpr int
getRank()
{
  return std::is_same<FieldType, AbstractSTKFieldContainer::ScalarFieldType>::value ?
             0 :
             (std::is_same<FieldType, AbstractSTKFieldContainer::VectorFieldType>::value ?
                  1 :
                  (std::is_same<FieldType, AbstractSTKFieldContainer::TensorFieldType>::value ? 2 : -1));
}

// Fill the result vector
// Create a multidimensional array view of the
// solution field data for this bucket of nodes.
// The array is two dimensional ( Cartesian X NumberNodes )
// and indexed by ( 0..2 , 0..NumberNodes-1 )

template <class FieldType>
void
STKFieldContainerHelper<FieldType>::fillVector(
    Thyra_Vector&                                 field_thyra,
    const FieldType&                              field_stk,
    const Teuchos::RCP<const GlobalLocalIndexer>& indexer,
    const stk::mesh::Bucket&                      bucket,
    const NodalDOFManager&                        nodalDofManager,
    int const                                     offset)
{
  constexpr int rank = getRank<FieldType>();
  ALBANY_PANIC(rank != 0 && rank != 1, "Error! Can only handle ScalarFieldType and VectorFieldType for now.\n");

  BucketArray<FieldType> field_array(field_stk, bucket);

  using SFT                = AbstractSTKFieldContainer::ScalarFieldType;
  constexpr bool is_SFT    = std::is_same<FieldType, SFT>::value;
  constexpr int  nodes_dim = is_SFT ? 0 : 1;

  int const num_nodes_in_bucket = field_array.dimension(nodes_dim);

  const stk::mesh::BulkData& mesh = field_stk.get_mesh();
  auto                       data = getNonconstLocalData(field_thyra);
  int                        num_vec_components;
  num_vec_components = nodalDofManager.numComponents();

  for (int i = 0; i < num_nodes_in_bucket; ++i) {
    const GO node_gid = mesh.identifier(bucket[i]) - 1;
    const LO node_lid = indexer->getLocalElement(node_gid);
    auto stk_data = stk::mesh::field_data(field_stk,bucket[i]);
    auto nn = stk::mesh::field_scalars_per_entity(field_stk, bucket[i]); 
    if (nn == 1) 
      num_vec_components = nn; 

    for (int j = 0; j < num_vec_components; ++j) {
      data[nodalDofManager.getLocalDOF(node_lid, offset + j)] = stk_data[j];

    }
  }
}

template <class FieldType>
void
STKFieldContainerHelper<FieldType>::saveVector(
    Thyra_Vector const&                           field_thyra,
    FieldType&                                    field_stk,
    const Teuchos::RCP<const GlobalLocalIndexer>& indexer,
    const stk::mesh::Bucket&                      bucket,
    const NodalDOFManager&                        nodalDofManager,
    int const                                     offset)
{
  constexpr int rank = getRank<FieldType>();
  ALBANY_PANIC(rank != 0 && rank != 1, "Error! Can only handle ScalarFieldType and VectorFieldType for now.\n");

  BucketArray<FieldType> field_array(field_stk, bucket);

  using SFT                = AbstractSTKFieldContainer::ScalarFieldType;
  constexpr bool is_SFT    = std::is_same<FieldType, SFT>::value;
  constexpr int  nodes_dim = is_SFT ? 0 : 1;

  int const num_nodes_in_bucket = field_array.dimension(nodes_dim);

  const stk::mesh::BulkData& mesh = field_stk.get_mesh();
  auto                       data = getLocalData(field_thyra);
  int                        num_vec_components;
  num_vec_components = nodalDofManager.numComponents();

  for (int i = 0; i < num_nodes_in_bucket; ++i) {
    const GO node_gid = mesh.identifier(bucket[i]) - 1;
    const LO node_lid = indexer->getLocalElement(node_gid);
    auto stk_data = stk::mesh::field_data(field_stk,bucket[i]);
    auto nn = stk::mesh::field_scalars_per_entity(field_stk, bucket[i]); 
    if (nn == 1) 
      num_vec_components = nn; 

    for (int j = 0; j < num_vec_components; ++j) {
      stk_data[j] = data[nodalDofManager.getLocalDOF(node_lid, offset + j)];
    }
  }
}

template <class FieldType>
void
STKFieldContainerHelper<FieldType>::copySTKField(const FieldType& source, FieldType& target)
{
  constexpr int rank = getRank<FieldType>();
  ALBANY_PANIC(rank != 0 && rank != 1, "Error! Can only handle ScalarFieldType and VectorFieldType for now.\n");

  const stk::mesh::BulkData&     mesh = source.get_mesh();
  const stk::mesh::BucketVector& bv   = mesh.buckets(stk::topology::NODE_RANK);

  using SFT                = AbstractSTKFieldContainer::ScalarFieldType;
  constexpr bool is_SFT    = std::is_same<FieldType, SFT>::value;
  constexpr int  nodes_dim = is_SFT ? 0 : 1;

  for (stk::mesh::BucketVector::const_iterator it = bv.begin(); it != bv.end(); ++it) {
    const stk::mesh::Bucket& bucket = **it;

    BucketArray<FieldType> source_array(source, bucket);
    BucketArray<FieldType> target_array(target, bucket);

    int num_source_components = source_array.dimension(0);
    int num_target_components = target_array.dimension(0);

    int const num_nodes_in_bucket   = source_array.dimension(nodes_dim);

    int const uneven_downsampling = num_source_components % num_target_components;

    ALBANY_PANIC(
        (uneven_downsampling) || (num_nodes_in_bucket != target_array.dimension(nodes_dim)),
        "Error in stk fields: specification of coordinate vector vs. solution "
        "layout is incorrect."
            << std::endl);

    for (int i = 0; i < num_nodes_in_bucket; ++i) {
      // In source, j varies over neq (num phys vectors * numDim)
      // We want target to only vary over the first numDim components
      // Not sure how to do this generally...
      auto source_stk_data = stk::mesh::field_data(source,bucket[i]);
      auto target_stk_data = stk::mesh::field_data(target,bucket[i]);
      auto nn = stk::mesh::field_scalars_per_entity(source, bucket[i]); 
      if (nn == 1) 
        num_source_components = nn;
      nn = stk::mesh::field_scalars_per_entity(target, bucket[i]); 
      if (nn == 1) 
        num_target_components = nn;

      for (int j = 0; j < num_target_components; ++j) {
	target_stk_data[j] = source_stk_data[j]; 
      }
    }
  }
}

}  // namespace Albany
