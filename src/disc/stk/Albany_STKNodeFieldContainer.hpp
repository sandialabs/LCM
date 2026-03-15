// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_STK_NODE_FIELD_CONTAINER_HPP
#define ALBANY_STK_NODE_FIELD_CONTAINER_HPP

#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MetaData.hpp>

#include "Albany_AbstractNodeFieldContainer.hpp"
#include "Albany_BucketArray.hpp"  // for EntityDimension tag
#include <stk_mesh/base/FieldBase.hpp>
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_StateInfoStruct.hpp"  // For MDArray
#include "Albany_ThyraUtils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {

/*!
 * \brief Abstract interface for an STK NodeField container
 *
 */
class AbstractSTKNodeFieldContainer : public AbstractNodeFieldContainer
{
 public:
  AbstractSTKNodeFieldContainer()          = default;
  virtual ~AbstractSTKNodeFieldContainer() = default;

  virtual MDArray
  getMDA(const stk::mesh::Bucket& buck) = 0;
};

Teuchos::RCP<AbstractNodeFieldContainer>
buildSTKNodeField(
    std::string const&                             name,
    std::vector<PHX::DataLayout::size_type> const& dim,
    const Teuchos::RCP<stk::mesh::MetaData>&       metaData,
    bool const                                     output);

// Helper class for NodeData
template <typename DataType, unsigned ArrayDim>
struct NodeData_Traits
{
};

template <typename DataType, unsigned ArrayDim, class traits = NodeData_Traits<DataType, ArrayDim>>
class STKNodeField : public AbstractSTKNodeFieldContainer
{
 public:
  //! Type of traits class being used
  typedef traits traits_type;

  //! Define the field type
  typedef typename traits_type::field_type field_type;

  STKNodeField(
      std::string const&                             name,
      std::vector<PHX::DataLayout::size_type> const& dim,
      const Teuchos::RCP<stk::mesh::MetaData>&       metaData,
      bool const                                     output = false);

  virtual ~STKNodeField() = default;

  void
  saveFieldVector(const Teuchos::RCP<const Thyra_MultiVector>& mv, int offset) override;

  MDArray
  getMDA(const stk::mesh::Bucket& buck) override;

 private:
  std::string                             name;        // Name of data field
  field_type*                             node_field;  // stk::mesh::field
  std::vector<PHX::DataLayout::size_type> dims;
  Teuchos::RCP<stk::mesh::MetaData>       metaData;
};

// Explicit template definitions in support of the above

// Node Scalar
template <typename T>
struct NodeData_Traits<T, 1>
{
  enum
  {
    size = 1
  };  // One array dimension tag (Node), store type T values
  typedef stk::mesh::Field<T> field_type;
  static field_type*
  createField(std::string const& name, std::vector<PHX::DataLayout::size_type> const& /* dim */, stk::mesh::MetaData* metaData)
  {
    field_type* fld = &metaData->declare_field<T>(stk::topology::NODE_RANK, name);
    // Multi-dim order is Fortran Ordering, so reversed here
    stk::mesh::put_field_on_mesh(*fld, metaData->universal_part(), nullptr);

    return fld;  // Address is held by stk
  }

  static void
  saveFieldData(const Teuchos::RCP<const Thyra_MultiVector>& overlap_node_vec, const stk::mesh::BucketVector& all_elements, field_type* fld, int offset)
  {
    Teuchos::ArrayRCP<const ST> const_overlap_node_view = getLocalData(overlap_node_vec->col(offset));

    auto indexer = createGlobalLocalIndexer(overlap_node_vec->range());
    for (auto it = all_elements.begin(); it != all_elements.end(); ++it) {
      const stk::mesh::Bucket&   bucket   = **it;
      const stk::mesh::BulkData& bulkData = bucket.mesh();

      int const num_nodes_in_bucket = bucket.size();

      for (std::size_t i = 0; i < num_nodes_in_bucket; i++) {
        const GO global_id = bulkData.identifier(bucket[i]) - 1;  // global node in mesh
        const LO local_id  = indexer->getLocalElement(global_id);
        T* data = stk::mesh::field_data(*fld, bucket[i]);
        data[0] = const_overlap_node_view[local_id];
      }
    }
  }
};

// Node Vector
template <typename T>
struct NodeData_Traits<T, 2>
{
  enum
  {
    size = 2
  };  // Two array dimension tags (Node, Dim), store type T values
  typedef stk::mesh::Field<T> field_type;
  static field_type*
  createField(std::string const& name, std::vector<PHX::DataLayout::size_type> const& dim, stk::mesh::MetaData* metaData)
  {
    field_type* fld = &metaData->declare_field<T>(stk::topology::NODE_RANK, name);
    // Multi-dim order is Fortran Ordering, so reversed here
    stk::mesh::put_field_on_mesh(*fld, metaData->universal_part(), dim[1], nullptr);

    return fld;  // Address is held by stk
  }

  static void
  saveFieldData(const Teuchos::RCP<const Thyra_MultiVector>& overlap_node_vec, const stk::mesh::BucketVector& all_elements, field_type* fld, int offset)
  {
    auto indexer = createGlobalLocalIndexer(overlap_node_vec->range());
    for (auto it = all_elements.begin(); it != all_elements.end(); ++it) {
      const stk::mesh::Bucket& bucket = **it;
      stk::mesh::BulkData const& bulkData = bucket.mesh();

      int const num_vec_components  = stk::mesh::field_scalars_per_entity(*fld, bucket);
      int const num_nodes_in_bucket = bucket.size();

      for (int j = 0; j < num_vec_components; ++j) {
        Teuchos::ArrayRCP<const ST> const_overlap_node_view = getLocalData(overlap_node_vec->col(offset + j));

        for (int i = 0; i < num_nodes_in_bucket; ++i) {
          const GO global_id = bulkData.identifier(bucket[i]) - 1;  // global node in mesh
          const LO local_id  = indexer->getLocalElement(global_id);
          T* data = stk::mesh::field_data(*fld, bucket[i]);
          data[j] = const_overlap_node_view[local_id];
        }
      }
    }
  }
};

// Node Tensor
template <typename T>
struct NodeData_Traits<T, 3>
{
  enum
  {
    size = 3
  };  // Three array dimension tags (Node, Dim, Dim), store type T values
  typedef stk::mesh::Field<T> field_type;
  static field_type*
  createField(std::string const& name, std::vector<PHX::DataLayout::size_type> const& dim, stk::mesh::MetaData* metaData)
  {
    field_type* fld = &metaData->declare_field<T>(stk::topology::NODE_RANK, name);
    // Multi-dim order is Fortran Ordering, so reversed here
    stk::mesh::put_field_on_mesh(*fld, metaData->universal_part(), dim[2], dim[1], nullptr);

    return fld;  // Address is held by stk
  }

  static void
  saveFieldData(const Teuchos::RCP<const Thyra_MultiVector>& overlap_node_vec, const stk::mesh::BucketVector& all_elements, field_type* fld, int offset)
  {
    auto indexer = createGlobalLocalIndexer(overlap_node_vec->range());
    for (auto it = all_elements.begin(); it != all_elements.end(); ++it) {
      const stk::mesh::Bucket&   bucket   = **it;
      stk::mesh::BulkData const& bulkData = bucket.mesh();

      int const num_nodes_in_bucket = bucket.size();
      int const scalars_per = stk::mesh::field_scalars_per_entity(*fld, bucket);
      int const num_i_components = stk::mesh::field_extent0_per_entity(*fld, bucket);
      int const num_j_components = (num_i_components > 0) ? scalars_per / num_i_components : 1;

      for (int j = 0; j < num_j_components; ++j) {
        for (int k = 0; k < num_i_components; ++k) {
          Teuchos::ArrayRCP<const ST> const_overlap_node_view = getLocalData(overlap_node_vec->col(offset + j * num_i_components + k));

          for (int i = 0; i < num_nodes_in_bucket; ++i) {
            const GO global_id = bulkData.identifier(bucket[i]) - 1;  // global node in mesh
            const LO local_id  = indexer->getLocalElement(global_id);
            T* data = stk::mesh::field_data(*fld, bucket[i]);
            data[k + j * num_i_components] = const_overlap_node_view[local_id];
          }
        }
      }
    }
  }
};

}  // namespace Albany

#endif  // ALBANY_STK_NODE_FIELD_CONTAINER_HPP
