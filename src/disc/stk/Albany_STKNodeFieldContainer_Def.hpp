// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <stk_io/IossBridge.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/parallel/Parallel.hpp>

#include "Albany_STKNodeFieldContainer.hpp"
#include "Shards_Array.hpp"

namespace Albany {

Teuchos::RCP<AbstractNodeFieldContainer>
buildSTKNodeField(
    std::string const&                             name,
    std::vector<PHX::DataLayout::size_type> const& dim,
    const Teuchos::RCP<stk::mesh::MetaData>&       metaData,
    bool const                                     output)
{
  Teuchos::RCP<AbstractNodeFieldContainer> nfc;
  switch (dim.size()) {
    case 1:  // scalar
      nfc = Teuchos::rcp(new STKNodeField<double, 1>(name, dim, metaData, output));
      break;

    case 2:  // vector
      nfc = Teuchos::rcp(new STKNodeField<double, 2>(name, dim, metaData, output));
      break;

    case 3:  // tensor
      nfc = Teuchos::rcp(new STKNodeField<double, 3>(name, dim, metaData, output));
      break;

    default: ALBANY_PANIC(true, "Error: unexpected argument for dimension");
  }
  return nfc;
}

template <typename DataType, unsigned ArrayDim, class traits>
STKNodeField<DataType, ArrayDim, traits>::STKNodeField(
    std::string const&                             name_,
    std::vector<PHX::DataLayout::size_type> const& dims_,
    const Teuchos::RCP<stk::mesh::MetaData>&       metaData_,
    bool const                                     output)
    : name(name_), dims(dims_), metaData(metaData_)
{
  // amb-leak Look into this later.
  node_field = traits_type::createField(name, dims, metaData_.get());

  if (output) {
    stk::io::set_field_role(*node_field, Ioss::Field::TRANSIENT);
  }
}

template <typename DataType, unsigned ArrayDim, class traits>
void
STKNodeField<DataType, ArrayDim, traits>::saveFieldVector(const Teuchos::RCP<const Thyra_MultiVector>& mv, int offset)
{
  // Iterate over the processor-visible nodes
  const stk::mesh::Selector select_owned_or_shared = metaData->locally_owned_part() | metaData->globally_shared_part();

  // Iterate over the overlap nodes by getting node buckets and iterating over
  // each bucket.
  stk::mesh::BulkData&           mesh         = node_field->get_mesh();
  const stk::mesh::BucketVector& all_elements = mesh.get_buckets(stk::topology::NODE_RANK, select_owned_or_shared);

  traits_type::saveFieldData(mv, all_elements, node_field, offset);
}

template <typename DataType, unsigned ArrayDim, class traits>
MDArray
STKNodeField<DataType, ArrayDim, traits>::getMDA(const stk::mesh::Bucket& buck)
{
  BucketArray<field_type> array(*node_field, buck);
  return array;
}

}  // namespace Albany
