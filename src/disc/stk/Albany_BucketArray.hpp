// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_BUCKETARRAY_HPP
#define ALBANY_BUCKETARRAY_HPP

#include <Shards_Array.hpp>
#include <cassert>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>

namespace Albany {

struct EntityDimension : public shards::ArrayDimTag
{
  char const*
  name() const
  {
    static char const n[] = "EntityDimension";
    return n;
  }

  static const EntityDimension&
  tag()  ///< Singleton
  {
    static const EntityDimension self;
    return self;
  }

 private:
  EntityDimension() {}
  EntityDimension(const EntityDimension&);
  EntityDimension&
  operator=(const EntityDimension&);
};

// With simple fields, all STK fields are Field<double> (or Field<int>, etc.).
// BucketArray provides a shards::Array view of the field data for a bucket.
// It determines the array shape from field extents at runtime.
//
// The old tagged-field BucketArray produced FortranOrder arrays:
//   Scalar Field<double>:                  1D (Entities)
//   Vector Field<double, Cartesian>:       2D (Dim, Entities)
//   Tensor Field<double, Cart, Cart>:      3D (Dim, Dim, Entities)
//
// We replicate the same layout using FortranOrder with EntityDimension tags.

template <class FieldType>
class BucketArray
{
 public:
  typedef typename FieldType::value_type scalar_type;
  typedef shards::Array<scalar_type, shards::FortranOrder> array_type;
  typedef typename array_type::size_type size_type;
  typedef EntityDimension Tag;

  BucketArray(const FieldType& f, const stk::mesh::Bucket& k)
    : m_array()
  {
    if (k.field_data_is_allocated(f)) {
      scalar_type* data = stk::mesh::field_data(f, k);
      const size_type num_entities = k.size();
      const size_type scalars_per_entity = stk::mesh::field_scalars_per_entity(f, k);

      if (scalars_per_entity <= 0) {
        return;
      }

      const size_type extent0 = stk::mesh::field_extent0_per_entity(f, k);
      const size_type extent1 = stk::mesh::field_extent1_per_entity(f, k);

      if (scalars_per_entity == 1) {
        // Scalar field: 1D (Entities)
        m_array.template assign<Tag>(data, num_entities);
      } else if (extent1 <= 1) {
        // Vector field: 2D (Dim, Entities) in FortranOrder
        m_array.template assign<Tag, Tag>(data, extent0, num_entities);
      } else {
        // Tensor field: 3D (Dim, Dim, Entities) in FortranOrder
        m_array.template assign<Tag, Tag, Tag>(data, extent0, extent1, num_entities);
      }
    }
  }

  // Allow implicit conversion to the underlying shards::Array
  operator const array_type&() const { return m_array; }
  operator array_type&() { return m_array; }

  // Forward common array methods
  size_type rank() const { return m_array.rank(); }
  size_type size() const { return m_array.size(); }
  size_type dimension(size_type i) const { return m_array.dimension(i); }

  scalar_type& operator()(size_type i) { return m_array(i); }
  scalar_type& operator()(size_type i, size_type j) { return m_array(i, j); }
  scalar_type& operator()(size_type i, size_type j, size_type k) { return m_array(i, j, k); }

  const scalar_type& operator()(size_type i) const { return m_array(i); }
  const scalar_type& operator()(size_type i, size_type j) const { return m_array(i, j); }
  const scalar_type& operator()(size_type i, size_type j, size_type k) const { return m_array(i, j, k); }

 private:
  array_type m_array;

  BucketArray();
  BucketArray(const BucketArray&);
  BucketArray& operator=(const BucketArray&);
};

}  // namespace Albany

#endif  // ALBANY_BUCKETARRAY_HPP
