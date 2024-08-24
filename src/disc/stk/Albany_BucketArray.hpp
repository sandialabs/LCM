// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_BUCKETARRAY_HPP
#define ALBANY_BUCKETARRAY_HPP

#include <Shards_Array.hpp>
#include <cassert>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FindRestriction.hpp>
#include <stk_mesh/base/MetaData.hpp>

// #define IKT_DEBUG

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

template <class FieldType>
struct BucketArray
{
};

/** \brief  \ref stk::mesh::Field "Field" data \ref shards::Array "Array"
 *          for a given scalar field and bucket
 */
template <typename ScalarType>
struct BucketArray<stk::mesh::Field<ScalarType, void, void, void, void, void, void, void>>
    : public shards::Array<ScalarType, shards::FortranOrder, EntityDimension, void, void, void, void, void, void>
{
 private:
  typedef unsigned char* byte_p;
  BucketArray();
  BucketArray(const BucketArray&);
  BucketArray&
  operator=(const BucketArray&);

 public:
  typedef stk::mesh::Field<ScalarType, void, void, void, void, void, void, void>                               field_type;
  typedef shards::Array<ScalarType, shards::FortranOrder, EntityDimension, void, void, void, void, void, void> array_type;

  BucketArray(const field_type& f, const stk::mesh::Bucket& k)
  {
    if (k.field_data_is_allocated(f)) {
      array_type::assign((ScalarType*)(k.field_data_location(f)), k.size());
    }
  }
};

template <typename Tag>
inline size_t
get_size(stk::mesh::Bucket const&)
{
  return Tag::Size;
}

template <>
inline size_t
get_size<void>(stk::mesh::Bucket const&)
{
  return 0;
}

//IKT 8/23/2024: TODO FIXME HACK!  Bring back this routine.
/*template <>
inline size_t
get_size<stk::mesh::Cartesian>(stk::mesh::Bucket const& b)
{
  return b.mesh().mesh_meta_data().spatial_dimension();
}*/

/** \brief  \ref stk::mesh::Field "Field" data \ref shards::Array "Array"
 *          for a given array field and bucket
 */
template <typename ScalarType, class Tag1, class Tag2, class Tag3, class Tag4, class Tag5, class Tag6, class Tag7>
struct BucketArray<stk::mesh::Field<ScalarType, Tag1, Tag2, Tag3, Tag4, Tag5, Tag6, Tag7>>
    : public shards::ArrayAppend<shards::Array<ScalarType, shards::FortranOrder, Tag1, Tag2, Tag3, Tag4, Tag5, Tag6, Tag7>, EntityDimension>::type
{
 private:
  typedef unsigned char* byte_p;
  BucketArray();
  BucketArray(const BucketArray&);
  BucketArray&
  operator=(const BucketArray&);

 public:
  typedef stk::mesh::Field<ScalarType, Tag1, Tag2, Tag3, Tag4, Tag5, Tag6, Tag7> field_type;

  typedef
      typename shards::ArrayAppend<shards::Array<ScalarType, shards::FortranOrder, Tag1, Tag2, Tag3, Tag4, Tag5, Tag6, Tag7>, EntityDimension>::type array_type;

  BucketArray(const field_type& f, const stk::mesh::Bucket& b)
  {
    if (b.field_data_is_allocated(f)) {
      int stride[4];
      if (f.field_array_rank() == 1) {
        stride[0] = stk::mesh::field_scalars_per_entity(f, b);
      } else if (f.field_array_rank() == 2) {
        int dim0  = stk::mesh::find_restriction(f, b.entity_rank(), b.supersets()).dimension();
        stride[0] = dim0;
        stride[1] = stk::mesh::field_scalars_per_entity(f, b);
      } else if (f.field_array_rank() == 3) {
        int dim0 = stk::mesh::find_restriction(f, b.entity_rank(), b.supersets()).dimension();
        if (dim0 == 4) {
          stride[0] = dim0;
          stride[1] = get_size<Tag2>(b) * dim0;
          stride[2] = stk::mesh::field_scalars_per_entity(f, b);
        } else {
          // IKT, 12/20/18: this changes the way the qp_tensor field
          // for 1D and 3D problems appears in the output exodus field.
          // Fields appear like: Cauchy_Stress_1_1, ...  Cauchy_Stress_8_9,
          // instead of Cauchy_Stress_1_01 .. Cauchy_Stress_3_24 to make it
          // more clear which entry corresponds to which component/quad point.
          // I believe for 2D problems the original layout is correct, hence
          // the if statement above here.
          stride[0] = get_size<Tag1>(b);
          stride[1] = get_size<Tag2>(b) * stride[0];
          stride[2] = stk::mesh::field_scalars_per_entity(f, b);
        }
      } else if (f.field_array_rank() == 4) {
        int dim0  = stk::mesh::find_restriction(f, b.entity_rank(), b.supersets()).dimension();
        stride[0] = dim0;
        stride[1] = get_size<Tag2>(b) * dim0;
        stride[2] = get_size<Tag3>(b) * stride[1];
        stride[3] = stk::mesh::field_scalars_per_entity(f, b);
      } else {
        assert(false);
      }

      array_type::assign_stride((ScalarType*)(b.field_data_location(f)), stride, (typename array_type::size_type)b.size());
    }
  }
};

}  // namespace Albany

#endif  // ALBANY_BUCKETARRAY_HPP
