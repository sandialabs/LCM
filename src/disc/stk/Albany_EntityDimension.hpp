// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_ENTITYDIMENSION_HPP
#define ALBANY_ENTITYDIMENSION_HPP

#include <Shards_Array.hpp>

namespace Albany {

// Dimension tag used for the shards::Array views that carry mesh state into
// the workset arrays. It labels every dimension (element, node, component),
// so the arrays are shaped at run time rather than by the tag type.
//
// This header used to also provide a BucketArray class that wrapped a raw
// stk::mesh::field_data pointer in a FortranOrder shards::Array. That baked
// in the assumption that an entity's components are adjacent in memory, which
// is true only for stk::mesh::Layout::Right; a unified-memory STK build lays
// host Field data out as Layout::Left and the view would have been silently
// transposed. It was never instantiated, so it was removed rather than
// migrated to the new STK Field data API.
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

}  // namespace Albany

#endif  // ALBANY_ENTITYDIMENSION_HPP
