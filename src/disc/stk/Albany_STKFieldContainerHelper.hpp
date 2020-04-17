// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_STK_FIELD_CONTAINER_HELPER_HPP
#define ALBANY_STK_FIELD_CONTAINER_HELPER_HPP

#include <stk_mesh/base/Bucket.hpp>

#include "Albany_NodalDOFManager.hpp"
#include "Albany_ThyraTypes.hpp"

namespace Albany {

class GlobalLocalIndexer;

template <class FieldType>
struct STKFieldContainerHelper
{
  // Fill (aka get) and save (aka set) methods

  // FieldType can be either scalar or vector, the code is the same. Either way,
  // offset must be less than the dimension of the field.
  static void
  fillVector(
      Thyra_Vector&                                 field_thyra,
      const FieldType&                              field_stk,
      const Teuchos::RCP<const GlobalLocalIndexer>& node_vs,
      const stk::mesh::Bucket&                      bucket,
      const NodalDOFManager&                        nodalDofManager,
      int const                                     offset);

  static void
  saveVector(
      Thyra_Vector const&                           field_thyra,
      FieldType&                                    field_stk,
      const Teuchos::RCP<const GlobalLocalIndexer>& node_vs,
      const stk::mesh::Bucket&                      bucket,
      const NodalDOFManager&                        nodalDofManager,
      int const                                     offset);

  // Convenience function to copy one field's contents to another
  static void
  copySTKField(const FieldType& source, FieldType& target);
};

}  // namespace Albany

#endif  // ALBANY_STK_FIELD_CONTAINER_HELPER_HPP
