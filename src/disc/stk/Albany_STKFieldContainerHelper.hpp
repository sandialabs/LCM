// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_STK_FIELD_CONTAINER_HELPER_HPP
#define ALBANY_STK_FIELD_CONTAINER_HELPER_HPP

#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/Field.hpp>

#include "Albany_NodalDOFManager.hpp"
#include "Albany_ThyraTypes.hpp"

namespace Albany {

class GlobalLocalIndexer;

// With simple fields, all STK field types (Scalar, Vector, Tensor) are
// stk::mesh::Field<double>. This struct is no longer a template.
struct STKFieldContainerHelper
{
  using FieldType = stk::mesh::Field<double>;

  // Fill (aka get) and save (aka set) methods
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
