// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_GENERIC_STK_FIELD_CONTAINER_HPP
#define ALBANY_GENERIC_STK_FIELD_CONTAINER_HPP

#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Teuchos_ParameterList.hpp"

// Forward declaration is enough
namespace stk {
namespace mesh {
class BulkData;
class MetaData;
}  // namespace mesh
}  // namespace stk

namespace Albany {

template <bool Interleaved>
class GenericSTKFieldContainer : public AbstractSTKFieldContainer
{
 public:
  GenericSTKFieldContainer(
      const Teuchos::RCP<Teuchos::ParameterList>& params_,
      const Teuchos::RCP<stk::mesh::MetaData>&    metaData_,
      const Teuchos::RCP<stk::mesh::BulkData>&    bulkData_,
      int const                                   neq_,
      int const                                   numDim_);

  virtual ~GenericSTKFieldContainer() = default;

  // Add StateStructs to the list of stored ones
  void
  addStateStructs(const Teuchos::RCP<Albany::StateInfoStruct>& sis);

 protected:
  Teuchos::RCP<stk::mesh::MetaData>    metaData;
  Teuchos::RCP<stk::mesh::BulkData>    bulkData;
  Teuchos::RCP<Teuchos::ParameterList> params;

  int neq;
  int numDim;
};

}  // namespace Albany

#endif  // ALBANY_GENERIC_STK_FIELD_CONTAINER_HPP
