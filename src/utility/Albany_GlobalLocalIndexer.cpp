#include "Albany_GlobalLocalIndexer.hpp"

#include "Albany_GlobalLocalIndexerTpetra.hpp"
#include "Albany_TpetraThyraUtils.hpp"

namespace Albany {

Teuchos::RCP<const GlobalLocalIndexer>
createGlobalLocalIndexer(Teuchos::RCP<Thyra_VectorSpace const> const& vs)
{
  Teuchos::RCP<const GlobalLocalIndexer> indexer;

  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs, false);
  if (!tmap.is_null()) {
    indexer = Teuchos::rcp(new GlobalLocalIndexerTpetra(vs, tmap));
  }

  ALBANY_PANIC(
      indexer.is_null(),
      "Error! Could not cast the input vector space to any of the supported "
      "concrete types.\n");

  return indexer;
}

}  // namespace Albany
