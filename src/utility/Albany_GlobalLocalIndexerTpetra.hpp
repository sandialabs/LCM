#ifndef ALBANY_GLOBAL_LOCAL_INDEXER_TPETRA_HPP
#define ALBANY_GLOBAL_LOCAL_INDEXER_TPETRA_HPP

#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_TpetraTypes.hpp"

namespace Albany {

class GlobalLocalIndexerTpetra : public GlobalLocalIndexer
{
 public:
  GlobalLocalIndexerTpetra(Teuchos::RCP<Thyra_VectorSpace const> const& vs, const Teuchos::RCP<const Tpetra_Map>& tmap)
      : GlobalLocalIndexer(vs), m_tmap(tmap)
  {
    ALBANY_PANIC(tmap.is_null(), "Error! Input tpetra map pointer is null.\n");
  }

  GO
  getGlobalElement(const LO lid) const
  {
    return m_tmap->getGlobalElement(lid);
  }

  LO
  getLocalElement(const GO gid) const
  {
    return m_tmap->getLocalElement(gid);
  }

  LO
  getNumLocalElements() const
  {
    return m_tmap->getNodeNumElements();
  }
  GO
  getNumGlobalElements() const
  {
    return m_tmap->getGlobalNumElements();
  }

  bool
  isLocallyOwnedElement(const GO gid) const
  {
    return m_tmap->isNodeGlobalElement(gid);
  }

  Teuchos::RCP<Teuchos_Comm const>
  getComm() const
  {
    return m_tmap->getComm();
  }

 protected:
  Teuchos::RCP<const Tpetra_Map> m_tmap;
};

}  // namespace Albany

#endif  // ALBANY_GLOBAL_LOCAL_INDEXER_TPETRA_HPP
