#ifndef ALBANY_GLOBAL_LOCAL_INDEXER_HPP
#define ALBANY_GLOBAL_LOCAL_INDEXER_HPP

#include "Albany_CommTypes.hpp"
#include "Albany_Macros.hpp"
#include "Albany_ThyraTypes.hpp"

namespace Albany {

class GlobalLocalIndexer
{
 public:
  GlobalLocalIndexer(Teuchos::RCP<Thyra_VectorSpace const> const& vs) : m_vs(vs)
  {
    ALBANY_PANIC(
        m_vs.is_null(), "Error! Input vector space pointer is null.\n");
  }

  virtual ~GlobalLocalIndexer() = default;

  virtual GO
  getGlobalElement(const LO lid) const = 0;

  virtual LO
  getLocalElement(const GO gid) const = 0;

  virtual LO
  getNumLocalElements() const = 0;

  virtual GO
  getNumGlobalElements() const = 0;

  virtual Teuchos::RCP<Teuchos_Comm const>
  getComm() const = 0;

  virtual bool
  isLocallyOwnedElement(const GO gid) const = 0;

  Teuchos::RCP<Thyra_VectorSpace const>
  getVectorSpace() const
  {
    return m_vs;
  }

 protected:
  Teuchos::RCP<Thyra_VectorSpace const> m_vs;
};

// Create an indexer from a vector space
// WARNING: this is a COLLECTIVE operation. All ranks in the comm associated
//          with the vector space MUST call this function.
Teuchos::RCP<const GlobalLocalIndexer>
createGlobalLocalIndexer(Teuchos::RCP<Thyra_VectorSpace const> const& vs);

}  // namespace Albany

#endif  // ALBANY_GLOBAL_LOCAL_INDEXER_HPP
