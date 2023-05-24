// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_CommUtils.hpp"

#include "Albany_ThyraUtils.hpp"

namespace Albany {

MPI_Comm
getMpiCommFromTeuchosComm(Teuchos::RCP<Teuchos_Comm const>& tc)
{
  Teuchos::Ptr<const Teuchos::MpiComm<int>> mpiComm = Teuchos::ptr_dynamic_cast<const Teuchos::MpiComm<int>>(Teuchos::ptrFromRef(*tc));
  return *mpiComm->getRawMpiComm();
}

Teuchos::RCP<Teuchos_Comm const>
createTeuchosCommFromMpiComm(const MPI_Comm& mc)
{
  // The default tag in the MpiComm is used in Teuchos send/recv operations
  // *only if* the user does not specify a tag for the message. Here, I pick a
  // weird large number, unlikely to ever be hit by a tag used by albany.
  return Teuchos::rcp(new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(mc), 1984));
}

Teuchos::RCP<Teuchos_Comm const>
getDefaultComm()
{
  return Teuchos::DefaultComm<int>::getComm();
}

Teuchos::RCP<Teuchos_Comm const>
createTeuchosCommFromThyraComm(const Teuchos::RCP<const Teuchos::Comm<Teuchos::Ordinal>>& tc_in)
{
  const Teuchos::RCP<const Teuchos::MpiComm<Teuchos::Ordinal>> mpiCommIn = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<Teuchos::Ordinal>>(tc_in);
  if (nonnull(mpiCommIn)) {
    return Teuchos::createMpiComm<int>(mpiCommIn->getRawMpiComm());
  }

  // Assert conversion to Teuchos::SerialComm as a last resort (or throw)
  Teuchos::rcp_dynamic_cast<const Teuchos::SerialComm<Teuchos::Ordinal>>(tc_in, true);

  return Teuchos::createSerialComm<int>();
}

Teuchos::RCP<const Teuchos::Comm<Teuchos::Ordinal>>
createThyraCommFromTeuchosComm(const Teuchos::RCP<Teuchos_Comm const>& tc_in)
{
  const Teuchos::RCP<const Teuchos::MpiComm<int>> mpiCommIn = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(tc_in);
  if (nonnull(mpiCommIn)) {
    return Teuchos::createMpiComm<Teuchos::Ordinal>(mpiCommIn->getRawMpiComm());
  }

  // Assert conversion to Teuchos::SerialComm as a last resort (or throw)
  Teuchos::rcp_dynamic_cast<const Teuchos::SerialComm<int>>(tc_in, true);

  return Teuchos::createSerialComm<Teuchos::Ordinal>();
}

}  // namespace Albany
