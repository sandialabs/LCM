#include "Albany_Gather.hpp"

#include "Albany_Macros.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_RCP.hpp"

#if defined(ALBANY_MPI)
#include <mpi.h>

#include <Teuchos_DefaultMpiComm.hpp>
#endif

#include "Albany_Macros.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_DefaultSerialComm.hpp"
#include "Teuchos_Details_MpiTypeTraits.hpp"

namespace Albany {

void
gatherAllV(
    const Teuchos::RCP<Teuchos_Comm const>& comm,
    const Teuchos::ArrayView<const GO>&     myVals,
    Teuchos::Array<GO>&                     allVals)
{
  int const myCount = myVals.size();
#if defined(ALBANY_MPI)
  if (const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>*>(comm.get())) {
    MPI_Comm rawComm = (*mpiComm->getRawMpiComm().get())();

    int allCount;
    MPI_Allreduce(&myCount, &allCount, 1, MPI_INT, MPI_SUM, rawComm);
    allVals.resize(allCount);

    int const           cpuCount = mpiComm->getSize();
    Teuchos::Array<int> allValCounts(cpuCount);
    int const           ierr = MPI_Allgather(&myCount, 1, MPI_INT, allValCounts.getRawPtr(), 1, MPI_INT, rawComm);
    ALBANY_PANIC(ierr != 0);

    Teuchos::Array<int> allValDisps(cpuCount, 0);
    for (int i = 1; i < cpuCount; ++i) {
      allValDisps[i] = allValDisps[i - 1] + allValCounts[i - 1];
    }
    ALBANY_EXPECT(allCount == allValCounts.back() + allValDisps.back(), "Error! Mismatch in values counts.\n");

    auto GO_type = Teuchos::Details::MpiTypeTraits<GO>::getType();
    MPI_Allgatherv(
        const_cast<GO*>(myVals.getRawPtr()),
        myCount,
        GO_type,
        allVals.getRawPtr(),
        allValCounts.getRawPtr(),
        allValDisps.getRawPtr(),
        GO_type,
        rawComm);
  } else
#endif
      if (dynamic_cast<const Teuchos::SerialComm<int>*>(comm.get())) {
    allVals.resize(myCount);
    std::copy(myVals.getRawPtr(), myVals.getRawPtr() + myCount, allVals.getRawPtr());
  } else {
    bool const commTypeNotSupported = true;
    ALBANY_PANIC(commTypeNotSupported);
  }
}

void
gatherV(
    const Teuchos::RCP<Teuchos_Comm const>& comm,
    const Teuchos::ArrayView<const GO>&     myVals,
    Teuchos::Array<GO>&                     allVals,
    const LO                                root_rank)
{
  int const myCount = myVals.size();
#if defined(ALBANY_MPI)
  if (const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>*>(comm.get())) {
    MPI_Comm rawComm = (*mpiComm->getRawMpiComm().get())();

    int allCount;
    MPI_Allreduce(&myCount, &allCount, 1, MPI_INT, MPI_SUM, rawComm);

    int myRank     = comm->getRank();
    int myAllCount = (myRank == root_rank ? allCount : 0);
    allVals.resize(myAllCount);

    int const           cpuCount = mpiComm->getSize();
    Teuchos::Array<int> allValCounts(cpuCount);
    int const ierr = MPI_Gather(&myCount, 1, MPI_INT, allValCounts.getRawPtr(), 1, MPI_INT, root_rank, rawComm);
    ALBANY_PANIC(ierr != 0);

    Teuchos::Array<int> allValDisps(cpuCount, 0);
    for (int i = 1; i < cpuCount; ++i) {
      allValDisps[i] = allValDisps[i - 1] + allValCounts[i - 1];
    }
    ALBANY_EXPECT(
        myRank != root_rank || (allCount == allValCounts.back() + allValDisps.back()),
        "Error! Mismatch in values counts.\n");

    auto GO_type = Teuchos::Details::MpiTypeTraits<GO>::getType();
    MPI_Gatherv(
        const_cast<GO*>(myVals.getRawPtr()),
        myCount,
        GO_type,
        allVals.getRawPtr(),
        allValCounts.getRawPtr(),
        allValDisps.getRawPtr(),
        GO_type,
        root_rank,
        rawComm);
  } else
#endif
      if (dynamic_cast<const Teuchos::SerialComm<int>*>(comm.get())) {
    allVals.resize(myCount);
    std::copy(myVals.getRawPtr(), myVals.getRawPtr() + myCount, allVals.getRawPtr());
  } else {
    bool const commTypeNotSupported = true;
    ALBANY_PANIC(commTypeNotSupported);
  }
}

}  // namespace Albany
