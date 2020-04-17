// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

// @HEADER

#include "PerformanceContext.hpp"

namespace util {

PerformanceContext PerformanceContext::instance_ = PerformanceContext();

PerformanceContext&
PerformanceContext::instance()
{
  // Static object lifetime
  return instance_;
}

void
PerformanceContext::summarizeAll(
    Teuchos::Ptr<Teuchos::Comm<int> const> comm,
    std::ostream&                          out)
{
  timeMonitor_.summarize(comm, out);
  counterMonitor_.summarize(comm, out);
  variableMonitor_.summarize(comm, out);
}

void
PerformanceContext::summarizeAll(std::ostream& out)
{
  // MPI should be initialized before this call
  Teuchos::RCP<Teuchos::Comm<int> const> comm =
      Teuchos::DefaultComm<int>::getComm();

  summarizeAll(comm.ptr(), out);
}

}  // namespace util
