// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_COMM_UTILS_HPP
#define ALBANY_COMM_UTILS_HPP

// Get Albany configuration macros
#include "Albany_CommTypes.hpp"
#include "Albany_config.h"
#include "Teuchos_ConfigDefs.hpp"  // For Ordinal
#include "Teuchos_RCP.hpp"

namespace Albany {

Teuchos::RCP<Teuchos_Comm const>
getDefaultComm();

Albany_MPI_Comm
getMpiCommFromTeuchosComm(Teuchos::RCP<Teuchos_Comm const>& tc);

Teuchos::RCP<Teuchos_Comm const>
createTeuchosCommFromMpiComm(const Albany_MPI_Comm& mc);

Teuchos::RCP<Teuchos_Comm const>
createTeuchosCommFromThyraComm(
    const Teuchos::RCP<const Teuchos::Comm<Teuchos::Ordinal>>& tc_in);

Teuchos::RCP<const Teuchos::Comm<Teuchos::Ordinal>>
createThyraCommFromTeuchosComm(const Teuchos::RCP<Teuchos_Comm const>& tc_in);

}  // namespace Albany

#endif  // ALBANY_COMM_UTILS_HPP
