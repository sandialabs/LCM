// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_PROBLEMUTILS_HPP
#define ALBANY_PROBLEMUTILS_HPP

#include "Albany_ScalarOrdinalTypes.hpp"
#include "Intrepid2_Basis.hpp"
#include "Phalanx_KokkosDeviceTypes.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {

//! Helper Factory function to construct Intrepid2 Basis from Shards
//! CellTopologyData
Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
getIntrepid2Basis(const CellTopologyData& ctd, bool compositeTet = false);

bool
mesh_depends_on_solution();
bool
mesh_depends_on_parameters();
bool
params_depend_on_solution();

}  // namespace Albany

#endif  // ALBANY_PROBLEMUTILS_HPP
