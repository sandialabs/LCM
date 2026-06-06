// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_SCHWARZ_TRANSFER_HPP
#define ALBANY_SCHWARZ_TRANSFER_HPP

#include "Albany_AbstractSTKFieldContainer.hpp"
#include "DTK_MapOperator.hpp"
#include "DTK_STKMeshManager.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_MultiVector.hpp"

namespace Albany {

class Application;

// One-field DTK interpolation: source STK field on coupled_manager's mesh →
// target STK field on this_manager's mesh, using the DataTransferKit MapOperator
// configured by `dtk_params`. Returns the interpolated values as a Tpetra
// MultiVector with `neq` columns. The kernel performs both `MapOperator::setup`
// and `apply` — no operator caching (defer optimization).
Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>
doDTKInterpolation(
    DataTransferKit::STKMeshManager&                    coupled_manager,
    DataTransferKit::STKMeshManager&                    this_manager,
    AbstractSTKFieldContainer::VectorFieldType*         coupled_field,
    AbstractSTKFieldContainer::VectorFieldType*         this_field,
    int const                                           neq,
    Teuchos::ParameterList&                             dtk_params);

// Full Schwarz transfer for a single (this_app, coupled_app, nodeset_name)
// triple. Builds STKMeshManagers for the coupled (source) and this (target)
// subdomains, then calls `doDTKInterpolation` once per time-derivative slot
// available in the coupled app's solution_field array.
//
// Returns an array of length `num_sol_vecs = num_time_deriv + 1`:
//   index 0 → displacement (always present)
//   index 1 → velocity     (present if num_time_deriv >= 1)
//   index 2 → acceleration (present if num_time_deriv >= 2)
// Each MultiVector has `neq` columns (one per DOF component) and one row per
// node in the target nodeset's local subdomain. Indexing into a result is
// `data[neq * overlap_lid + eq]`.
Teuchos::Array<Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>>
computeSchwarzTransferDTK(
    Application const& this_app,
    Application const& coupled_app,
    std::string const& nodeset_name);

}  // namespace Albany

#endif  // ALBANY_SCHWARZ_TRANSFER_HPP
