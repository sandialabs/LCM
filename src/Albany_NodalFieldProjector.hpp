// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_NODAL_FIELD_PROJECTOR_HPP
#define ALBANY_NODAL_FIELD_PROJECTOR_HPP

#include <string>
#include <vector>

#include "Albany_Application.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Phalanx_FieldManager.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {

//
// Projects already-saved quadrature-point states to nodal fields (L2 "Full" or
// "Lumped" mass matrix) WITHOUT re-running the constitutive model.
//
// The projection's only inputs are the mesh geometry (-> basis functions) and
// the integration-point field values. Sourcing those values from the saved STK
// quadrature-point states (via LoadStateField), rather than from a fresh
// constitutive evaluation, means the projection touches no solver state: no
// solution scatter, no Dirichlet expansion, no constitutive re-run, no plastic
// or element-death bookkeeping. It therefore cannot perturb a solve -- which is
// what makes it safe to drive from the ACE thermo-mechanical coupling loop,
// where the standard response/observer path re-runs the constitutive model and
// corrupts the next step.
//
// It is deliberately problem-agnostic: it builds its own minimal Residual-only
// Phalanx field manager per element block from generic Albany evaluators, so it
// works for any problem with saved QP states and is intended to replace the
// response-based IPtoNodalField / ProjectIPtoNodalField projections.
//
class NodalFieldProjector
{
 public:
  // One projected field: the saved QP state name and its layout
  // ("Scalar", "Vector", or "Tensor"). The nodal output is written as the
  // state "proj_nodal_<name>".
  struct FieldSpec
  {
    std::string name;
    std::string layout;
  };

  // Builds one projection field manager per element block. `fields` lists the
  // saved QP states to project; `mass_matrix_type` is "Full" (L2) or "Lumped".
  // The projected nodal states must already be registered (the projector reuses
  // them; registration is idempotent), which holds whenever the corresponding
  // "Project IP to Nodal Field" response was declared.
  NodalFieldProjector(
      Teuchos::RCP<Application> const& app,
      std::vector<FieldSpec> const&    fields,
      std::string const&               mass_matrix_type = "Full",
      bool const                       output_to_exodus = true);

  // Read the current saved QP states and write the proj_nodal_* nodal states.
  void
  project(double const time) const;

 private:
  Teuchos::RCP<Application> app_;

  // One field manager per element block, indexed by workset physics index.
  std::vector<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>> fms_;

  // The projection evaluator holds a raw pointer to its config list, so the
  // lists must outlive the field managers.
  std::vector<Teuchos::RCP<Teuchos::ParameterList>> param_lists_;

  // Per-block phalanx-setup evaluation name, used by loadWorksetBucketInfo.
  std::vector<std::string> eval_names_;
};

}  // namespace Albany

#endif  // ALBANY_NODAL_FIELD_PROJECTOR_HPP
