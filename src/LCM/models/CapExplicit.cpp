// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "CapExplicit.hpp"

#include "CapExplicit_Def.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "KernelConstitutiveModel_Def.hpp"

template <typename EvalT, typename Traits>
LCM::CapExplicit<EvalT, Traits>::CapExplicit(Teuchos::ParameterList* p, const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::KernelConstitutiveModel<EvalT, Traits, CapExplicitKernel<EvalT, Traits>>(p, dl)
{
}

PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::CapExplicitKernel)
PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::CapExplicit)
