// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "ACEice.hpp"

#include "ACEice_Def.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "ParallelConstitutiveModel_Def.hpp"

template <typename EvalT, typename Traits>
LCM::ACEice<EvalT, Traits>::ACEice(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ParallelConstitutiveModel<
          EvalT,
          Traits,
          ACEiceMiniKernel<EvalT, Traits>>(p, dl)
{
}

PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::ACEiceMiniKernel)
PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::ACEice)
