// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "ParallelNeohookeanModel.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "ParallelConstitutiveModel_Def.hpp"
#include "ParallelNeohookeanModel_Def.hpp"

template <typename EvalT, typename Traits>
LCM::ParallelNeohookeanModel<EvalT, Traits>::ParallelNeohookeanModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ParallelConstitutiveModel<
          EvalT,
          Traits,
          NeohookeanKernel<EvalT, Traits>>(p, dl)
{
}

PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::NeohookeanKernel)
PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::ParallelNeohookeanModel)
