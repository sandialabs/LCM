// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "J2Erosion.hpp"

#include "J2Erosion_Def.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "ParallelConstitutiveModel_Def.hpp"

template <typename EvalT, typename Traits>
LCM::J2Erosion<EvalT, Traits>::J2Erosion(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ParallelConstitutiveModel<
          EvalT,
          Traits,
          J2ErosionKernel<EvalT, Traits>>(p, dl)
{
}

PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::J2ErosionKernel)
PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::J2Erosion)
