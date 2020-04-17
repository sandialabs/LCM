// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "CrystalPlasticityModel.hpp"

#include "CrystalPlasticityModel_Def.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "ParallelConstitutiveModel_Def.hpp"

template <typename EvalT, typename Traits>
LCM::CrystalPlasticityModel<EvalT, Traits>::CrystalPlasticityModel(
    Teuchos::ParameterList*              p,
    Teuchos::RCP<Albany::Layouts> const& dl)
    : LCM::ParallelConstitutiveModel<
          EvalT,
          Traits,
          CrystalPlasticityKernel<EvalT, Traits>>(p, dl)
{
}

PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::CrystalPlasticityKernel)
PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::CrystalPlasticityModel)
