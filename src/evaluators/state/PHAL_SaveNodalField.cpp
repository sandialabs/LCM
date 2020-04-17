// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "PHAL_SaveNodalField.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_SaveNodalField_Def.hpp"

template <typename EvalT, typename Traits>
std::string const PHAL::SaveNodalFieldBase<EvalT, Traits>::className =
    "Save_Nodal_Field";

PHAL_INSTANTIATE_TEMPLATE_CLASS(PHAL::SaveNodalField)
PHAL_INSTANTIATE_TEMPLATE_CLASS(PHAL::SaveNodalFieldBase)
