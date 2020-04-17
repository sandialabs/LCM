// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "PHAL_ScatterScalarNodalParameter.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_ScatterScalarNodalParameter_Def.hpp"

template <typename Traits>
std::string const PHAL::ScatterScalarNodalParameter<
    PHAL::AlbanyTraits::Residual,
    Traits>::className = "ScatterScalarNodalParameter";

template <typename Traits>
std::string const PHAL::ScatterScalarExtruded2DNodalParameter<
    PHAL::AlbanyTraits::Residual,
    Traits>::className = "ScatterScalarExtruded2NodalParameter";

PHAL_INSTANTIATE_TEMPLATE_CLASS(PHAL::ScatterScalarNodalParameterBase)
PHAL_INSTANTIATE_TEMPLATE_CLASS(PHAL::ScatterScalarNodalParameter)
PHAL_INSTANTIATE_TEMPLATE_CLASS(PHAL::ScatterScalarExtruded2DNodalParameter)
