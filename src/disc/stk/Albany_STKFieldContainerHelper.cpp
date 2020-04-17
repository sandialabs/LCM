// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_STKFieldContainerHelper.hpp"

#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Albany_STKFieldContainerHelper_Def.hpp"

namespace Albany {

template struct STKFieldContainerHelper<
    Albany::AbstractSTKFieldContainer::ScalarFieldType>;
template struct STKFieldContainerHelper<
    Albany::AbstractSTKFieldContainer::VectorFieldType>;

}  // namespace Albany
