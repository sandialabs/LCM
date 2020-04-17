// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_BCUtils.hpp"

#include "Albany_BCUtils_Def.hpp"

// Initialize statics

std::string const Albany::DirichletTraits::bcParamsPl = "Dirichlet BCs";
std::string const Albany::NeumannTraits::bcParamsPl   = "Neumann BCs";

BCUTILS_INSTANTIATE_TEMPLATE_CLASS(Albany::BCUtils)
