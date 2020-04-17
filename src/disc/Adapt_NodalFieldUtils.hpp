// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ADAPT_NODAL_FIELD_UTILS_HPP
#define ADAPT_NODAL_FIELD_UTILS_HPP

#include <map>
#include <vector>

namespace Adapt {

struct NodeFieldSize
{
  std::string name;
  int         offset;
  int         ndofs;
};

typedef std::vector<NodeFieldSize>               NodeFieldSizeVector;
typedef std::map<std::string const, std::size_t> NodeFieldSizeMap;

}  // namespace Adapt

#endif  // ADAPT_NODAL_FIELD_UTILS_HPP
