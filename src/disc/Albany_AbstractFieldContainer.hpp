// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_ABSTRACT_FIELD_CONTAINER_HPP
#define ALBANY_ABSTRACT_FIELD_CONTAINER_HPP

#include <string>
#include <vector>

namespace Albany {

/*!
 * \brief Abstract interface for a field container
 *
 */
class AbstractFieldContainer
{
 public:
  typedef std::vector<std::string> FieldContainerRequirements;

  //! Destructor
  virtual ~AbstractFieldContainer() = default;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_FIELD_CONTAINER_HPP
