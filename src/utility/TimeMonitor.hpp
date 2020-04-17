// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

// @HEADER

#ifndef UTIL_TIMEMONITOR_HPP
#define UTIL_TIMEMONITOR_HPP

/**
 *  \file TimeMonitor.hpp
 *
 *  \brief
 */

#include <Teuchos_Time.hpp>

#include "MonitorBase.hpp"

namespace util {
class TimeMonitor : public MonitorBase<Teuchos::Time>
{
 public:
  TimeMonitor();
  virtual ~TimeMonitor(){};

 protected:
  virtual string
  getStringValue(const monitored_type& val) override;
};
}  // namespace util

#endif  // UTIL_TIMEMONITOR_HPP
