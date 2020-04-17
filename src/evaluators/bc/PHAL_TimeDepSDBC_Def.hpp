// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_Utils.hpp"
#include "PHAL_SDirichlet_Def.hpp"
#include "PHAL_TimeDepSDBC.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template <typename EvalT, typename Traits>
TimeDepSDBC_Base<EvalT, Traits>::TimeDepSDBC_Base(Teuchos::ParameterList& p)
    : PHAL::SDirichlet<EvalT, Traits>(p)
{
  offset_ = p.get<int>("Equation Offset");
  times_  = p.get<Teuchos::Array<RealType>>("Time Values").toVector();
  values_ = p.get<Teuchos::Array<RealType>>("BC Values").toVector();

  ALBANY_ASSERT(
      times_.size() == values_.size(),
      "Number of times and number of values must match");
}

template <typename EvalT, typename Traits>
typename TimeDepSDBC_Base<EvalT, Traits>::ScalarT
TimeDepSDBC_Base<EvalT, Traits>::computeVal(RealType time)
{
  auto const n = times_.size();
  if (time <= times_[0]) return values_[0];
  if (time >= times_[n - 1]) return values_[n - 1];

  for (auto i = 1; i < n; ++i) {
    if (time < times_[i]) {
      RealType const dv    = values_[i] - values_[i - 1];
      RealType const dt    = times_[i] - times_[i - 1];
      RealType const slope = dv / dt;
      ScalarT        value = values_[i - 1] + slope * (time - times_[i - 1]);
      return value;
    }
  }
  return 0.0;
}

template <typename EvalT, typename Traits>
TimeDepSDBC<EvalT, Traits>::TimeDepSDBC(Teuchos::ParameterList& p)
    : TimeDepSDBC_Base<EvalT, Traits>(p)
{
}

template <typename EvalT, typename Traits>
void
TimeDepSDBC<EvalT, Traits>::preEvaluate(typename Traits::EvalData workset)
{
  this->value = this->computeVal(workset.current_time);
  PHAL::SDirichlet<EvalT, Traits>::preEvaluate(workset);
}

template <typename EvalT, typename Traits>
void
TimeDepSDBC<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  this->value = this->computeVal(workset.current_time);
  PHAL::SDirichlet<EvalT, Traits>::evaluateFields(workset);
}

}  // namespace PHAL
