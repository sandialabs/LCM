// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#ifndef ALBANY_OBSERVER_IMPL_HPP
#define ALBANY_OBSERVER_IMPL_HPP

#include <string>
#include <vector>

#include "Albany_NodalFieldProjector.hpp"
#include "Albany_StatelessObserverImpl.hpp"

namespace Albany {

class ObserverImpl : public StatelessObserverImpl
{
 public:
  explicit ObserverImpl(const Teuchos::RCP<Application>& app);

  void
  observeSolution(
      double                                  stamp,
      Thyra_Vector const&                     nonOverlappedSolution,
      const Teuchos::Ptr<Thyra_Vector const>& nonOverlappedSolutionDot,
      const Teuchos::Ptr<Thyra_Vector const>& nonOverlappedSolutionDotDot) override;

  void
  observeSolution(double stamp, const Thyra_MultiVector& nonOverlappedSolution) override;

  void
  parameterChanged(std::string const& param);

 private:
  // Project saved quadrature-point states to nodal fields, driven explicitly
  // just before each solution write. Built lazily from the problem's "Project
  // IP to Nodal Field" responses (the standalone projector supersedes the
  // response-based projection, whose evaluation lags the output write and so
  // leaves the first written dynamic step's nodal fields zero -- GitHub #11).
  void
  projectNodalFields(double stamp);

  std::vector<Teuchos::RCP<NodalFieldProjector>> projectors_;
  bool                                           projectors_built_{false};
};

}  // namespace Albany

#endif  // ALBANY_OBSERVER_IMPL_HPP
