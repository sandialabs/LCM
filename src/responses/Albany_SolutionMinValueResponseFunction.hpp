//
// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
//

#ifndef ALBANY_SOLUTION_MIN_VALUE_RESPONSE_FUNCTION_HPP
#define ALBANY_SOLUTION_MIN_VALUE_RESPONSE_FUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"

namespace Albany {

/*!
 * \brief Reponse function representing the min of the solution values
 */
class SolutionMinValueResponseFunction
    : public SamplingBasedScalarResponseFunction
{
 public:
  //! Default constructor
  SolutionMinValueResponseFunction(
      const Teuchos::RCP<const Teuchos_Comm>& comm,
      int                                     neq                 = 1,
      int                                     eq                  = 0,
      bool                                    interleavedOrdering = true);

  //! Destructor
  ~SolutionMinValueResponseFunction() = default;

  //! Get the number of responses
  unsigned int
  numResponses() const
  {
    return 1;
  }

  //! Evaluate responses
  void
  evaluateResponse(
      double const                            current_time,
      Teuchos::RCP<Thyra_Vector const> const& x,
      Teuchos::RCP<Thyra_Vector const> const& xdot,
      Teuchos::RCP<Thyra_Vector const> const& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      const Teuchos::RCP<Thyra_Vector>&       g);

  //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
  void
  evaluateGradient(
      double const                            current_time,
      Teuchos::RCP<Thyra_Vector const> const& x,
      Teuchos::RCP<Thyra_Vector const> const& xdot,
      Teuchos::RCP<Thyra_Vector const> const& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      ParamVec*                               deriv_p,
      const Teuchos::RCP<Thyra_Vector>&       g,
      const Teuchos::RCP<Thyra_MultiVector>&  dg_dx,
      const Teuchos::RCP<Thyra_MultiVector>&  dg_dxdot,
      const Teuchos::RCP<Thyra_MultiVector>&  dg_dxdotdot,
      const Teuchos::RCP<Thyra_MultiVector>&  dg_dp);

 protected:
  //! Compute min value
  void
  computeMinValue(Teuchos::RCP<Thyra_Vector const> const& x, ST& val);

  //! Number of equations per node
  int neq;

  //! Equation we want to get the max value from
  int eq;

  Teuchos::RCP<const Teuchos_Comm> comm_;

  //! Flag for interleaved verus blocked unknown ordering
  bool interleavedOrdering;
};

}  // namespace Albany

#endif  // ALBANY_SOLUTION_MIN_VALUE_RESPONSE_FUNCTION_HPP
