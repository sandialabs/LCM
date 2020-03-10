//
// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
//

#ifndef ALBANY_SOLUTION_AVERAGE_RESPONSE_FUNCTION_HPP
#define ALBANY_SOLUTION_AVERAGE_RESPONSE_FUNCTION_HPP

#include "Albany_ScalarResponseFunction.hpp"

namespace Albany {

/*!
 * \brief Reponse function representing the average of the solution values
 */
class SolutionAverageResponseFunction : public ScalarResponseFunction
{
 public:
  //! Default constructor
  SolutionAverageResponseFunction(const Teuchos::RCP<Teuchos_Comm const>& comm);

  //! Destructor
  ~SolutionAverageResponseFunction() = default;

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

 private:
  void
  evaluateResponseImpl(Thyra_Vector const& x, Thyra_Vector& g);

  Teuchos::RCP<Thyra_Vector>      one;
  Teuchos::RCP<Thyra_MultiVector> ones;
};

}  // namespace Albany

#endif  // ALBANY_SOLUTION_AVERAGE_RESPONSE_FUNCTION_HPP
