// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

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
      Teuchos::RCP<Thyra_Vector> const&       g);

  //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
  void
  evaluateGradient(
      double const                            current_time,
      Teuchos::RCP<Thyra_Vector const> const& x,
      Teuchos::RCP<Thyra_Vector const> const& xdot,
      Teuchos::RCP<Thyra_Vector const> const& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      ParamVec*                               deriv_p,
      Teuchos::RCP<Thyra_Vector> const&       g,
      Teuchos::RCP<Thyra_MultiVector> const&  dg_dx,
      Teuchos::RCP<Thyra_MultiVector> const&  dg_dxdot,
      Teuchos::RCP<Thyra_MultiVector> const&  dg_dxdotdot,
      Teuchos::RCP<Thyra_MultiVector> const&  dg_dp);

 private:
  void
  evaluateResponseImpl(Thyra_Vector const& x, Thyra_Vector& g);

  Teuchos::RCP<Thyra_Vector>      one;
  Teuchos::RCP<Thyra_MultiVector> ones;
};

}  // namespace Albany

#endif  // ALBANY_SOLUTION_AVERAGE_RESPONSE_FUNCTION_HPP
