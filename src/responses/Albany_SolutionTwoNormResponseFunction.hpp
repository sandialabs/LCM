// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_SOLUTIONTWONORMRESPONSEFUNCTION_HPP
#define ALBANY_SOLUTIONTWONORMRESPONSEFUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"

namespace Albany {

/*!
 * \brief Reponse function representing the average of the solution values
 */
class SolutionTwoNormResponseFunction
    : public SamplingBasedScalarResponseFunction
{
 public:
  //! Default constructor
  SolutionTwoNormResponseFunction(
      const Teuchos::RCP<Teuchos_Comm const>& commT);

  //! Destructor
  virtual ~SolutionTwoNormResponseFunction();

  //! Get the number of responses
  virtual unsigned int
  numResponses() const;

  //! Evaluate responses
  virtual void
  evaluateResponse(
      double const                            current_time,
      Teuchos::RCP<Thyra_Vector const> const& x,
      Teuchos::RCP<Thyra_Vector const> const& xdot,
      Teuchos::RCP<Thyra_Vector const> const& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      Teuchos::RCP<Thyra_Vector> const&       g);

  virtual void
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
  //! Private to prohibit copying
  SolutionTwoNormResponseFunction(const SolutionTwoNormResponseFunction&);

  SolutionTwoNormResponseFunction&
  operator=(const SolutionTwoNormResponseFunction&);
};

}  // namespace Albany

#endif  // ALBANY_SOLUTIONTWONORMRESPONSEFUNCTION_HPP
