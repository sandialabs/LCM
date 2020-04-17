// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_CUMULATIVE_SCALAR_RESPONSE_FUNCTION_HPP
#define ALBANY_CUMULATIVE_SCALAR_RESPONSE_FUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

/*!
 * \brief A response function that aggregates together multiple response
 * functions into one.
 */
class CumulativeScalarResponseFunction
    : public SamplingBasedScalarResponseFunction
{
 public:
  //! Default constructor
  CumulativeScalarResponseFunction(
      const Teuchos::RCP<Teuchos_Comm const>&                     commT,
      const Teuchos::Array<Teuchos::RCP<ScalarResponseFunction>>& responses);

  //! Setup response function
  virtual void
  setup();

  //! Perform post registration setup
  virtual void
  postRegSetup();

  //! Destructor
  virtual ~CumulativeScalarResponseFunction();

  //! Get the number of responses
  virtual unsigned int
  numResponses() const;

  //! Evaluate response
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
  CumulativeScalarResponseFunction(const CumulativeScalarResponseFunction&);

  //! Private to prohibit copying
  CumulativeScalarResponseFunction&
  operator=(const CumulativeScalarResponseFunction&);

 protected:
  //! Response functions to add
  Teuchos::Array<Teuchos::RCP<ScalarResponseFunction>> responses;
  unsigned int                                         num_responses;
};

}  // namespace Albany

#endif  // ALBANY_CUMULATIVE_SCALAR_RESPONSE_FUNCTION_HPP
