// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_AGGREGATE_SCALAR_RESPONSE_FUNCTION_HPP
#define ALBANY_AGGREGATE_SCALAR_RESPONSE_FUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

/*!
 * \brief A response function that aggregates together multiple response
 * functions into one.
 */
class AggregateScalarResponseFunction
    : public SamplingBasedScalarResponseFunction
{
 public:
  //! Default constructor
  AggregateScalarResponseFunction(
      const Teuchos::RCP<Teuchos_Comm const>&                     comm,
      const Teuchos::Array<Teuchos::RCP<ScalarResponseFunction>>& responses);

  //! Setup response function
  void
  setup() override;

  //! Perform post registration setup
  void
  postRegSetup() override;

  //! Destructor
  ~AggregateScalarResponseFunction() = default;

  //! Get the number of responses
  unsigned int
  numResponses() const override;

  //! Evaluate response
  void
  evaluateResponse(
      double const                            current_time,
      Teuchos::RCP<Thyra_Vector const> const& x,
      Teuchos::RCP<Thyra_Vector const> const& xdot,
      Teuchos::RCP<Thyra_Vector const> const& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      Teuchos::RCP<Thyra_Vector> const&       gT) override;

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
      Teuchos::RCP<Thyra_MultiVector> const&  dg_dp) override;

 protected:
  //! Response functions to aggregate
  Teuchos::Array<Teuchos::RCP<ScalarResponseFunction>> responses;

  Teuchos::RCP<const Thyra_ProductVectorSpace> productVectorSpace;
};

}  // namespace Albany

#endif  // ALBANY_AGGREGATE_SCALAR_RESPONSE_FUNCTION_HPP
