// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_KL_RESPONSE_FUNCTION_HPP
#define ALBANY_KL_RESPONSE_FUNCTION_HPP

#include "Albany_AbstractResponseFunction.hpp"
#include "Albany_Application.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_ParameterList.hpp"

namespace Albany {

/*!
 * \brief A response function given by the KL decomposition of another
 * response function.
 *
 * It only defines the SG methods.
 */
class KLResponseFunction : public AbstractResponseFunction
{
 public:
  //! Default constructor
  KLResponseFunction(
      const Teuchos::RCP<AbstractResponseFunction>& response,
      Teuchos::ParameterList&                       responseParams);

  //! Destructor
  ~KLResponseFunction() = default;

  //! Setup response function
  void
  setup() override
  {
    response->setup();
  }

  //! Perform post registration setup (do nothing)
  void
  postRegSetup() override
  {
  }

  //! Get the vector space associated with this response
  Teuchos::RCP<Thyra_VectorSpace const>
  responseVectorSpace() const override
  {
    return response->responseVectorSpace();
  }

  /*!
   * \brief Is this response function "scalar" valued, i.e., has a replicated
   * local response map.
   */
  bool
  isScalarResponse() const override
  {
    return response->isScalarResponse();
  }

  //! Create operator for gradient (e.g., dg/dx)
  Teuchos::RCP<Thyra_LinearOp>
  createGradientOp() const override
  {
    return response->createGradientOp();
  }

  //! \name Deterministic evaluation functions
  //@{

  //! Evaluate responses
  void
  evaluateResponse(
      double const                            current_time,
      Teuchos::RCP<Thyra_Vector const> const& x,
      Teuchos::RCP<Thyra_Vector const> const& xdot,
      Teuchos::RCP<Thyra_Vector const> const& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      Teuchos::RCP<Thyra_Vector> const&       g) override;

  //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
  void
  evaluateDerivative(
      double const                                     current_time,
      Teuchos::RCP<Thyra_Vector const> const&          x,
      Teuchos::RCP<Thyra_Vector const> const&          xdot,
      Teuchos::RCP<Thyra_Vector const> const&          xdotdot,
      const Teuchos::Array<ParamVec>&                  p,
      ParamVec*                                        deriv_p,
      Teuchos::RCP<Thyra_Vector> const&                g,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp) override;
  //@}

 protected:
  //! Response function we work with
  Teuchos::RCP<AbstractResponseFunction> response;

  //! Response parameters
  Teuchos::ParameterList responseParams;

  //! Output stream;
  Teuchos::RCP<Teuchos::FancyOStream> out;

  //! Number of KL terms
  int num_kl;
};

}  // namespace Albany

#endif  // ALBANY_KL_RESPONSE_FUNCTION_HPP
