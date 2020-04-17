// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_ABSTRACT_RESPONSE_FUNCTION_HPP
#define ALBANY_ABSTRACT_RESPONSE_FUNCTION_HPP

#include "Albany_DataTypes.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"
#include "Thyra_ModelEvaluatorBase.hpp"

namespace Albany {

/*!
 * \brief Abstract interface for representing a response function
 */
class AbstractResponseFunction
{
 public:
  //! Default constructor
  AbstractResponseFunction(){};

  //! Destructor
  virtual ~AbstractResponseFunction(){};

  //! Setup response function
  virtual void
  setup() = 0;

  //! Get the vector space associated with this response.
  virtual Teuchos::RCP<Thyra_VectorSpace const>
  responseVectorSpace() const = 0;

  /*!
   * \brief Is this response function "scalar" valued, i.e., has a replicated
   * local response map.
   */
  virtual bool
  isScalarResponse() const = 0;

  //! Create Thyra operator for gradient (e.g., dg/dx)
  virtual Teuchos::RCP<Thyra_LinearOp>
  createGradientOp() const = 0;

  //! perform post registration setup
  virtual void
  postRegSetup() = 0;

  //! \name Deterministic evaluation functions
  //@{

  //! Evaluate responses
  virtual void
  evaluateResponse(
      double const                            current_time,
      Teuchos::RCP<Thyra_Vector const> const& x,
      Teuchos::RCP<Thyra_Vector const> const& xdot,
      Teuchos::RCP<Thyra_Vector const> const& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      Teuchos::RCP<Thyra_Vector> const&       g) = 0;

  virtual void
  evaluateDerivative(
      double const                                     current_time,
      Teuchos::RCP<Thyra_Vector const> const&          x,
      Teuchos::RCP<Thyra_Vector const> const&          xdot,
      Teuchos::RCP<Thyra_Vector const> const&          xdotdot,
      const Teuchos::Array<ParamVec>&                  p,
      ParamVec*                                        deriv_p,
      Teuchos::RCP<Thyra_Vector> const&                gT,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp) = 0;

 private:
  //! Private to prohibit copying
  AbstractResponseFunction(const AbstractResponseFunction&);

  //! Private to prohibit copying
  AbstractResponseFunction&
  operator=(const AbstractResponseFunction&);
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_RESPONSE_FUNCTION_HPP
