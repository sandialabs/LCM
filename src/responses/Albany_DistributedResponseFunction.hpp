// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_DISTRIBUTED_RESPONSE_FUNCTION_HPP
#define ALBANY_DISTRIBUTED_RESPONSE_FUNCTION_HPP

#include "Albany_AbstractResponseFunction.hpp"

namespace Albany {

/*!
 * \brief Interface for distributed response functions
 *
 * Implements a few methods of AbstractResponseFunction specifically for
 * distributred responses, i.e., those that involve a distributed map.
 */
class DistributedResponseFunction : public AbstractResponseFunction
{
 public:
  //! Default constructor
  DistributedResponseFunction(){};

  //! Destructor
  virtual ~DistributedResponseFunction(){};

  //! Perform post registration setup (do nothing)
  virtual void
  postRegSetup(){};

  //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
  virtual void
  evaluateGradient(
      double const                            current_time,
      Teuchos::RCP<Thyra_Vector const> const& x,
      Teuchos::RCP<Thyra_Vector const> const& xdot,
      Teuchos::RCP<Thyra_Vector const> const& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      ParamVec*                               deriv_p,
      Teuchos::RCP<Thyra_Vector> const&       g,
      const Teuchos::RCP<Thyra_LinearOp>&     dg_dx,
      const Teuchos::RCP<Thyra_LinearOp>&     dg_dxdot,
      const Teuchos::RCP<Thyra_LinearOp>&     dg_dxdotdot,
      Teuchos::RCP<Thyra_MultiVector> const&  dg_dp) = 0;

  //! \name Implementation of AbstractResponseFunction virtual methods
  //@{

  /*!
   * \brief Is this response function "scalar" valued, i.e., has a replicated
   * local response map.
   */
  virtual bool
  isScalarResponse() const
  {
    return false;
  }

  //! Evaluate derivative dg/dx, dg/dxdot, dg/dp
  virtual void
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
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp);

  //@}

 private:
  //! Private to prohibit copying
  DistributedResponseFunction(const DistributedResponseFunction&);

  //! Private to prohibit copying
  DistributedResponseFunction&
  operator=(const DistributedResponseFunction&);

 protected:
  //! Comm for forming response map
  Teuchos::RCP<Teuchos_Comm const> commT;
};

}  // namespace Albany

#endif  // ALBANY_SCALAR_RESPONSE_FUNCTION_HPP
