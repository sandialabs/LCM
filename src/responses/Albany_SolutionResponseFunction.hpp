// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_SOLUTION_RESPONSE_FUNCTION_HPP
#define ALBANY_SOLUTION_RESPONSE_FUNCTION_HPP

#include "Albany_DistributedResponseFunction.hpp"
#include "Albany_ThyraCrsMatrixFactory.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace Albany {

class Application;

/*!
 * \brief A response function given by (possibly a portion of) the solution
 */
class SolutionResponseFunction : public DistributedResponseFunction
{
 public:
  //! Default constructor
  SolutionResponseFunction(
      const Teuchos::RCP<Albany::Application>& application,
      Teuchos::ParameterList const&            responseParams);

  //! Destructor
  virtual ~SolutionResponseFunction() = default;

  //! Setup response function
  void
  setup() override;

  //! Get the map associate with this response
  Teuchos::RCP<Thyra_VectorSpace const>
  responseVectorSpace() const override
  {
    return culled_vs;
  }

  //! Create operator for gradient
  Teuchos::RCP<Thyra_LinearOp>
  createGradientOp() const override;

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
      Teuchos::RCP<Thyra_MultiVector> const&  dg_dp) override;

 protected:
  void
  cullSolution(
      const Teuchos::RCP<const Thyra_MultiVector>& x,
      Teuchos::RCP<Thyra_MultiVector> const&       x_culled) const;

  //! Mask for DOFs to keep
  Teuchos::Array<bool> keepDOF;
  int                  numKeepDOF;

  //! Vector space for response
  Teuchos::RCP<const Thyra_SpmdVectorSpace> solution_vs;
  Teuchos::RCP<const Thyra_SpmdVectorSpace> culled_vs;

  //! The restriction operator, use to cull the solution
  Teuchos::RCP<Thyra_LinearOp> cull_op;

  //! Factory for the culling operator
  Teuchos::RCP<ThyraCrsMatrixFactory> cull_op_factory;
};

}  // namespace Albany

#endif  // ALBANY_SOLUTION_RESPONSE_FUNCTION_HPP
