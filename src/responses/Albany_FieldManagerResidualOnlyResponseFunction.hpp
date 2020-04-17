// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_FIELD_MANAGER_RESIDUAL_ONLY_RESPONSE_FUNCTION_HPP
#define ALBANY_FIELD_MANAGER_RESIDUAL_ONLY_RESPONSE_FUNCTION_HPP

#include "Albany_FieldManagerScalarResponseFunction.hpp"

namespace Albany {

/*!
 * \brief Reponse function that calls an evaluator that implements only EvalT=
 * PHAL::AlbanyTraits::Residual.
 *
 * It seems that a common use case for a response function is to do something
 * with solution data and data available in evaluator worksets, but not
 * necessarily to implement a mathematical function g whose derivatives can be
 * formed. Examples including transferring data to another module in a loose
 * coupling of Albany with other software, or writing special files.
 *   This Response Function calls only EvalT=PHAL::AlbanyTraits::Residual
 * forms of overridden methods. It returns 0 for all derivatives. Hence a
 * sensitivity will turn out to be 0.
 */
class FieldManagerResidualOnlyResponseFunction
    : public FieldManagerScalarResponseFunction
{
 public:
  //! Constructor
  FieldManagerResidualOnlyResponseFunction(
      const Teuchos::RCP<Albany::Application>&     application,
      const Teuchos::RCP<Albany::AbstractProblem>& problem,
      const Teuchos::RCP<Albany::MeshSpecsStruct>& ms,
      const Teuchos::RCP<Albany::StateManager>&    stateMgr,
      Teuchos::ParameterList&                      responseParams);

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
  FieldManagerResidualOnlyResponseFunction(
      const FieldManagerResidualOnlyResponseFunction&);
  FieldManagerResidualOnlyResponseFunction&
  operator=(const FieldManagerResidualOnlyResponseFunction&);
};

}  // namespace Albany

#endif  // ALBANY_FIELD_MANAGER_RESIDUAL_ONLY_RESPONSE_FUNCTION_HPP
