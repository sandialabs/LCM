// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_SOLUTION_VALUES_RESPONSE_FUNCTION_HPP
#define ALBANY_SOLUTION_VALUES_RESPONSE_FUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"
#include "Albany_Utils.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {

class SolutionCullingStrategyBase;
class Application;
class CombineAndScatterManager;

/*!
 * \brief Reponse function representing the average of the solution values
 */
class SolutionValuesResponseFunction
    : public SamplingBasedScalarResponseFunction
{
 public:
  //! Constructor
  SolutionValuesResponseFunction(
      const Teuchos::RCP<const Application>& app,
      Teuchos::ParameterList&                responseParams);

  //! Get the number of responses
  unsigned int
  numResponses() const;

  //! Setup response function
  void
  setup();

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
  Teuchos::RCP<const Application> app_;

  Teuchos::RCP<SolutionCullingStrategyBase> cullingStrategy_;

  Teuchos::RCP<Thyra_Vector>             culledVec;
  Teuchos::RCP<CombineAndScatterManager> cas_manager;

  class SolutionPrinter;
  Teuchos::RCP<SolutionPrinter> sol_printer_;

  void
  updateCASManager();
};

}  // namespace Albany

#endif  // ALBANY_SOLUTION_VALUES_RESPONSE_FUNCTION_HPP
