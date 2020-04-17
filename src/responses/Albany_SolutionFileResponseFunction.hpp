// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_SOLUTION_FILE_RESPONSE_FUNCTION_HPP
#define ALBANY_SOLUTION_FILE_RESPONSE_FUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"

namespace Albany {

/*!
 * \brief Response function representing the difference from a stored vector on
 * disk
 */
template <class VectorNorm>
class SolutionFileResponseFunction : public SamplingBasedScalarResponseFunction
{
 public:
  //! Default constructor
  SolutionFileResponseFunction(const Teuchos::RCP<Teuchos_Comm const>& comm);

  //! Destructor
  ~SolutionFileResponseFunction() = default;

  //! Get the number of responses
  unsigned int
  numResponses() const
  {
    return 1;
  }

  //! Perform optimization setup
  virtual void
  postRegSetup(){};

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
  int
  MatrixMarketFile(
      char const*                            filename,
      Teuchos::RCP<Thyra_MultiVector> const& mv);

  //! Reference Vector - Thyra
  Teuchos::RCP<Thyra_Vector> RefSoln;

  // A temp vector used in the response. Store it, so we create/destroy it only
  // once.
  Teuchos::RCP<Thyra_Vector> diff;

  bool solutionLoaded;
};

struct NormTwo
{
  static double
  Norm(Thyra_Vector const& vec)
  {
    auto norm = vec.norm_2();
    return norm * norm;
  }

  static void
  NormDerivative(
      Thyra_Vector const& x,
      Thyra_Vector const& soln,
      Thyra_Vector&       grad)
  {
    Teuchos::Array<ST> coeffs(2);
    coeffs[0] = 2.0;
    coeffs[1] = -2.0;
    Teuchos::Array<Teuchos::Ptr<Thyra_Vector const>> vecs(2);
    vecs[0] = Teuchos::constPtr(x);
    vecs[1] = Teuchos::constPtr(soln);
    grad.linear_combination(coeffs, vecs, 0.0);
  }
};

struct NormInf
{
  static double
  Norm(Thyra_Vector const& vec)
  {
    return vec.norm_inf();
  }

  static void
  NormDerivative(
      Thyra_Vector const& /* x */,
      Thyra_Vector const& /* soln */,
      Thyra_Vector& /* grad */)
  {
    ALBANY_ABORT(
        "SolutionFileResponseFunction::NormInf::NormDerivative is not "
        "Implemented yet!\n");
  }
};

}  // namespace Albany

#endif  // ALBANY_SOLUTION_FILE_RESPONSE_FUNCTION_HPP
