// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_AggregateScalarResponseFunction.hpp"

#include "Albany_Application.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"
#include "Thyra_VectorBase.hpp"

namespace Albany {

AggregateScalarResponseFunction::AggregateScalarResponseFunction(
    const Teuchos::RCP<Teuchos_Comm const>&                     comm,
    const Teuchos::Array<Teuchos::RCP<ScalarResponseFunction>>& responses_)
    : SamplingBasedScalarResponseFunction(comm), responses(responses_)
{
  // Nothing to be done here
}

void
AggregateScalarResponseFunction::setup()
{
  typedef Teuchos::Array<Teuchos::RCP<ScalarResponseFunction>> ResponseArray;
  for (ResponseArray::iterator it = responses.begin(), it_end = responses.end(); it != it_end; ++it) {
    (*it)->setup();
  }

  // Now that all responses are setup, build the product vector space
  Teuchos::Array<Teuchos::RCP<Thyra_VectorSpace const>> vss(responses.size());
  for (int i = 0; i < responses.size(); ++i) {
    vss[i] = responses[i]->responseVectorSpace();
  }

  productVectorSpace = Thyra::productVectorSpace(vss());
}

void
AggregateScalarResponseFunction::postRegSetup()
{
  typedef Teuchos::Array<Teuchos::RCP<ScalarResponseFunction>> ResponseArray;
  for (ResponseArray::iterator it = responses.begin(), it_end = responses.end(); it != it_end; ++it) {
    (*it)->postRegSetup();
  }
}

unsigned int
AggregateScalarResponseFunction::numResponses() const
{
  unsigned int n = 0;
  for (int i = 0; i < responses.size(); i++) n += responses[i]->numResponses();
  return n;
}

void
AggregateScalarResponseFunction::evaluateResponse(
    double const                            current_time,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& xdot,
    Teuchos::RCP<Thyra_Vector const> const& xdotdot,
    const Teuchos::Array<ParamVec>&         p,
    Teuchos::RCP<Thyra_Vector> const&       g)
{
  if (g.is_null()) {
    return;
  }

  // NOTE: You CANNOT use ProductVector's if you want to maintain support for
  // EpetraExt::ModelEvaluator,
  //       since that class has no knowledge of vector spaces, and would try to
  //       build a monolithic map for the aggregate response. For now, stick
  //       with monolithic responses and manual copies

  /*
   * // Cast response to product vector
   * auto g_prod = getProductVector(g);
   *
   * for (unsigned int i=0; i<responses.size(); i++) {
   *   // Evaluate response function
   *   responses[i]->evaluateResponse(current_time, x, xdot, xdotdot, p,
   * g_prod->getNonconstVectorBlock(i));
   * }
   */

  Teuchos::ArrayRCP<ST>       g_data = getNonconstLocalData(g);  // We already checked g is not null
  Teuchos::ArrayRCP<const ST> gi_data;

  unsigned int offset = 0;
  for (unsigned int i = 0; i < responses.size(); i++) {
    // Create Thyra_Vector for response function
    Teuchos::RCP<Thyra_Vector> g_i = Thyra::createMember(productVectorSpace->getBlock(i));
    g_i->assign(0.0);

    gi_data = getLocalData(g_i.getConst());

    // Evaluate response function
    responses[i]->evaluateResponse(current_time, x, xdot, xdotdot, p, g_i);

    // Copy into the monolithic vector
    for (unsigned int j = 0; j < responses[i]->numResponses(); ++j) {
      g_data[offset + j] = gi_data[j];
    }

    // Update offset
    offset += responses[i]->numResponses();
  }
}

void
AggregateScalarResponseFunction::evaluateGradient(
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
    Teuchos::RCP<Thyra_MultiVector> const&  dg_dp)
{
  // NOTE: You CANNOT use ProductVector's if you want to maintain support for
  // EpetraExt::ModelEvaluator,
  //       since that class has no knowledge of vector spaces, and would try to
  //       build a monolithic map for the aggregate response. For now, stick
  //       with monolithic responses and manual copies

  /*
   * // Cast response (and param deriv) to product (multi)vector
   * auto g_prod     = getProductVector(g);
   */

  Teuchos::ArrayRCP<ST>                    g_data;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> dgdp_data;

  if (!g.is_null()) {
    g_data = getNonconstLocalData(g);
  }
  if (!dg_dp.is_null()) {
    dgdp_data = getNonconstLocalData(dg_dp);
  }

  Teuchos::ArrayRCP<const ST>                    g_i_data;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> dgdp_i_data;

  unsigned int offset = 0;
  for (unsigned int i = 0; i < responses.size(); i++) {
    auto vs_i = productVectorSpace->getBlock(i);

    // dg_dx, dg_dxdot, and dg_dxdotdot for the i-th response are simply
    // a subview of the columns of the corresponding input MV's, at the proper
    // offset
    Teuchos::Range1D colRange(offset, offset + vs_i->dim() - 1);

    // Create Thyra_(Multi)Vector's for response function
    Teuchos::RCP<Thyra_Vector>      g_i;
    Teuchos::RCP<Thyra_MultiVector> dgdx_i, dgdxdot_i, dgdxdotdot_i, dgdp_i;
    if (!g.is_null()) {
      g_i      = Thyra::createMember(vs_i);
      g_i_data = getLocalData(g_i.getConst());
    }
    if (!dg_dx.is_null()) {
      dgdx_i = dg_dx->subView(colRange);
    }
    if (!dg_dxdot.is_null()) {
      dgdxdot_i = dg_dxdot->subView(colRange);
    }
    if (!dg_dxdotdot.is_null()) {
      dgdxdotdot_i = dg_dxdotdot->subView(colRange);
    }
    if (!dg_dp.is_null()) {
      dgdp_i      = Thyra::createMembers(vs_i, dg_dp->domain()->dim());
      dgdp_i_data = getLocalData(dgdp_i.getConst());
    }

    // Evaluate response function
    responses[i]->evaluateGradient(
        current_time, x, xdot, xdotdot, p, deriv_p, g_i, dgdx_i, dgdxdot_i, dgdxdotdot_i, dgdp_i);

    // Copy into the monolithic (multi)vectors
    for (unsigned int j = 0; j < responses[i]->numResponses(); ++j) {
      if (!g.is_null()) {
        g_data[offset + j] = g_i_data[j];
      }
      if (!dg_dp.is_null()) {
        for (int col = 0; col < dg_dp->domain()->dim(); ++col) {
          dgdp_data[col][offset + j] = dgdp_i_data[col][j];
        }
      }
    }

    // Update the offset
    offset += vs_i->dim();
  }
}

}  // namespace Albany
