// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_SCATTER_SCALAR_RESPONSE_HPP
#define PHAL_SCATTER_SCALAR_RESPONSE_HPP

#include "Albany_Layouts.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Teuchos_ParameterList.hpp"

namespace PHAL {

/** \brief Handles scattering of scalar response functions
 *
 * Base implementation useable by specializations below
 */
template <typename EvalT, typename Traits>
class ScatterScalarResponseBase
    : public virtual PHX::EvaluatorWithBaseImpl<Traits>,
      public virtual PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ScatterScalarResponseBase(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData /* d */) {}

  //! Get tag for response field (to determine the number of responses)
  Teuchos::RCP<const PHX::FieldTag>
  getResponseFieldTag() const
  {
    return global_response.fieldTag().clone();
  }

  //! Get tag for evaluated field (required field for field manager)
  Teuchos::RCP<const PHX::FieldTag>
  getEvaluatedFieldTag() const
  {
    return scatter_operation;
  }

 protected:
  // Default constructor for child classes
  ScatterScalarResponseBase() {}

  // Child classes should call setup once p is filled out
  void
  setup(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  Teuchos::RCP<Teuchos::ParameterList const>
  getValidResponseParameters() const;

 protected:
  typedef typename EvalT::ScalarT ScalarT;
  bool                            stand_alone;
  PHX::MDField<ScalarT const>     global_response;
  PHX::MDField<ScalarT>           global_response_eval;
  Teuchos::RCP<PHX::FieldTag>     scatter_operation;
};

template <typename EvalT, typename Traits>
class ScatterScalarResponse
{
};

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************

// **************************************************************
// Residual
// **************************************************************
template <typename Traits>
class ScatterScalarResponse<PHAL::AlbanyTraits::Residual, Traits>
    : public ScatterScalarResponseBase<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  ScatterScalarResponse(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);
  void
  postEvaluate(typename Traits::PostEvalData d);

 protected:
  typedef PHAL::AlbanyTraits::Residual EvalT;
  ScatterScalarResponse() {}
  void
  setup(
      Teuchos::ParameterList const&        p,
      const Teuchos::RCP<Albany::Layouts>& dl)
  {
    ScatterScalarResponseBase<EvalT, Traits>::setup(p, dl);
  }

 private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
};

// **************************************************************
// Jacobian -- No implementation can be provided
// **************************************************************

}  // namespace PHAL

#endif
