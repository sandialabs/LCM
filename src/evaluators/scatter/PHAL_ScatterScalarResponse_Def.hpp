// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"
#include "PHAL_Utilities.hpp"
#include "Phalanx_DataLayout.hpp"

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace PHAL {

template <typename EvalT, typename Traits>
ScatterScalarResponseBase<EvalT, Traits>::ScatterScalarResponseBase(
    Teuchos::ParameterList const&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
{
  setup(p, dl);
}

template <typename EvalT, typename Traits>
void
ScatterScalarResponseBase<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(global_response, fm);
  if (!stand_alone) this->utils.setFieldData(global_response_eval, fm);
}

template <typename EvalT, typename Traits>
void
ScatterScalarResponseBase<EvalT, Traits>::setup(
    Teuchos::ParameterList const&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
{
  stand_alone = p.get<bool>("Stand-alone Evaluator");

  // Setup fields we require
  auto global_response_tag =
      p.get<PHX::Tag<ScalarT>>("Global Response Field Tag");
  global_response = decltype(global_response)(global_response_tag);
  if (stand_alone) {
    this->addDependentField(global_response);
  } else {
    global_response_eval = decltype(global_response_eval)(global_response_tag);
    this->addEvaluatedField(global_response_eval);
  }

  // Setup field we evaluate
  std::string fieldName = global_response_tag.name() + " Scatter Response";
  scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));
  this->addEvaluatedField(*scatter_operation);

  //! get and validate parameter list
  Teuchos::ParameterList* plist =
      p.get<Teuchos::ParameterList*>("Parameter List");
  if (stand_alone) {
    Teuchos::RCP<Teuchos::ParameterList const> reflist =
        this->getValidResponseParameters();
    plist->validateParameters(*reflist, 0);
  }

  if (stand_alone) this->setName(fieldName + " Scatter Response");
}

template <typename EvalT, typename Traits>
Teuchos::RCP<Teuchos::ParameterList const>
ScatterScalarResponseBase<EvalT, Traits>::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      rcp(new Teuchos::ParameterList("Valid ScatterScalarResponse Params"));
  return validPL;
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template <typename Traits>
ScatterScalarResponse<PHAL::AlbanyTraits::Residual, Traits>::
    ScatterScalarResponse(
        Teuchos::ParameterList const&        p,
        const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setup(p, dl);
}

template <typename Traits>
void
ScatterScalarResponse<PHAL::AlbanyTraits::Residual, Traits>::postEvaluate(
    typename Traits::PostEvalData workset)
{
  // Here we scatter the *global* response
  Teuchos::RCP<Thyra_Vector> g = workset.g;  // Tpetra version
  if (g != Teuchos::null) {
    Teuchos::ArrayRCP<ST> g_nonconstView = Albany::getNonconstLocalData(g);
    for (PHAL::MDFieldIterator<ScalarT const> gr(this->global_response);
         !gr.done();
         ++gr) {
      g_nonconstView[gr.idx()] = *gr;
    }
  }
}

}  // namespace PHAL
