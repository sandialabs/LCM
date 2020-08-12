// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include "Adapt_ElementSizeField.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_Utils.hpp"
#include "IPtoNodalField.hpp"
#include "PHAL_ResponseFieldIntegral.hpp"
#include "PHAL_ResponseSquaredL2Difference.hpp"
#include "PHAL_ResponseSquaredL2DifferenceSide.hpp"
#include "PHAL_ResponseThermalEnergy.hpp"
#include "PHAL_SaveNodalField.hpp"
#include "ProjectIPtoNodalField.hpp"

template <typename EvalT, typename Traits>
Albany::ResponseUtilities<EvalT, Traits>::ResponseUtilities(Teuchos::RCP<Albany::Layouts> dl_) : dl(dl_)
{
}

template <typename EvalT, typename Traits>
Teuchos::RCP<const PHX::FieldTag>
Albany::ResponseUtilities<EvalT, Traits>::constructResponses(
    PHX::FieldManager<PHAL::AlbanyTraits>& fm,
    Teuchos::ParameterList&                responseParams,
    Teuchos::RCP<Teuchos::ParameterList>   paramsFromProblem,
    Albany::StateManager&                  stateMgr,
    Albany::MeshSpecsStruct const*         meshSpecs)
{
  using PHX::DataLayout;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  std::string        responseName = responseParams.get<std::string>("Name");
  RCP<ParameterList> p            = rcp(new ParameterList);
  p->set<ParameterList*>("Parameter List", &responseParams);
  p->set<RCP<ParameterList>>("Parameters From Problem", paramsFromProblem);
  RCP<PHX::Evaluator<Traits>> res_ev;

  if (responseName == "Squared L2 Difference Source ST Target ST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSST_TST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Source ST Target MST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSST_TMST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Source ST Target PST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSST_TPST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Source PST Target ST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSPST_TST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Source PST Target MST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSPST_TMST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Source PST Target PST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSPST_TPST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Source MST Target ST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSMST_TST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Source MST Target MST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSMST_TMST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Source MST Target PST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSMST_TPST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Side Source ST Target ST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSST_TST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Side Source ST Target MST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSST_TMST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Side Source ST Target PST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSST_TPST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Side Source PST Target ST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSPST_TST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Side Source PST Target MST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSPST_TMST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Side Source PST Target PST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSPST_TPST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Side Source MST Target ST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSMST_TST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Side Source MST Target MST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSMST_TMST<EvalT, Traits>(*p, dl));
  } else if (responseName == "Squared L2 Difference Side Source MST Target PST") {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSMST_TPST<EvalT, Traits>(*p, dl));
  } else if (responseName == "PHAL Field Integral") {
    res_ev = rcp(new PHAL::ResponseFieldIntegral<EvalT, Traits>(*p, dl));
  } else if (responseName == "PHAL Thermal Energy") {
    res_ev = rcp(new PHAL::ResponseThermalEnergy<EvalT, Traits>(*p, dl));
  } else if (responseName == "Element Size Field") {
    p->set<Albany::StateManager*>("State Manager Ptr", &stateMgr);
    p->set<RCP<DataLayout>>("Dummy Data Layout", dl->dummy);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<std::string>("Weights Name", "Weights");

    res_ev = rcp(new Adapt::ElementSizeField<EvalT, Traits>(*p, dl));
  } else if (responseName == "Save Nodal Fields") {
    p->set<Albany::StateManager*>("State Manager Ptr", &stateMgr);

    res_ev = rcp(new PHAL::SaveNodalField<EvalT, Traits>(*p, dl));
  } else if (responseName == "IP to Nodal Field") {
    p->set<Albany::StateManager*>("State Manager Ptr", &stateMgr);
    p->set<RCP<DataLayout>>("Dummy Data Layout", dl->dummy);
    res_ev = rcp(new LCM::IPtoNodalField<EvalT, Traits>(*p, dl, meshSpecs));
  } else if (responseName == "Project IP to Nodal Field") {
    p->set<Albany::StateManager*>("State Manager Ptr", &stateMgr);
    p->set<RCP<DataLayout>>("Dummy Data Layout", dl->dummy);
    p->set<std::string>("BF Name", "BF");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");

    res_ev = rcp(new LCM::ProjectIPtoNodalField<EvalT, Traits>(*p, dl, meshSpecs));
  }

  else
    ALBANY_ABORT(
        std::endl
        << "Error!  Unknown response function " << responseName << "!" << std::endl
        << "Supplied parameter list is " << std::endl
        << responseParams);

  // Register the evaluator
  fm.template registerEvaluator<EvalT>(res_ev);

  // Fetch the response tag. Usually it is the tag of the first evaluated field
  Teuchos::RCP<const PHX::FieldTag> ev_tag = res_ev->evaluatedFields()[0];

  // The response tag is not the same of the evaluated field tag for
  // PHAL::ScatterScalarResponse
  Teuchos::RCP<PHAL::ScatterScalarResponseBase<EvalT, Traits>> sc_resp;
  sc_resp = Teuchos::rcp_dynamic_cast<PHAL::ScatterScalarResponseBase<EvalT, Traits>>(res_ev);
  if (sc_resp != Teuchos::null) {
    ev_tag = sc_resp->getResponseFieldTag();
  }

  // Require the response tag;
  fm.requireField<EvalT>(*ev_tag);

  return ev_tag;
}
