// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include "ThermoElasticityProblem.hpp"

#include "Albany_BCUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_Utils.hpp"
#include "PHAL_AlbanyTraits.hpp"

Albany::ThermoElasticityProblem::ThermoElasticityProblem(
    const Teuchos::RCP<Teuchos::ParameterList>& params_,
    const Teuchos::RCP<ParamLib>&               paramLib_,
    int const                                   numDim_)
    : Albany::AbstractProblem(params_, paramLib_, numDim_ + 1), haveSource(false), use_sdbcs_(false), numDim(numDim_)
{
  std::string& method = params->get("Name", "ThermoElasticity ");
  *out << "Problem Name = " << method << std::endl;

  haveSource = params->isSublist("Source Functions");

// Changing this ifdef changes ordering from  (X,Y,T) to (T,X,Y)
//#define NUMBER_T_FIRST
#if defined(NUMBER_T_FIRST)
  T_offset = 0;
  X_offset = 1;
#else
  X_offset = 0;
  T_offset = numDim;
#endif

  // the following function returns the problem information required for setting
  // the rigid body modes (RBMs) for elasticity problems
  // written by IK, Feb. 2012

  int numScalar    = 1;
  int nullSpaceDim = 0;
  if (numDim == 1) {
    nullSpaceDim = 0;
  } else {
    if (numDim == 2) {
      nullSpaceDim = 3;
    }
    if (numDim == 3) {
      nullSpaceDim = 6;
    }
  }

  rigidBodyModes->setParameters(numDim + 1, numDim, numScalar, nullSpaceDim);
}

Albany::ThermoElasticityProblem::~ThermoElasticityProblem() {}

void
Albany::ThermoElasticityProblem::buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> meshSpecs,
    Albany::StateManager&                                    stateMgr)
{
  /* Construct All Phalanx Evaluators */
  ALBANY_PANIC(meshSpecs.size() != 1, "Problem supports one Material Block");
  fm.resize(1);
  fm[0] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, BUILD_RESID_FM, Teuchos::null);
  constructDirichletEvaluators(*meshSpecs[0]);
}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
Albany::ThermoElasticityProblem::buildEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits>&      fm0,
    Albany::MeshSpecsStruct const&              meshSpecs,
    Albany::StateManager&                       stateMgr,
    Albany::FieldManagerChoice                  fmchoice,
    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<ThermoElasticityProblem>        op(*this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
Albany::ThermoElasticityProblem::constructDirichletEvaluators(Albany::MeshSpecsStruct const& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  dirichletNames[X_offset] = "X";
  if (numDim > 1) dirichletNames[X_offset + 1] = "Y";
  if (numDim > 2) dirichletNames[X_offset + 2] = "Z";
  dirichletNames[T_offset] = "T";
  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm         = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames, this->params, this->paramLib);
  use_sdbcs_  = dirUtils.useSDBCs();
  offsets_    = dirUtils.getOffsets();
  nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

Teuchos::RCP<Teuchos::ParameterList const>
Albany::ThermoElasticityProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getGenericProblemParams("ValidThermoElasticityProblemParams");

  validPL->sublist("Elastic Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");

  return validPL;
}

void
Albany::ThermoElasticityProblem::getAllocatedStates(
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> oldState_,
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> newState_) const
{
  oldState_ = oldState;
  newState_ = newState;
}
