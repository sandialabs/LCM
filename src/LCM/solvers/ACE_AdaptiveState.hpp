
// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_ACEAdaptiveState_hpp)
#define LCM_ACEAdaptiveState_hpp

#include "Albany_ScalarOrdinalTypes.hpp"
#include "Thyra_AdaptiveSolutionManager.hpp"
#include "Thyra_ModelEvaluatorDelegatorBase.hpp"

namespace LCM {

class ACEAdaptiveState : public Thyra::AdaptiveStateBase
{
 public:
  ACEAdaptiveState(Teuchos::RCP<Thyra::ModelEvaluator<ST>> const& model) : AdaptiveStateBase(model) {}
  ~ACEAdaptiveState() {}
  void
  buildSolutionGroup()
  {
  }
};

class ACEModelEvaluatorDelegator : public Thyra::ModelEvaluatorDelegatorBase<ST>
{
 public:
  ACEModelEvaluatorDelegator(Teuchos::RCP<Thyra::ModelEvaluator<ST>> const& model_rcp) : model_rcp_(model_rcp) {}
  Teuchos::RCP<Thyra::ModelEvaluator<ST>>
  getNonconstUnderlyingModel()
  {
    return model_rcp_;
  }

  Thyra::ModelEvaluatorBase::OutArgs<ST>
  createOutArgsImpl() const
  {
    return Thyra::ModelEvaluatorBase::OutArgs<ST>();
  }

  void
  evalModelImpl(Thyra::ModelEvaluatorBase::InArgs<ST> const&, Thyra::ModelEvaluatorBase::OutArgs<ST> const&) const
  {
  }

 private:
  Teuchos::RCP<Thyra::ModelEvaluator<ST>> model_rcp_;
};

}  // namespace LCM

#endif  // LCM_ACEAdaptiveState_hpp
