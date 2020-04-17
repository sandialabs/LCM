// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

namespace PHAL {

template <typename EvalT, typename Traits>
DummyResidual<EvalT, Traits>::DummyResidual(
    Teuchos::ParameterList const&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : solution(p.get<std::string>("Solution Variable Name"), dl->node_scalar),
      residual(p.get<std::string>("Residual Variable Name"), dl->node_scalar)
{
  this->addDependentField(solution);
  this->addEvaluatedField(residual);

  this->setName("DummyResidual" + PHX::print<EvalT>());
}

template <typename EvalT, typename Traits>
void
DummyResidual<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(solution, fm);
  this->utils.setFieldData(residual, fm);
}

template <typename EvalT, typename Traits>
void DummyResidual<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData /*workset*/)
{
  // Note: if solution!=0 (for EvalT=Residual), then this will trigger
  //       one iteration of the (non)linear solver.
  residual.deep_copy(solution);
}

}  // Namespace PHAL
