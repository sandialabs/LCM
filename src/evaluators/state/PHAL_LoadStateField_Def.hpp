// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <string>
#include <vector>

#include "Albany_Macros.hpp"
#include "PHAL_LoadStateField.hpp"
#include "PHAL_Utilities.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template <typename EvalT, typename Traits, typename ScalarType>
LoadStateFieldBase<EvalT, Traits, ScalarType>::LoadStateFieldBase(
    Teuchos::ParameterList const& p)
{
  fieldName = p.get<std::string>("Field Name");
  stateName = p.get<std::string>("State Name");

  PHX::MDField<ScalarType> f(
      fieldName, p.get<Teuchos::RCP<PHX::DataLayout>>("State Field Layout"));
  data = f;

  this->addEvaluatedField(data);
  this->setName("LoadStateField(" + stateName + ")" + PHX::print<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits, typename ScalarType>
void
LoadStateFieldBase<EvalT, Traits, ScalarType>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(data, fm);

  d.fill_field_dependencies(this->dependentFields(), this->evaluatedFields());
}

// **********************************************************************
template <typename EvalT, typename Traits, typename ScalarType>
void
LoadStateFieldBase<EvalT, Traits, ScalarType>::evaluateFields(
    typename Traits::EvalData workset)
{
  // cout << "LoadStateFieldBase importing state " << stateName << " to field "
  //     << fieldName << " with size " << data.size() << endl;

  const Albany::MDArray& stateToLoad = (*workset.stateArrayPtr)[stateName];
  PHAL::MDFieldIterator<ScalarType> d(data);
  for (int i = 0; !d.done() && i < stateToLoad.size(); ++d, ++i)
    *d = stateToLoad[i];
  for (; !d.done(); ++d) *d = 0.;
}

template <typename EvalT, typename Traits>
LoadStateField<EvalT, Traits>::LoadStateField(Teuchos::ParameterList const& p)
{
  fieldName = p.get<std::string>("Field Name");
  stateName = p.get<std::string>("State Name");

  PHX::MDField<ParamScalarT> f(
      fieldName, p.get<Teuchos::RCP<PHX::DataLayout>>("State Field Layout"));
  data = f;

  this->addEvaluatedField(data);
  this->setName("Load State Field" + PHX::print<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
LoadStateField<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(data, fm);

  d.fill_field_dependencies(this->dependentFields(), this->evaluatedFields());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
LoadStateField<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  // cout << "LoadStateField importing state " << stateName << " to field "
  //     << fieldName << " with size " << data.size() << endl;

  const Albany::MDArray& stateToLoad = (*workset.stateArrayPtr)[stateName];
  PHAL::MDFieldIterator<ParamScalarT> d(data);
  for (int i = 0; !d.done() && i < stateToLoad.size(); ++d, ++i)
    *d = stateToLoad[i];
  for (; !d.done(); ++d) *d = 0.;
}

}  // namespace PHAL
