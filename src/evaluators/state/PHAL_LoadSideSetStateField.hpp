// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_LOAD_SIDE_SET_STATE_FIELD_HPP
#define PHAL_LOAD_SIDE_SET_STATE_FIELD_HPP

#include "PHAL_Utilities.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Teuchos_ParameterList.hpp"

namespace PHAL {
/** \brief LoadSideSetStatField

*/

template <typename EvalT, typename Traits, typename ScalarType>
class LoadSideSetStateFieldBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                  public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  LoadSideSetStateFieldBase(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& fm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  PHX::MDField<ScalarType> field;

  std::string sideSetName;
  std::string fieldName;
  std::string stateName;
};

template <typename EvalT, typename Traits>
using LoadSideSetStateFieldST =
    LoadSideSetStateFieldBase<EvalT, Traits, typename EvalT::ScalarT>;

template <typename EvalT, typename Traits>
using LoadSideSetStateFieldPST =
    LoadSideSetStateFieldBase<EvalT, Traits, typename EvalT::ParamScalarT>;

template <typename EvalT, typename Traits>
using LoadSideSetStateFieldMST =
    LoadSideSetStateFieldBase<EvalT, Traits, typename EvalT::MeshScalarT>;

template <typename EvalT, typename Traits>
using LoadSideSetStateFieldRT =
    LoadSideSetStateFieldBase<EvalT, Traits, RealType>;

// The default is the ParamScalarT
template <typename EvalT, typename Traits>
using LoadSideSetStateField =
    LoadSideSetStateFieldBase<EvalT, Traits, typename EvalT::ParamScalarT>;

}  // Namespace PHAL

#endif  // PHAL_LOAD_SIDE_SET_STATE_FIELD_HPP
