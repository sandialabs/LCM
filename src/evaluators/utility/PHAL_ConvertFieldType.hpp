// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_CONVERTFIELDTYPE_HPP
#define PHAL_CONVERTFIELDTYPE_HPP

#include "Albany_Layouts.hpp"
#include "Albany_Types.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template <
    typename EvalT,
    typename Traits,
    typename InputType,
    typename OutputType>
class ConvertFieldType : public PHX::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ConvertFieldType(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  // Input:
  PHX::MDField<const InputType> in_field;
  // Output:
  PHX::MDField<OutputType> out_field;
};

template <typename EvalT, typename Traits>
using ConvertFieldTypeRTtoMST =
    ConvertFieldType<EvalT, Traits, RealType, typename EvalT::MeshScalarT>;

template <typename EvalT, typename Traits>
using ConvertFieldTypeRTtoPST =
    ConvertFieldType<EvalT, Traits, RealType, typename EvalT::ParamScalarT>;

template <typename EvalT, typename Traits>
using ConvertFieldTypeRTtoST =
    ConvertFieldType<EvalT, Traits, RealType, typename EvalT::ScalarT>;

template <typename EvalT, typename Traits>
using ConvertFieldTypeMSTtoPST = ConvertFieldType<
    EvalT,
    Traits,
    typename EvalT::MeshScalarT,
    typename EvalT::ParamScalarT>;

template <typename EvalT, typename Traits>
using ConvertFieldTypeMSTtoST = ConvertFieldType<
    EvalT,
    Traits,
    typename EvalT::MeshScalarT,
    typename EvalT::ScalarT>;

template <typename EvalT, typename Traits>
using ConvertFieldTypePSTtoST = ConvertFieldType<
    EvalT,
    Traits,
    typename EvalT::ParamScalarT,
    typename EvalT::ScalarT>;

}  // Namespace PHAL

#endif  // PHAL_DOF_INTERPOLATION_HPP
