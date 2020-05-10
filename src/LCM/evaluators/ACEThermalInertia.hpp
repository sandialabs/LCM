// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ACETHERMALINERTIA_HPP
#define ACETHERMALINERTIA_HPP

#include "Albany_MaterialDatabase.hpp"
#include "Albany_Types.hpp"
#include "Albany_config.h"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
/**
 * \brief Evaluates thermal inertia.

This class may be used in two ways.

1. The simplest is to use a constant thermal inertia across the entire
domain (one element block, one material), say with a value of 5.0. In this case,
one would declare at the "Problem" level, that a constant thermal inertia
was being used, and its value was 5.0:

<ParameterList name="Problem">
   ...
    <ParameterList name="ACEThermalInertia">
       <Parameter name="ACEThermalInertia Type" type="string"
value="Constant"/> <Parameter name="Value" type="double" value="5.0"/>
    </ParameterList>
</ParameterList>

2. The other extreme is to have a multiple element block problem, say 3, with
each element block corresponding to a material. Each element block has its own
field manager, and different evaluators are used in each element block. 

 */

template <typename EvalT, typename Traits>
class ACEThermalInertia : public PHX::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalT, Traits>,
                            public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ACEThermalInertia(Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

  ScalarT&
  getValue(std::string const& n);

 private:
  //! Validate the name strings under "ACEThermalInertia" section in xml input
  //! file,
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidThermalCondParameters() const;

  bool is_constant;

  std::size_t                                           numQPs;
  std::size_t                                           numDims;
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> coordVec;
  PHX::MDField<ScalarT, Cell, QuadPoint>                thermal_inertia;

  //! Inertia type
  std::string type;

  //! Constant value
  ScalarT constant_value;

  //! Values of the random variables
  Teuchos::Array<ScalarT> rv;

  //! Material database - holds thermal inertia among other quantities
  Teuchos::RCP<Albany::MaterialDatabase> materialDB;

  //! Convenience function to initialize constant thermal inertia
  void
  init_constant(ScalarT value, Teuchos::ParameterList& p);

};
}  // namespace LCM

#endif
