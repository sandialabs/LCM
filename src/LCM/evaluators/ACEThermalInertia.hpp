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
 * \brief Evaluates thermal inertia for ACE stand-alone thermal problem.
 */

template <typename EvalT, typename Traits>
class ACEThermalInertia : public PHX::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalT, Traits>,
                            public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ACEThermalInertia(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl); 

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

  ScalarT&
  getValue(std::string const& n);

 private:
  //! Validate the name strings under "ACE Thermal Inertia" section in input file 
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidThermalCondParameters() const;

  std::size_t                                           num_qps_;
  std::size_t                                           num_dims_;
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> coord_vec_;
  PHX::MDField<ScalarT, Cell, QuadPoint>                thermal_inertia_;

  //! Constant value
  ScalarT constant_value_{0.0};

  //! Material database - holds thermal inertia among other quantities
  Teuchos::RCP<Albany::MaterialDatabase> material_db_;

};


}  // namespace LCM

#endif
