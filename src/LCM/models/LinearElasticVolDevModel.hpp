// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_LinearElasticVolDevModel_hpp)
#define LCM_LinearElasticVolDevModel_hpp

#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {

///
/// Constitutive Model Base Class
///
template <typename EvalT, typename Traits>
class LinearElasticVolDevModel : public LCM::ConstitutiveModel<EvalT, Traits>
{
 public:
  using Base        = LCM::ConstitutiveModel<EvalT, Traits>;
  using DepFieldMap = typename Base::DepFieldMap;
  using FieldMap    = typename Base::FieldMap;

  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;
  using ConstitutiveModel<EvalT, Traits>::compute_energy_;
  using ConstitutiveModel<EvalT, Traits>::compute_tangent_;

  ///
  /// Constructor
  ///
  LinearElasticVolDevModel(Teuchos::ParameterList* p, const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual ~LinearElasticVolDevModel() {};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual void
  computeState(typename Traits::EvalData workset, DepFieldMap dep_fields, FieldMap eval_fields);

  virtual void
  computeStateParallel(typename Traits::EvalData workset, DepFieldMap dep_fields, FieldMap eval_fields)
  {
    ALBANY_ABORT("Not implemented.");
  }

  LinearElasticVolDevModel(const LinearElasticVolDevModel&) = delete;

  LinearElasticVolDevModel&
  operator=(const LinearElasticVolDevModel&) = delete;

 private:
  RealType bulk_modulus_{0.0};
  RealType shear_modulus_{0.0};
};
}  // namespace LCM

#endif
