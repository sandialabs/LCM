// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_NeohookeanModel_hpp)
#define LCM_NeohookeanModel_hpp

#include "Albany_Layouts.hpp"
#include "ConstitutiveModel.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {

///
/// \brief Neohookean Model
template <typename EvalT, typename Traits>
class NeohookeanModel : public LCM::ConstitutiveModel<EvalT, Traits>
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

  // optional temperature support
  using ConstitutiveModel<EvalT, Traits>::have_temperature_;
  using ConstitutiveModel<EvalT, Traits>::expansion_coeff_;
  using ConstitutiveModel<EvalT, Traits>::ref_temperature_;
  using ConstitutiveModel<EvalT, Traits>::heat_capacity_;
  using ConstitutiveModel<EvalT, Traits>::density_;
  using ConstitutiveModel<EvalT, Traits>::temperature_;

  ///
  /// Constructor
  ///
  NeohookeanModel(Teuchos::ParameterList* p, Teuchos::RCP<Albany::Layouts> const& dl);

  ///
  /// Virtual Destructor
  ///
  virtual ~NeohookeanModel() {};

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

  // No copy constructor or copy assignment.
  NeohookeanModel(NeohookeanModel const&) = delete;
  NeohookeanModel&
  operator=(NeohookeanModel const&) = delete;
};

}  // namespace LCM
#endif
