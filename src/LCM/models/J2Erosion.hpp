// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_J2Erosion_hpp)
#define LCM_J2Erosion_hpp

#include "ParallelConstitutiveModel.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
struct J2ErosionKernel : public ParallelKernel<EvalT, Traits>
{
  ///
  /// Constructor
  ///
  J2ErosionKernel(ConstitutiveModel<EvalT, Traits>& model, Teuchos::ParameterList* p, Teuchos::RCP<Albany::Layouts> const& dl);

  ///
  /// No copy constructor
  ///
  J2ErosionKernel(J2ErosionKernel const&) = delete;

  ///
  /// No copy assignment
  ///
  J2ErosionKernel&
  operator=(J2ErosionKernel const&) = delete;

  using ScalarT          = typename EvalT::ScalarT;
  using ScalarField      = PHX::MDField<ScalarT>;
  using ConstScalarField = PHX::MDField<ScalarT const>;
  using BaseKernel       = ParallelKernel<EvalT, Traits>;
  using Workset          = typename BaseKernel::Workset;

  using BaseKernel::field_name_map_;
  using BaseKernel::num_dims_;
  using BaseKernel::num_pts_;

  // optional temperature support
  using BaseKernel::density_;
  using BaseKernel::expansion_coeff_;
  using BaseKernel::have_temperature_;
  using BaseKernel::heat_capacity_;
  using BaseKernel::ref_temperature_;

  using BaseKernel::addStateVariable;
  using BaseKernel::setDependentField;
  using BaseKernel::setEvaluatedField;

  /// Pointer to NOX status test, allows the material model to force
  /// a global load step reduction
  using BaseKernel::nox_status_test_;

  // Input constant MDFields
  ConstScalarField def_grad_;
  ConstScalarField delta_time_;
  ConstScalarField elastic_modulus_;
  ConstScalarField hardening_modulus_;
  ConstScalarField J_;
  ConstScalarField poissons_ratio_;
  ConstScalarField yield_strength_;
  ConstScalarField temperature_;
  ConstScalarField ice_saturation_;
  ConstScalarField displacement_;
  ConstScalarField cumulative_time_old_;

  // Output MDFields
  ScalarField Fp_;
  ScalarField eqps_;
  ScalarField cumulative_time_;
  ScalarField source_;
  ScalarField stress_;
  ScalarField yield_surf_;
  ScalarField j2_stress_;
  ScalarField tilt_angle_;
  ScalarField failed_;

  // Failure indicators
  ScalarField yield_indicator_;
  ScalarField tensile_indicator_;
  ScalarField strain_indicator_;
  ScalarField angle_indicator_;
  ScalarField displacement_indicator_;

  // Workspace arrays
  Albany::MDArray Fp_old_;
  Albany::MDArray eqps_old_;
  //Albany::MDArray cumulative_time_old_;

  bool                       have_cell_boundary_indicator_{false};
  Teuchos::ArrayRCP<double*> cell_boundary_indicator_;

  PHX::MDField<bool> exposed_;

  // Baseline constants
  RealType sat_mod_{0.0};
  RealType sat_exp_{0.0};
  RealType critical_angle_{0.0};
  RealType bulk_porosity_{0.0};
  RealType soil_yield_strength_{0.0};
  RealType residual_elastic_modulus_{0.0};
  RealType Y_weakening_factor_{1.0};
  RealType E_weakening_factor_{1.0};
  RealType SL_weakening_factor_{1.0};
  RealType tensile_strength_{0.0};
  RealType strain_limit_{0.0};

  // Params with depth or time:
  std::vector<RealType> z_above_mean_sea_level_;
  std::vector<RealType> peat_from_file_;
  std::vector<RealType> porosity_from_file_;
  std::vector<RealType> sea_level_;
  std::vector<RealType> time_;

  // Sea level arrays
  RealType current_time_{0.0};
  RealType time_step_{0.0};

  void
  init(Workset& workset, FieldMap<ScalarT const>& dep_fields, FieldMap<ScalarT>& eval_fields);

  KOKKOS_INLINE_FUNCTION
  void
  operator()(int cell, int pt) const;
};

template <typename EvalT, typename Traits>
class J2Erosion : public LCM::ParallelConstitutiveModel<EvalT, Traits, J2ErosionKernel<EvalT, Traits>>
{
 public:
  J2Erosion(Teuchos::ParameterList* p, const Teuchos::RCP<Albany::Layouts>& dl);
};
}  // namespace LCM
#endif  // LCM_J2Erosion_hpp
