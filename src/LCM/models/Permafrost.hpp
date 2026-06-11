// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_Permafrost_hpp)
#define LCM_Permafrost_hpp

#include <MiniTensor.h>

#include "KernelConstitutiveModel.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
struct PermafrostKernel : public ParallelKernel<EvalT, Traits>
{
  ///
  /// Constructor
  ///
  PermafrostKernel(ConstitutiveModel<EvalT, Traits>& model, Teuchos::ParameterList* p, Teuchos::RCP<Albany::Layouts> const& dl);

  ///
  /// No copy constructor
  ///
  PermafrostKernel(PermafrostKernel const&) = delete;

  ///
  /// No copy assignment
  ///
  PermafrostKernel&
  operator=(PermafrostKernel const&) = delete;

  using ScalarT          = typename EvalT::ScalarT;
  using ScalarField      = PHX::MDField<ScalarT>;
  using ConstScalarField = PHX::MDField<ScalarT const>;
  using BaseKernel       = ParallelKernel<EvalT, Traits>;
  using Workset          = typename BaseKernel::Workset;

  using BaseKernel::field_name_map_;
  using BaseKernel::num_dims_;
  using BaseKernel::num_pts_;

  using BaseKernel::addStateVariable;
  using BaseKernel::setDependentField;
  using BaseKernel::setEvaluatedField;

  using BaseKernel::expansion_coeff_;
  using BaseKernel::have_temperature_;
  using BaseKernel::ref_temperature_;

  using BaseKernel::nox_status_test_;

  // Dependent MDFields
  ConstScalarField strain_;
  ConstScalarField def_grad_;
  ConstScalarField J_;
  ConstScalarField ice_saturation_;
  ConstScalarField temperature_;

  // Evaluated MDFields
  ScalarField stress_;
  ScalarField Fp_;
  ScalarField ice_sat_state_;
  ScalarField backStress_;
  ScalarField capParameter_;
  ScalarField eqps_;
  ScalarField volPlasticStrain_;

  // Old state arrays
  Albany::MDArray stress_old_;
  Albany::MDArray strain_old_;
  Albany::MDArray def_grad_old_;
  Albany::MDArray Fp_old_;
  Albany::MDArray backStress_old_;
  Albany::MDArray capParameter_old_;
  Albany::MDArray eqps_old_;
  Albany::MDArray volPlasticStrain_old_;
  Albany::MDArray ice_sat_state_old_;

  // Finite-deformation (exponential/logarithmic map) kinematics flag
  bool finite_deformation_;

  // End-member material parameters. The frozen set carries the
  // ice-bonding-dependent quantities; the thawed set additionally
  // carries the sediment-skeleton (friction/shape) quantities shared by
  // both ends. See the saturation-to-parameter map in
  // doc/developersGuide/cap_plasticity.tex, Section "Planned Extension:
  // Permafrost and Erosion".
  struct EndMember
  {
    RealType K;       // bulk modulus
    RealType G;       // shear modulus
    RealType A;       // shear strength asymptote
    RealType C;       // strength deficit at zero pressure
    RealType N;       // yield offset below the failure envelope
    RealType kappa0;  // initial cap branch point
    RealType W;       // maximum compactive volumetric plastic strain
    RealType D1;      // crush-curve shape (linear)
    RealType calpha;  // kinematic hardening rate
  };

  EndMember frozen_;
  EndMember thawed_;

  // Sediment-skeleton (saturation-independent) parameters, read from the
  // thawed sublist.
  RealType D_;
  RealType theta_;
  RealType L_;
  RealType phi_;
  RealType R_;
  RealType Q_;
  RealType psi_;
  RealType D2_;

  // Numerical guard: cap on the effective Poisson ratio (the thawed
  // saturated limit nu -> 0.5 volumetrically locks low-order elements).
  RealType nu_max_;

  // Ice-saturation source: ACE field, time table, or constant.
  bool                     have_ice_field_;
  Teuchos::Array<RealType> ice_sat_times_;
  Teuchos::Array<RealType> ice_sat_values_;
  RealType                 ice_sat_constant_;

  RealType substep_tolerance_;
  int      max_substeps_;

  RealType current_time_{0.0};

  void
  init(Workset& workset, FieldMap<ScalarT const>& dep_fields, FieldMap<ScalarT>& eval_fields);

  KOKKOS_INLINE_FUNCTION
  void
  operator()(int cell, int pt) const;
};

template <typename EvalT, typename Traits>
class Permafrost : public LCM::KernelConstitutiveModel<EvalT, Traits, PermafrostKernel<EvalT, Traits>>
{
 public:
  Permafrost(Teuchos::ParameterList* p, const Teuchos::RCP<Albany::Layouts>& dl);
};

}  // namespace LCM

#endif  // LCM_Permafrost_hpp
