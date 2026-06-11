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

  ConstScalarField displacement_;

  // Evaluated MDFields
  ScalarField stress_;
  ScalarField Fp_;
  ScalarField ice_sat_state_;
  ScalarField backStress_;
  ScalarField capParameter_;
  ScalarField eqps_;
  ScalarField volPlasticStrain_;

  // Failure indicators (each normalized so 1 means exhaustion of the
  // underlying mechanism) and the death bookkeeping fields.
  ScalarField tension_indicator_;
  ScalarField backstress_indicator_;
  ScalarField crush_indicator_;
  ScalarField eqps_indicator_;
  ScalarField angle_indicator_;
  ScalarField displacement_indicator_;
  ScalarField tilt_angle_;
  ScalarField failed_;
  ScalarField dead_;

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

  // Per-(cell, pt) failure-mode bitmask, carried as an STK-backed element
  // state so it follows each cell correctly when the discretization
  // rebuilds its worksets (erosion re-buckets the mesh). Bits encode:
  //   0x01 = tension (stress at the shear-surface apex, Ff - N -> 0)
  //   0x02 = backstress saturation (sqrt(J2(alpha)) -> N, G^alpha = 0)
  //   0x04 = crush exhaustion (|eps_v^p(kappa)| -> W, cap lock)
  //   0x08 = tilt angle
  //   0x10 = displacement norm
  //   0x20 = equivalent plastic strain limit
  // Once a bit is set at (cell, pt) it stays set. failure_modes_old_ holds
  // the value converged at the previous step; failure_modes_ is this
  // fill's updated value (old | newly tripped bits), saved back to the
  // state. The mask is the source of truth for failure_state and the
  // cell-death predicate.
  Albany::MDArray failure_modes_old_;
  ScalarField     failure_modes_;

  // Live death-status signal shared with the assembly (scatter skips dead
  // cells); populated by the ACE solver from the converged cell_death
  // state, written here when a cell dies mid-fill.
  Teuchos::RCP<std::vector<double>> death_status_vec_;
  bool                              has_failed_old_{false};

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

  // ACE environment (Phase 3): depth profiles of porosity (overrides the
  // crushable volume W) and peat (raises the eqps limit), sea level
  // against time, and the ocean-exposure weakening factors applied to
  // submerged cells of the erodible sets.
  std::vector<RealType> z_above_mean_sea_level_;
  std::vector<RealType> porosity_from_file_;
  std::vector<RealType> peat_from_file_;
  std::vector<RealType> sea_level_;
  std::vector<RealType> time_;
  RealType              bulk_porosity_{0.0};  // constant fallback; 0 = no W override
  RealType              cohesion_weakening_{1.0};
  RealType              stiffness_weakening_{1.0};
  RealType              eqps_limit_weakening_{1.0};
  bool                  have_height_{false};

  // Per-cell flag for "*-erodible" side-set membership (ocean exposure).
  Teuchos::ArrayRCP<std::uint8_t> cell_is_erodible_;

  RealType substep_tolerance_;
  int      max_substeps_;

  // Failure criteria. Each indicator reaches 1 at exhaustion of its
  // mechanism; a criterion is enabled by a positive threshold (or limit)
  // and disabled at the default 0. The backstress and crush indicators
  // approach 1 only asymptotically, so their thresholds must be < 1; the
  // tension indicator must trip strictly below 1 because the yield
  // function has a spurious branch beyond the apex (see the developers
  // guide).
  RealType tension_indicator_threshold_{0.0};
  RealType backstress_indicator_threshold_{0.0};
  RealType crush_indicator_threshold_{0.0};
  RealType maximum_eqps_{0.0};
  RealType critical_angle_{0.0};
  RealType maximum_displacement_{0.0};
  bool     disable_erosion_{false};

  // Number of integration points whose failure_modes mask must be
  // non-zero before the cell is declared dead. 0 means "all"
  // (= num_pts_); other values are clamped into [1, num_pts_].
  int num_failed_pts_for_death_{0};

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
