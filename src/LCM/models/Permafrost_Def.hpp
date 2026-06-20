// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
//
// Permafrost constitutive model: the three-invariant cap plasticity
// model specialized for frozen/thawing sediment in the ACE Arctic
// coastal erosion application, replacing J2Erosion. The integration
// and constitutive functions live in the shared CapIntegrator
// (verified against an independent reference); this kernel supplies
// parameters that vary per integration point with the ice saturation
// f. The design is recorded in
// doc/developersGuide/cap_plasticity.tex, Section "The Permafrost
// Model".
//
// Saturation-to-parameter map:
//   cohesion/bonding  A, C, N, calpha     linear between end members
//   crush             kappa0, W, D1       linear between end members
//   friction/shape    D, theta, L, phi,   thawed (sediment skeleton)
//                     R, Q, psi, D2       values, f-independent
//   elasticity        G(f) log-linear; K(f) linear; effective nu
//                     capped at nu_max (default 0.45) preserving G
//
// Ice saturation comes from the ACE_Ice_Saturation field
// (Have ACE Ice Saturation: true), from a time table (Ice Saturation
// Time Values / Ice Saturation Values), or from the constant
// Ice Saturation parameter (default 1.0, frozen). The cap-parameter
// state is initialized at the FROZEN kappa0: simulations are assumed to
// start from the frozen state.
//
// Failure and element death: per-point indicators, each
// normalized to reach 1 at exhaustion of its mechanism --
//   tension      stress at the shear-surface apex, Ff(I1) - N -> 0
//   backstress   kinematic-hardening saturation, sqrt(J2(alpha)) -> N
//   crush        cap exhaustion, |eps_v^p(kappa)| -> W
//   eqps         equivalent plastic strain against a user limit
// plus the structural tilt-angle and displacement criteria carried over
// from J2Erosion. Tripped modes accumulate in the failure_modes bitmask
// (an STK-backed element state); a cell dies when enough integration
// points have failed (default: all). The death bookkeeping
// (failure_state, cell_death, death_status_vec live propagation) follows
// J2Erosion exactly, so the ACE solver and the scatter evaluators
// consume Permafrost cells without modification.
//
// ACE environment: depth profiles of porosity (W override) and peat
// (eqps-limit boost), sea level against time, and ocean-exposure
// weakening of submerged erodible cells.

#include <MiniTensor.h>

#include "ACEcommon.hpp"
#include "Albany_Utils.hpp"
#include "CapIntegrator.hpp"
#include "Permafrost.hpp"

namespace LCM {

namespace {

// Piecewise-linear interpolation with end clamping (mirrored exactly by
// the verification reference implementation).
template <typename T>
T
interpolate_table(Teuchos::Array<RealType> const& times, Teuchos::Array<RealType> const& values, T const& t)
{
  auto const n = times.size();
  if (t <= times[0]) return T(values[0]);
  if (t >= times[n - 1]) return T(values[n - 1]);
  for (auto i = 1; i < n; ++i) {
    if (t <= times[i]) {
      return values[i - 1] + (values[i] - values[i - 1]) * (t - times[i - 1]) / (times[i] - times[i - 1]);
    }
  }
  return T(values[n - 1]);
}

}  // anonymous namespace

template <typename EvalT, typename Traits>
PermafrostKernel<EvalT, Traits>::PermafrostKernel(
    ConstitutiveModel<EvalT, Traits>& model,
    Teuchos::ParameterList*           p,
    Teuchos::RCP<Albany::Layouts> const& dl)
    : BaseKernel(model)
{
  finite_deformation_ = p->get<bool>("Finite Deformation", false);

  auto read_end_member = [](Teuchos::ParameterList& pl) {
    EndMember m;
    m.K      = pl.get<RealType>("K");
    m.G      = pl.get<RealType>("G");
    m.A      = pl.get<RealType>("A");
    m.C      = pl.get<RealType>("C");
    m.N      = pl.get<RealType>("N");
    m.kappa0 = pl.get<RealType>("kappa0");
    m.W      = pl.get<RealType>("W");
    m.D1     = pl.get<RealType>("D1");
    m.calpha = pl.get<RealType>("calpha");
    return m;
  };

  auto& frozen_pl = p->sublist("Frozen Parameters");
  auto& thawed_pl = p->sublist("Thawed Parameters");

  frozen_ = read_end_member(frozen_pl);
  thawed_ = read_end_member(thawed_pl);

  // Sediment-skeleton parameters, f-independent, from the thawed set.
  D_     = thawed_pl.get<RealType>("D");
  theta_ = thawed_pl.get<RealType>("theta");
  L_     = thawed_pl.get<RealType>("L");
  phi_   = thawed_pl.get<RealType>("phi");
  R_     = thawed_pl.get<RealType>("R");
  Q_     = thawed_pl.get<RealType>("Q");
  psi_   = thawed_pl.get<RealType>("psi", 1.0);
  D2_    = thawed_pl.get<RealType>("D2", 0.0);

  nu_max_ = p->get<RealType>("Maximum Poissons Ratio", 0.45);

  // Ice-saturation source
  have_ice_field_   = p->get<bool>("Have ACE Ice Saturation", false);
  ice_sat_constant_ = p->get<RealType>("Ice Saturation", 1.0);
  if (p->isParameter("Ice Saturation Time Values")) {
    ice_sat_times_  = p->get<Teuchos::Array<RealType>>("Ice Saturation Time Values");
    ice_sat_values_ = p->get<Teuchos::Array<RealType>>("Ice Saturation Values");
    ALBANY_ASSERT(
        ice_sat_times_.size() == ice_sat_values_.size() && ice_sat_times_.size() >= 2,
        "Ice Saturation Time Values / Values must have equal length >= 2");
  }

  // Sloan-style adaptive substepping of the explicit integration.
  substep_tolerance_ = p->get<RealType>("Substep Tolerance", 1.0e-4);
  max_substeps_      = p->get<int>("Maximum Substeps", 200);

  // Failure criteria: a positive threshold (or limit) enables each
  // criterion; the default 0 disables it. The thresholds for the
  // asymptotic indicators (tension, backstress, crush) must be < 1.
  tension_indicator_threshold_    = p->get<RealType>("Tension Indicator Threshold", 0.0);
  backstress_indicator_threshold_ = p->get<RealType>("Backstress Indicator Threshold", 0.0);
  crush_indicator_threshold_      = p->get<RealType>("Crush Indicator Threshold", 0.0);
  maximum_eqps_                   = p->get<RealType>("Maximum Equivalent Plastic Strain", 0.0);
  critical_angle_                 = p->get<RealType>("Critical Angle", 0.0);
  maximum_displacement_           = p->get<RealType>("Maximum Displacement", 0.0);
  maximum_distortion_             = p->get<RealType>("Maximum Distortion", 0.0);
  disable_erosion_                = p->get<bool>("Disable Erosion", false);
  num_failed_pts_for_death_       = p->get<int>("Failed Integration Points For Death", 0);

  // Gradual death: fade a marked cell's stiffness to zero over this many
  // accepted steps. 1 = instant removal (original behavior).
  death_steps_   = p->get<int>("Death Steps", 1);
  ALBANY_ASSERT(death_steps_ >= 1, "Permafrost: Death Steps must be >= 1");
  gradual_death_ = death_steps_ > 1;

  ALBANY_ASSERT(
      critical_angle_ <= 0.0 || finite_deformation_,
      "Permafrost: the Critical Angle (tilt) criterion requires Finite Deformation");
  ALBANY_ASSERT(
      maximum_distortion_ <= 0.0 || finite_deformation_,
      "Permafrost: the Maximum Distortion criterion requires Finite Deformation");
  ALBANY_ASSERT(
      maximum_distortion_ <= 0.0 || maximum_distortion_ > 1.0,
      "Permafrost: Maximum Distortion must be > 1 (the distortion measure is 1 at rest)");
  ALBANY_ASSERT(
      tension_indicator_threshold_ < 1.0 && backstress_indicator_threshold_ < 1.0 && crush_indicator_threshold_ < 1.0,
      "Permafrost: indicator thresholds must be < 1 (the indicators reach 1 only asymptotically)");

  // ACE environment. Depth profiles are functions of z above mean sea
  // level; when a porosity source is present (file or the constant
  // ACE Bulk Porosity) it overrides the crushable volume W at each
  // integration point -- the pore space is the crushable volume,
  // independent of what fills it. Peat raises the eqps limit
  // (peat is more ductile than mineral sediment; provisional parity rule
  // pending calibration). Submerged cells of the erodible sets are
  // weakened: the cohesion (the ice/cement bond, A - C and N) by the
  // cohesion factor, the elastic moduli (uniformly, preserving nu) by
  // the stiffness factor, and the eqps limit by its factor.
  bulk_porosity_        = p->get<RealType>("ACE Bulk Porosity", 0.0);
  cohesion_weakening_   = p->get<RealType>("ACE Cohesion Weakening Factor", 1.0);
  stiffness_weakening_  = p->get<RealType>("ACE Stiffness Weakening Factor", 1.0);
  eqps_limit_weakening_ = p->get<RealType>("ACE Eqps Limit Weakening Factor", 1.0);
  if (p->isParameter("ACE Sea Level File")) {
    sea_level_ = vectorFromFile(p->get<std::string>("ACE Sea Level File"));
  }
  if (p->isParameter("ACE Time File")) {
    time_ = vectorFromFile(p->get<std::string>("ACE Time File"));
  }
  ALBANY_ASSERT(time_.size() == sea_level_.size(), "Permafrost: ACE Time File and ACE Sea Level File must have the same length");
  if (p->isParameter("ACE Z Depth File")) {
    z_above_mean_sea_level_ = vectorFromFile(p->get<std::string>("ACE Z Depth File"));
  }
  // Deck-key convention is spaces; the legacy underscore spelling is
  // still accepted (it leaked from the ACE_Porosity FIELD name).
  std::string const porosity_file_key = p->isParameter("ACE Porosity File") ? "ACE Porosity File" : "ACE_Porosity File";
  if (p->isParameter(porosity_file_key)) {
    porosity_from_file_ = vectorFromFile(p->get<std::string>(porosity_file_key));
    ALBANY_ASSERT(
        z_above_mean_sea_level_.size() == porosity_from_file_.size(),
        "Permafrost: ACE Z Depth File and ACE_Porosity File must have the same length");
  }
  if (p->isParameter("ACE Peat File")) {
    peat_from_file_ = vectorFromFile(p->get<std::string>("ACE Peat File"));
    ALBANY_ASSERT(
        z_above_mean_sea_level_.size() == peat_from_file_.size(),
        "Permafrost: ACE Z Depth File and ACE Peat File must have the same length");
  }

  // Integration-point coordinates are needed only to evaluate the depth
  // profiles and the submersion test.
  have_height_ = z_above_mean_sea_level_.size() > 0 || sea_level_.size() > 0;
  if (have_height_) this->setIntegrationPointLocationFlag(true);

  // retrieve appropriate field name strings
  std::string const cauchy_string           = field_name_map_["Cauchy_Stress"];
  std::string const backStress_string       = field_name_map_["Back_Stress"];
  std::string const capParameter_string     = field_name_map_["Cap_Parameter"];
  std::string const eqps_string             = field_name_map_["eqps"];
  std::string const volPlasticStrain_string = field_name_map_["volPlastic_Strain"];
  std::string const strain_string           = field_name_map_["Strain"];
  std::string const F_string                = field_name_map_["F"];
  std::string const J_string                = field_name_map_["J"];
  std::string const Fp_string               = field_name_map_["Fp"];

  std::string const tension_indicator_string      = field_name_map_["Tension_Indicator"];
  std::string const backstress_indicator_string   = field_name_map_["Backstress_Indicator"];
  std::string const crush_indicator_string        = field_name_map_["Crush_Indicator"];
  std::string const eqps_indicator_string         = field_name_map_["Eqps_Indicator"];
  std::string const angle_indicator_string        = field_name_map_["Angle_Indicator"];
  std::string const displacement_indicator_string = field_name_map_["Displacement_Indicator"];
  std::string const strain_indicator_string       = field_name_map_["Strain_Indicator"];
  std::string const tilt_angle_string             = field_name_map_["Tilt_Angle"];

  // define the dependent fields. Elasticity is computed internally from
  // the end-member (K, G) pairs, so there is no dependence on the
  // Elastic Modulus / Poissons Ratio fields.
  if (have_ice_field_) setDependentField("ACE_Ice_Saturation", dl->qp_scalar);
  if (have_temperature_) setDependentField("Temperature", dl->qp_scalar);
  if (maximum_displacement_ > 0.0) setDependentField("Displacement", dl->qp_vector);

  if (finite_deformation_) {
    setDependentField(F_string, dl->qp_tensor);
    setDependentField(J_string, dl->qp_scalar);
    // F_old (needed to recover the elastic log strain at t_n) is already
    // registered as a state with history by MechanicsProblem.

    // plastic deformation gradient
    setEvaluatedField(Fp_string, dl->qp_tensor);
    addStateVariable(Fp_string, dl->qp_tensor, "identity", 0.0, true, true);
  } else {
    setDependentField("Strain", dl->qp_tensor);

    // strain is a state variable (old state needed)
    addStateVariable(strain_string, dl->qp_tensor, "scalar", 0.0, true, true);
  }

  // Ice-saturation state: the saturation seen at the previous converged
  // step, used as the start of the within-step parameter ramp. Its
  // initial value matches the saturation source so that the first step
  // does not ramp spuriously (the ACE field source initializes frozen).
  RealType f_init = 1.0;
  if (!have_ice_field_) {
    f_init = (ice_sat_times_.size() > 0) ? ice_sat_values_[0] : ice_sat_constant_;
  }
  setEvaluatedField("Ice_Saturation_State", dl->qp_scalar);
  addStateVariable("Ice_Saturation_State", dl->qp_scalar, "scalar", f_init, true, false);

  // define the evaluated fields
  setEvaluatedField(cauchy_string, dl->qp_tensor);
  setEvaluatedField(backStress_string, dl->qp_tensor);
  setEvaluatedField(capParameter_string, dl->qp_scalar);
  setEvaluatedField(eqps_string, dl->qp_scalar);
  setEvaluatedField(volPlasticStrain_string, dl->qp_scalar);

  setEvaluatedField(tension_indicator_string, dl->qp_scalar);
  setEvaluatedField(backstress_indicator_string, dl->qp_scalar);
  setEvaluatedField(crush_indicator_string, dl->qp_scalar);
  setEvaluatedField(eqps_indicator_string, dl->qp_scalar);
  setEvaluatedField(angle_indicator_string, dl->qp_scalar);
  setEvaluatedField(displacement_indicator_string, dl->qp_scalar);
  setEvaluatedField(strain_indicator_string, dl->qp_scalar);
  setEvaluatedField(tilt_angle_string, dl->qp_scalar);
  setEvaluatedField("failure_modes", dl->qp_scalar);
  setEvaluatedField("failure_state", dl->cell_scalar2);
  setEvaluatedField("cell_death", dl->cell_scalar2);
  // Stored per quadrature point (qp_scalar) rather than per cell: the
  // qp_scalar old-state buffer is the proven path for a nonzero initial
  // value (cap_scalar2's rank-1 old buffer is not reliably initialized).
  // The value is uniform across a cell's points.
  if (gradual_death_) setEvaluatedField("death_decay", dl->qp_scalar);

  // define the state variables
  addStateVariable(cauchy_string, dl->qp_tensor, "scalar", 0.0, true, true);

  addStateVariable(backStress_string, dl->qp_tensor, "scalar", 0.0, true, true);

  // The cap-parameter state is initialized at the frozen kappa0:
  // simulations are assumed to start from the frozen state.
  addStateVariable(capParameter_string, dl->qp_scalar, "scalar", frozen_.kappa0, true, true);

  addStateVariable(eqps_string, dl->qp_scalar, "scalar", 0.0, true, true);

  addStateVariable(volPlasticStrain_string, dl->qp_scalar, "scalar", 0.0, true, true);

  // Failure indicators (output-only) and the death bookkeeping states.
  // The failure_modes bitmask is the only one that needs its old state.
  addStateVariable(tension_indicator_string, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Tension Indicator", false));
  addStateVariable(backstress_indicator_string, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Backstress Indicator", false));
  addStateVariable(crush_indicator_string, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Crush Indicator", false));
  addStateVariable(eqps_indicator_string, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Eqps Indicator", false));
  addStateVariable(angle_indicator_string, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Angle Indicator", false));
  addStateVariable(displacement_indicator_string, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Displacement Indicator", false));
  addStateVariable(strain_indicator_string, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Strain Indicator", false));
  addStateVariable(tilt_angle_string, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Tilt Angle", false));
  addStateVariable("failure_modes", dl->qp_scalar, "scalar", 0.0, true, p->get<bool>("Output Failure Modes", false));
  addStateVariable("failure_state", dl->cell_scalar2, "scalar", 0.0, false, p->get<bool>("Output Failure State", false));
  addStateVariable("cell_death", dl->cell_scalar2, "scalar", 0.0, false, p->get<bool>("Output Cell Death", false));
  // Decay factor: needs its old state (it persists and advances one
  // increment per accepted step via the new->old rotation). Initialized to
  // 1.0 (intact).
  if (gradual_death_)
    addStateVariable("death_decay", dl->qp_scalar, "scalar", 1.0, true, p->get<bool>("Output Death Decay", false));
}

template <typename EvalT, typename Traits>
void
PermafrostKernel<EvalT, Traits>::init(
    Workset&                workset,
    FieldMap<ScalarT const>& dep_fields,
    FieldMap<ScalarT>&       eval_fields)
{
  std::string cauchy_string           = field_name_map_["Cauchy_Stress"];
  std::string backStress_string       = field_name_map_["Back_Stress"];
  std::string capParameter_string     = field_name_map_["Cap_Parameter"];
  std::string eqps_string             = field_name_map_["eqps"];
  std::string volPlasticStrain_string = field_name_map_["volPlastic_Strain"];
  std::string strain_string           = field_name_map_["Strain"];
  std::string F_string                = field_name_map_["F"];
  std::string J_string                = field_name_map_["J"];
  std::string Fp_string               = field_name_map_["Fp"];

  std::string tension_indicator_string      = field_name_map_["Tension_Indicator"];
  std::string backstress_indicator_string   = field_name_map_["Backstress_Indicator"];
  std::string crush_indicator_string        = field_name_map_["Crush_Indicator"];
  std::string eqps_indicator_string         = field_name_map_["Eqps_Indicator"];
  std::string angle_indicator_string        = field_name_map_["Angle_Indicator"];
  std::string displacement_indicator_string = field_name_map_["Displacement_Indicator"];
  std::string strain_indicator_string       = field_name_map_["Strain_Indicator"];
  std::string tilt_angle_string             = field_name_map_["Tilt_Angle"];

  // extract dependent MDFields
  if (have_ice_field_) ice_saturation_ = *dep_fields["ACE_Ice_Saturation"];
  if (have_temperature_) temperature_ = *dep_fields["Temperature"];
  if (maximum_displacement_ > 0.0) displacement_ = *dep_fields["Displacement"];
  if (finite_deformation_) {
    def_grad_ = *dep_fields[F_string];
    J_        = *dep_fields[J_string];
  } else {
    strain_ = *dep_fields["Strain"];
  }

  // extract evaluated MDFields
  stress_           = *eval_fields[cauchy_string];
  backStress_       = *eval_fields[backStress_string];
  capParameter_     = *eval_fields[capParameter_string];
  eqps_             = *eval_fields[eqps_string];
  volPlasticStrain_ = *eval_fields[volPlasticStrain_string];
  if (finite_deformation_) Fp_ = *eval_fields[Fp_string];
  ice_sat_state_ = *eval_fields["Ice_Saturation_State"];

  tension_indicator_      = *eval_fields[tension_indicator_string];
  backstress_indicator_   = *eval_fields[backstress_indicator_string];
  crush_indicator_        = *eval_fields[crush_indicator_string];
  eqps_indicator_         = *eval_fields[eqps_indicator_string];
  angle_indicator_        = *eval_fields[angle_indicator_string];
  displacement_indicator_ = *eval_fields[displacement_indicator_string];
  strain_indicator_       = *eval_fields[strain_indicator_string];
  tilt_angle_             = *eval_fields[tilt_angle_string];
  failure_modes_          = *eval_fields["failure_modes"];
  failed_                 = *eval_fields["failure_state"];
  dead_                   = *eval_fields["cell_death"];
  if (gradual_death_) {
    death_decay_     = *eval_fields["death_decay"];
    death_decay_old_ = (*workset.stateArrayPtr)["death_decay_old"];
  }

  // get old state variables
  if (finite_deformation_) {
    def_grad_old_ = (*workset.stateArrayPtr)[F_string + "_old"];
    Fp_old_       = (*workset.stateArrayPtr)[Fp_string + "_old"];
  } else {
    strain_old_ = (*workset.stateArrayPtr)[strain_string + "_old"];
  }
  stress_old_           = (*workset.stateArrayPtr)[cauchy_string + "_old"];
  backStress_old_       = (*workset.stateArrayPtr)[backStress_string + "_old"];
  capParameter_old_     = (*workset.stateArrayPtr)[capParameter_string + "_old"];
  eqps_old_             = (*workset.stateArrayPtr)[eqps_string + "_old"];
  volPlasticStrain_old_ = (*workset.stateArrayPtr)[volPlasticStrain_string + "_old"];
  ice_sat_state_old_    = (*workset.stateArrayPtr)["Ice_Saturation_State_old"];
  failure_modes_old_    = (*workset.stateArrayPtr)["failure_modes_old"];

  // Read death status from the workset. The ACE solver populates this at
  // the start of each step from the prior converged cell_death state, and
  // operator() writes into it live when new failures occur during a
  // Newton iteration. The scatter evaluator reads it to skip dead cells
  // in assembly. In a plain mechanics run it is null: failure states are
  // still tracked, but no cell is removed from assembly.
  has_failed_old_ = false;
  if (workset.death_status_vec != Teuchos::null) {
    death_status_vec_ = workset.death_status_vec;
    has_failed_old_   = true;
  }

  cell_is_erodible_ = workset.cell_is_erodible;

  current_time_ = workset.current_time;

  // Resolve the death threshold against num_pts_. A 0 value (the
  // default) means "all integration points must fail"; any other value
  // is clamped into [1, num_pts_].
  int const threshold = num_failed_pts_for_death_ <= 0 ? num_pts_ :
                        (num_failed_pts_for_death_ > static_cast<int>(num_pts_) ? static_cast<int>(num_pts_) :
                                                                                  num_failed_pts_for_death_);

  // Seed failed_ (diagnostic, decimal-encoded per-mode counts) and dead_
  // (binary cell-death flag) from the per-(cell, pt) failure-mode bitmask
  // converged at the previous step. Each set bit at (cell, pt)
  // contributes a decimal magnitude (1, 10, ..., 1000000) to
  // failed_(cell, 0) so the encoded value decodes back to per-mode trip
  // counts across all pts of the cell; operator() adds this fill's newly
  // tripped bits on top. dead_(cell, 0) is the predicate the ACE solver
  // consumes to populate death_status_vec.
  auto const num_cells = workset.numCells;
  for (auto cell = 0; cell < num_cells; ++cell) {
    double seed           = 0.0;
    int    num_failed_pts = 0;
    for (auto pt = 0; pt < num_pts_; ++pt) {
      uint8_t const m = static_cast<uint8_t>(failure_modes_old_(cell, pt));
      if (m & 0x01) seed += 1.0;
      if (m & 0x02) seed += 10.0;
      if (m & 0x04) seed += 100.0;
      if (m & 0x08) seed += 1000.0;
      if (m & 0x10) seed += 10000.0;
      if (m & 0x20) seed += 100000.0;
      if (m & 0x40) seed += 1000000.0;
      if (m != 0u) ++num_failed_pts;
    }
    failed_(cell, 0) = seed;
    // cell_death marks the cell as gone: in gradual mode that is when the
    // decay has reached 0 (fully dead); otherwise when enough points have
    // failed (instant removal). The fully-dead set is what the scatter,
    // calving, and dead-DOF rate zeroing consume.
    if (gradual_death_) {
      dead_(cell, 0) = death_decay_old_(cell, 0) <= 0.0 ? 1.0 : 0.0;
    } else {
      dead_(cell, 0) = num_failed_pts >= threshold ? 1.0 : 0.0;
    }
  }
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
PermafrostKernel<EvalT, Traits>::operator()(int cell, int pt) const
{
  using Tensor  = minitensor::Tensor<ScalarT>;
  using Tensor4 = minitensor::Tensor4<ScalarT>;

  Tensor const I(minitensor::eye<ScalarT>(num_dims_));

  // Element death: if this cell was previously marked as failed, set the
  // stress to zero and carry the internal state forward unchanged. The
  // scatter evaluator skips the cell entirely, so the values written here
  // are for output only.
  if (has_failed_old_ && (*death_status_vec_)[cell] > 0.0) {
    for (int i = 0; i < num_dims_; ++i) {
      for (int j = 0; j < num_dims_; ++j) {
        stress_(cell, pt, i, j)     = 0.0;
        backStress_(cell, pt, i, j) = backStress_old_(cell, pt, i, j);
        if (finite_deformation_) Fp_(cell, pt, i, j) = ScalarT(Fp_old_(cell, pt, i, j));
      }
    }
    capParameter_(cell, pt)     = capParameter_old_(cell, pt);
    eqps_(cell, pt)             = eqps_old_(cell, pt);
    volPlasticStrain_(cell, pt) = volPlasticStrain_old_(cell, pt);
    ice_sat_state_(cell, pt)    = ice_sat_state_old_(cell, pt);

    tension_indicator_(cell, pt)      = 0.0;
    backstress_indicator_(cell, pt)   = 0.0;
    crush_indicator_(cell, pt)        = 0.0;
    eqps_indicator_(cell, pt)         = 0.0;
    angle_indicator_(cell, pt)        = 0.0;
    displacement_indicator_(cell, pt) = 0.0;
    strain_indicator_(cell, pt)       = 0.0;
    tilt_angle_(cell, pt)             = 0.0;
    // Carry the failure-mode bitmask forward unchanged: a dead cell never
    // trips new bits, but the state must still be written every fill.
    failure_modes_(cell, pt) = failure_modes_old_(cell, pt);
    if (gradual_death_) death_decay_(cell, pt) = 0.0;
    return;
  }

  // Ice saturation at this integration point.
  ScalarT f_ice;
  if (have_ice_field_) {
    f_ice = ice_saturation_(cell, pt);
  } else if (ice_sat_times_.size() > 0) {
    f_ice = interpolate_table(ice_sat_times_, ice_sat_values_, ScalarT(current_time_));
  } else {
    f_ice = ice_sat_constant_;
  }
  if (f_ice < 0.0) f_ice = 0.0;
  if (f_ice > 1.0) f_ice = 1.0;


  // Saturation at the previous converged step: start of the parameter
  // ramp the integrator applies across its substeps.
  ScalarT f_old = ice_sat_state_old_(cell, pt);
  ice_sat_state_(cell, pt) = f_ice;

  // ACE environment at this integration point: depth profiles and the
  // submersion test. All evaluated on values (the profiles are data).
  RealType height = 0.0;
  if (have_height_) {
    auto const coords = this->model_.getCoordVecField();
    height            = Sacado::Value<ScalarT>::eval(coords(cell, pt, 2));
  }
  RealType const sea_level = sea_level_.size() > 0 ? interpolateVectors(time_, sea_level_, current_time_) : -999.0;
  RealType const porosity =
      porosity_from_file_.size() > 0 ? interpolateVectors(z_above_mean_sea_level_, porosity_from_file_, height) : bulk_porosity_;
  RealType const peat = peat_from_file_.size() > 0 ? interpolateVectors(z_above_mean_sea_level_, peat_from_file_, height) : 0.0;

  // Ocean exposure: cells of the erodible sets at or below sea level are
  // weakened. The weakening is instantaneous at the submersion flip (the
  // J2Erosion convention); unlike the saturation, it is not ramped
  // across the step, but it scales the moduli uniformly so the stored
  // stress re-expression is unaffected.
  bool const is_erodible = cell_is_erodible_.size() > 0 && cell_is_erodible_[cell] != 0;
  bool const submerged   = is_erodible && have_height_ && height <= sea_level;

  // Effective eqps limit: peat raises it, ocean exposure lowers it.
  RealType max_eqps_eff = maximum_eqps_;
  if (peat > 0.0) max_eqps_eff *= 1.0 + peat;
  if (submerged) max_eqps_eff /= eqps_limit_weakening_;

  // Effective distortion limit, J2Erosion's conventions for its Strain
  // Limit: peat raises the limit to at least 1 + peat, and ocean
  // exposure shrinks the allowed excess over 1 by the same factor that
  // weakens the eqps limit (the port of ACE SL Weakening Factor).
  RealType max_distortion_eff = maximum_distortion_;
  if (max_distortion_eff > 0.0) {
    if (peat > 0.0) max_distortion_eff = std::max(max_distortion_eff, 1.0 + peat);
    if (submerged) max_distortion_eff = 1.0 + (max_distortion_eff - 1.0) / eqps_limit_weakening_;
  }

  // Saturation-to-parameter map (see the file banner). Elasticity from
  // the (K, G) split: the shear modulus carries the order-of-magnitude
  // ice-bonding dependence (log-linear); the bulk modulus is bounded
  // below by the saturated mixture (linear between end members, with the
  // thawed K chosen at the Wood bound during calibration). The effective
  // Poisson ratio is capped at nu_max, preserving G (the trusted
  // physics) and reducing K.
  auto map_params = [&](ScalarT const& f) {
    CapParameters<ScalarT> P;
    ScalarT Kmod = (1.0 - f) * thawed_.K + f * frozen_.K;
    ScalarT const Gmod = std::exp((1.0 - f) * std::log(ScalarT(thawed_.G)) + f * std::log(ScalarT(frozen_.G)));
    ScalarT nu = (3.0 * Kmod - 2.0 * Gmod) / (2.0 * (3.0 * Kmod + Gmod));
    if (nu > nu_max_) {
      nu   = nu_max_;
      Kmod = 2.0 * Gmod * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu));
    }
    P.lame   = Kmod - 2.0 * Gmod / 3.0;
    P.mu     = Gmod;
    P.A      = (1.0 - f) * thawed_.A + f * frozen_.A;
    P.C      = (1.0 - f) * thawed_.C + f * frozen_.C;
    P.N      = (1.0 - f) * thawed_.N + f * frozen_.N;
    P.kappa0 = (1.0 - f) * thawed_.kappa0 + f * frozen_.kappa0;
    P.W      = (1.0 - f) * thawed_.W + f * frozen_.W;
    P.D1     = (1.0 - f) * thawed_.D1 + f * frozen_.D1;
    P.calpha = (1.0 - f) * thawed_.calpha + f * frozen_.calpha;
    P.D      = D_;
    P.theta  = theta_;
    P.L      = L_;
    P.phi    = phi_;
    P.R      = R_;
    P.Q      = Q_;
    P.psi    = psi_;
    P.D2     = D2_;

    // ACE overrides: the porosity profile sets the crushable volume (the
    // pore space is what crushes, independent of what fills it), and
    // ocean exposure knocks down the cohesion -- C rises toward A so the
    // zero-pressure strength (A - C) and the offset N shrink by the
    // factor, leaving the friction slope and asymptote intact -- and the
    // moduli, uniformly (nu, and hence the nu cap, unaffected).
    if (porosity > 0.0) P.W = porosity;
    if (submerged) {
      P.C    = P.A - (P.A - P.C) / cohesion_weakening_;
      P.N    = P.N / cohesion_weakening_;
      P.lame = P.lame / stiffness_weakening_;
      P.mu   = P.mu / stiffness_weakening_;
    }
    return P;
  };

  CapParameters<ScalarT> const P0 = map_params(f_old);
  CapParameters<ScalarT> const P1 = map_params(f_ice);

  ScalarT const mu          = P1.mu;
  ScalarT const lame        = P1.lame;
  ScalarT const bulkModulus = P1.lame + 2.0 * P1.mu / 3.0;
  ScalarT const E           = 9.0 * bulkModulus * P1.mu / (3.0 * bulkModulus + P1.mu);

  // elastic tangent
  Tensor4 const id1 = minitensor::identity_1<ScalarT>(num_dims_);
  Tensor4 const id2 = minitensor::identity_2<ScalarT>(num_dims_);
  Tensor4 const id3 = minitensor::identity_3<ScalarT>(num_dims_);

  Tensor4 const Celastic   = lame * id3 + mu * (id1 + id2);
  Tensor4 const compliance = (1.0 / bulkModulus / 9.0) * id3 +
      (1.0 / mu / 2.0) * (0.5 * (id1 + id2) - (1.0 / 3.0) * id3);

  // Load old back stress and cap parameter
  Tensor alphaVal(num_dims_);
  for (int i = 0; i < num_dims_; ++i)
    for (int j = 0; j < num_dims_; ++j)
      alphaVal(i, j) = backStress_old_(cell, pt, i, j);

  ScalarT kappaVal = capParameter_old_(cell, pt);

  // Spectral functions of symmetric tensors, for the exp/log kinematics.
  auto fun_sym = [&](Tensor const& Asym, ScalarT (*fun)(ScalarT const&)) {
    Tensor V(num_dims_), Dg(num_dims_);
    std::tie(V, Dg) = minitensor::eig_sym(Asym);
    Tensor B(num_dims_);
    B.fill(minitensor::Filler::ZEROS);
    for (int k = 0; k < num_dims_; ++k) {
      ScalarT const fk = fun(Dg(k, k));
      for (int i = 0; i < num_dims_; ++i)
        for (int j = 0; j < num_dims_; ++j)
          B(i, j) += fk * V(i, k) * V(j, k);
    }
    return B;
  };
  auto log_sym  = [&](Tensor const& A) { return fun_sym(A, +[](ScalarT const& x) { return ScalarT(std::log(x)); }); };
  auto exp_sym  = [&](Tensor const& A) { return fun_sym(A, +[](ScalarT const& x) { return ScalarT(std::exp(x)); }); };
  auto sqrt_sym = [&](Tensor const& A) { return fun_sym(A, +[](ScalarT const& x) { return ScalarT(std::sqrt(x)); }); };

  // Kinematics: sigmaN and depsilon feed the (small-strain) integrator.
  // In finite deformation they are the Kirchhoff stress and the increment
  // of logarithmic elastic strain (exponential/logarithmic-map approach).
  Tensor sigmaN(num_dims_);
  Tensor depsilon(num_dims_);
  Tensor Fval(num_dims_);
  ScalarT Jdet = 1.0;

  if (finite_deformation_) {
    Tensor F_n(num_dims_), Fp_n(num_dims_);
    for (int i = 0; i < num_dims_; ++i) {
      for (int j = 0; j < num_dims_; ++j) {
        Fval(i, j) = def_grad_(cell, pt, i, j);
        F_n(i, j)  = def_grad_old_(cell, pt, i, j);
        Fp_n(i, j) = Fp_old_(cell, pt, i, j);
      }
    }
    Jdet = J_(cell, pt);

    // Thermal stretch: the mechanical deformation gradient excludes the
    // thermal expansion (J2Erosion convention).
    if (have_temperature_) {
      ScalarT const dtemp           = temperature_(cell, pt) - ref_temperature_;
      ScalarT const thermal_stretch = std::exp(expansion_coeff_ * dtemp);
      Fval = (1.0 / thermal_stretch) * Fval;
    }

    Tensor const Fpinv  = minitensor::inverse(Fp_n);
    Tensor const Cpinv  = minitensor::dot(Fpinv, minitensor::transpose(Fpinv));

    // Trial elastic logarithmic strain from be_tr = F Cp^-1 F^T
    Tensor const be_tr  = minitensor::dot(Fval, minitensor::dot(Cpinv, minitensor::transpose(Fval)));
    Tensor const eps_tr = 0.5 * log_sym(be_tr);

    // Elastic logarithmic strain at t_n (same Cp^-1, old F); the
    // corresponding Kirchhoff stress tau_n = C : eps_e_n is the stress
    // the integrator starts from, so the trial state is exact for the
    // hyperelastic logarithmic model.
    Tensor const be_n    = minitensor::dot(F_n, minitensor::dot(Cpinv, minitensor::transpose(F_n)));
    Tensor const eps_e_n = 0.5 * log_sym(be_n);

    depsilon = eps_tr - eps_e_n;
    sigmaN   = minitensor::dotdot(Celastic, eps_e_n);
  } else {
    for (int i = 0; i < num_dims_; ++i) {
      for (int j = 0; j < num_dims_; ++j) {
        depsilon(i, j) = strain_(cell, pt, i, j) - strain_old_(cell, pt, i, j);
        sigmaN(i, j)   = stress_old_(cell, pt, i, j);
      }
    }
    // The elastic strain, not the stress, is the state: when the moduli
    // change with the ice saturation, the stored stress is re-expressed
    // through the current stiffness, sigma_n <- C(f) : C(f_old)^-1 :
    // sigma_n. (The finite-deformation path does this by construction,
    // recovering the elastic log strain from the stored geometry.)
    ScalarT const K0   = P0.lame + 2.0 * P0.mu / 3.0;
    ScalarT const tr_s = minitensor::trace(sigmaN);
    Tensor const  eps_e_n =
        (tr_s / (9.0 * K0)) * I + (1.0 / (2.0 * P0.mu)) * (sigmaN - (tr_s / 3.0) * I);
    sigmaN = minitensor::dotdot(Celastic, eps_e_n);
    // Thermal strain increment: not supported in the small-strain path
    // (it would require the old temperature state); temperature enters
    // through the finite-deformation path (the ACE configuration).
  }

  Tensor const sigmaTr = sigmaN + minitensor::dotdot(Celastic, depsilon);

  // Plastic strain increment invariants
  ScalarT deqps(0.0), devolps(0.0);

  // f has units of stress^2; scale the drift tolerance by E^2 so the
  // convergence check is invariant to the unit system.
  ScalarT const f_tolerance = 1.0e-12 * E * E;

  // The integrator ramps the parameters from the previous step's
  // saturation to the current one across its substeps.
  CapIntegrator<ScalarT> integ;
  integ.p0 = P0;
  integ.p1 = P1;
  integ.substep_tolerance = substep_tolerance_;
  integ.max_substeps      = max_substeps_;

  Tensor sigmaVal = integ.integrate(sigmaN, alphaVal, kappaVal, depsilon, f_tolerance);

  // Plastic strain increment from the stress correction (zero in the
  // elastic case, where sigmaVal == sigmaTr).
  {
    Tensor dsigma       = sigmaTr - sigmaVal;
    Tensor deps_plastic = minitensor::dotdot(compliance, dsigma);
    devolps             = minitensor::trace(deps_plastic);
    Tensor dev_plastic  = deps_plastic - (1.0 / 3.0) * devolps * I;
    deqps = std::sqrt(2.0 / 3.0) * minitensor::norm(dev_plastic);
  }

  // ---- Failure indicators and element death ----
  // All criteria branch on Sacado values so the Residual and Jacobian
  // evaluations take identical paths; the indicators are output-only
  // diagnostics. sigmaVal here is the integrator's stress (Kirchhoff in
  // finite deformation), the space the yield surface lives in.
  {
    auto const     sig_v   = Sacado::Value<Tensor>::eval(sigmaVal);
    auto const     alpha_v = Sacado::Value<Tensor>::eval(alphaVal);
    RealType const kappa_v = Sacado::Value<ScalarT>::eval(kappaVal);

    RealType const Av     = Sacado::Value<ScalarT>::eval(P1.A);
    RealType const Cv     = Sacado::Value<ScalarT>::eval(P1.C);
    RealType const Dv     = Sacado::Value<ScalarT>::eval(P1.D);
    RealType const thetav = Sacado::Value<ScalarT>::eval(P1.theta);
    RealType const Nv     = Sacado::Value<ScalarT>::eval(P1.N);
    RealType const Rv     = Sacado::Value<ScalarT>::eval(P1.R);
    RealType const k0v    = Sacado::Value<ScalarT>::eval(P1.kappa0);
    RealType const Wv     = Sacado::Value<ScalarT>::eval(P1.W);
    RealType const D1v    = Sacado::Value<ScalarT>::eval(P1.D1);
    RealType const D2v    = Sacado::Value<ScalarT>::eval(P1.D2);

    // Tension: fraction of the zero-pressure shear capacity Ff(0) - N
    // already lost to mean tension; 1 where the shear surface pinches to
    // the apex Ff(I1) = N and the material has no remaining strength.
    // The backstress is deviatoric, so I1 of the relative stress equals
    // I1 of the stress itself.
    RealType const I1v    = minitensor::trace(sig_v);
    RealType const Ff_I1  = Av - Cv * std::exp(Dv * I1v) - thetav * I1v;
    RealType       tension_ind = 0.0;
    if (Av - Cv - Nv > 0.0) tension_ind = ((Av - Cv) - Ff_I1) / (Av - Cv - Nv);

    // Backstress saturation: sqrt(J2(alpha)) / N. At 1 the kinematic
    // hardening is exhausted (G^alpha = 0) and the shear response is
    // perfectly plastic -- the rational replacement for J2Erosion's
    // yield-onset criterion.
    RealType const J2a            = 0.5 * minitensor::dotdot(alpha_v, alpha_v);
    RealType const backstress_ind = Nv > 0.0 ? std::sqrt(J2a) / Nv : 0.0;

    // Crush exhaustion: |eps_v^p(kappa)| / W on the crush curve, the
    // quantity whose approach to 1 engages the cap lock.
    RealType const Ff_k0 = Av - Cv * std::exp(Dv * k0v) - thetav * k0v;
    RealType const X0    = k0v - Rv * Ff_k0;
    RealType const Ff_k  = Av - Cv * std::exp(Dv * kappa_v) - thetav * kappa_v;
    RealType const Xv    = kappa_v - Rv * Ff_k;
    RealType const dX    = Xv - X0;
    RealType const evp_v = Wv * (std::exp(D1v * dX - D2v * dX * dX) - 1.0);
    RealType const crush_ind = Wv > 0.0 ? std::max(0.0, -evp_v) / Wv : 0.0;

    // Equivalent plastic strain against the effective limit (deck limit
    // adjusted by peat and ocean exposure above).
    RealType const eqps_new = eqps_old_(cell, pt) + Sacado::Value<ScalarT>::eval(deqps);
    RealType const eqps_ind = max_eqps_eff > 0.0 ? eqps_new / max_eqps_eff : 0.0;

    // Structural criteria carried over from J2Erosion: tilt angle of the
    // rotation part of F (finite deformation only; polar_rotation is
    // SVD-based, so compute it only when the criterion is enabled), and
    // displacement norm.
    RealType theta_tilt = 0.0;
    if (critical_angle_ > 0.0 && finite_deformation_) {
      auto const Fv     = Sacado::Value<Tensor>::eval(Fval);
      auto const Q      = minitensor::polar_rotation(Fv);
      RealType   cosine = 0.5 * (minitensor::trace(Q) - 1.0);
      cosine            = cosine > 1.0 ? 1.0 : (cosine < -1.0 ? -1.0 : cosine);
      theta_tilt        = std::acos(cosine);
    }
    RealType const angle_ind = critical_angle_ > 0.0 ? std::abs(theta_tilt) / critical_angle_ : 0.0;

    RealType disp_ind     = 0.0;
    bool     disp_failure = false;
    if (maximum_displacement_ > 0.0) {
      minitensor::Vector<ScalarT> u(num_dims_);
      u.fill(displacement_, cell, pt, 0);
      auto const     u_v       = Sacado::Value<decltype(u)>::eval(u);
      RealType const disp_norm = minitensor::norm(u_v);
      disp_ind     = disp_norm / maximum_displacement_;
      disp_failure = disp_norm > maximum_displacement_;
    }

    // Total distortion: the norm of the isochoric right Cauchy-Green
    // tensor over sqrt(3), J2Erosion's strain measure. It is 1 at rest
    // and includes the elastic stretch, so it catches the collapse of
    // thawed material whose stress -- and hence eqps -- stays small as
    // the stiffness drops with the ice bonding. The isochoric
    // normalization makes the thermal (spherical) stretch irrelevant.
    RealType distortion_ind     = 0.0;
    bool     distortion_failure = false;
    if (max_distortion_eff > 0.0 && finite_deformation_) {
      auto const     Fv         = Sacado::Value<Tensor>::eval(Fval);
      auto const     Cv         = minitensor::transpose(Fv) * Fv;
      RealType const Jv         = minitensor::det(Fv);
      RealType const distortion = minitensor::norm((1.0 / std::cbrt(Jv * Jv)) * Cv) / std::sqrt(3.0);
      distortion_ind            = distortion / max_distortion_eff;
      distortion_failure        = distortion >= max_distortion_eff;
    }

    tension_indicator_(cell, pt)      = tension_ind;
    backstress_indicator_(cell, pt)   = backstress_ind;
    crush_indicator_(cell, pt)        = crush_ind;
    eqps_indicator_(cell, pt)         = eqps_ind;
    angle_indicator_(cell, pt)        = angle_ind;
    displacement_indicator_(cell, pt) = disp_ind;
    strain_indicator_(cell, pt)       = distortion_ind;
    tilt_angle_(cell, pt)             = theta_tilt;

    // Per-(cell, pt) OR-accumulation: add each mode's decimal magnitude
    // to `failed` exactly the first time that mode trips at this
    // integration point. `mask` starts from the bitmask converged at the
    // previous step and is written back to failure_modes_ below, so a
    // criterion that keeps tripping contributes its magnitude once and
    // only once across the run. When disable_erosion_ is true the cell
    // cannot die, so the bitmask and the accumulator are left untouched;
    // the indicators above are still computed for diagnostics.
    auto&&  failed = failed_(cell, 0);
    uint8_t mask   = static_cast<uint8_t>(failure_modes_old_(cell, pt));
    auto    trip   = [&](bool fired, uint8_t bit, double magnitude) {
      if (fired == false) return;
      if (disable_erosion_) return;
      if ((mask & bit) != 0u) return;
      mask = static_cast<uint8_t>(mask | bit);
      failed += magnitude;
    };

    trip(tension_indicator_threshold_ > 0.0 && tension_ind >= tension_indicator_threshold_, 0x01, 1.0);
    trip(backstress_indicator_threshold_ > 0.0 && backstress_ind >= backstress_indicator_threshold_, 0x02, 10.0);
    trip(crush_indicator_threshold_ > 0.0 && crush_ind >= crush_indicator_threshold_, 0x04, 100.0);
    trip(critical_angle_ > 0.0 && std::abs(theta_tilt) >= critical_angle_, 0x08, 1000.0);
    trip(disp_failure, 0x10, 10000.0);
    trip(max_eqps_eff > 0.0 && eqps_ind >= 1.0, 0x20, 100000.0);
    trip(distortion_failure, 0x40, 1000000.0);

    // Persist this fill's updated bitmask (old | newly tripped bits) to
    // the STK-backed state, so it follows the cell across workset
    // rebuilds.
    failure_modes_(cell, pt) = static_cast<double>(mask);

    // Live death propagation: when the last pt of a cell is processed,
    // check whether enough pts now have at least one failure bit set to
    // meet the death threshold. If so, mark the cell dead so the scatter
    // evaluator sees it as dead in this same fill -- without this,
    // Newton keeps assembling nonphysical stress for condemned cells
    // through a death cascade. Points are processed in order, so at the
    // last pt every failure_modes_(cell, p) is current. The threshold
    // must match the one used by init() so the seed and the live update
    // are consistent.
    if (pt == num_pts_ - 1 && (has_failed_old_ || gradual_death_)) {
      int const threshold = num_failed_pts_for_death_ <= 0 ? num_pts_ :
                            (num_failed_pts_for_death_ > static_cast<int>(num_pts_) ? static_cast<int>(num_pts_) :
                                                                                      num_failed_pts_for_death_);
      int num_failed_pts = 0;
      for (auto p = 0; p < num_pts_; ++p) {
        if (static_cast<uint8_t>(Sacado::Value<ScalarT>::eval(failure_modes_(cell, p))) != 0u) {
          ++num_failed_pts;
        }
      }
      bool const marked = num_failed_pts >= threshold;

      if (gradual_death_) {
        // Advance the decay one increment per accepted step. Computing it
        // from the start-of-step value (death_decay_old_) makes the write
        // idempotent across the step's fills; the new->old rotation then
        // advances it exactly once per accepted step. cell_death flips to 1
        // only when the decay reaches 0 -- the fully-dead signal the scatter,
        // calving, and dead-DOF rate zeroing consume.
        double const d_old = death_decay_old_(cell, 0);
        double       d_new = d_old;
        if (marked && d_old > 0.0) {
          d_new = d_old - 1.0 / static_cast<double>(death_steps_);
          if (d_new < 0.0) d_new = 0.0;
        }
        // The decay is per cell; store it at every quadrature point (the
        // value is uniform across the cell).
        for (int p = 0; p < num_pts_; ++p) death_decay_(cell, p) = d_new;
        dead_(cell, 0) = d_new <= 0.0 ? 1.0 : 0.0;
      } else if (has_failed_old_) {
        // Instant removal (original behavior): mark the cell dead in the
        // shared scatter signal so it is skipped for the rest of this fill.
        if (marked && (*death_status_vec_)[cell] == 0.0) {
          (*death_status_vec_)[cell] = 1.0;
          dead_(cell, 0)             = 1.0;
        }
      }
    }
  }

  // Finite deformation: update the plastic deformation gradient from the
  // returned Kirchhoff stress (elastic log strain = compliance : tau),
  // and convert the stored stress to Cauchy.
  if (finite_deformation_) {
    Tensor const eps_e_new  = minitensor::dotdot(compliance, sigmaVal);
    Tensor const be_new     = exp_sym(ScalarT(2.0) * eps_e_new);
    Tensor const Finv       = minitensor::inverse(Fval);
    Tensor const Cpinv_new  = minitensor::dot(Finv, minitensor::dot(be_new, minitensor::transpose(Finv)));
    // Fp is taken as the unique symmetric positive-definite root; the
    // rotational part of Fp is irrelevant for an isotropic model.
    Tensor const Fp_new = sqrt_sym(minitensor::inverse(Cpinv_new));

    for (int i = 0; i < num_dims_; ++i)
      for (int j = 0; j < num_dims_; ++j)
        Fp_(cell, pt, i, j) = Fp_new(i, j);

    sigmaVal = (1.0 / Jdet) * sigmaVal;
  }

  // Store results. In gradual mode the stored stress -- the cell's residual
  // force and, since sigmaVal carries the AD seeds, its consistent tangent
  // -- is scaled by the start-of-step decay factor (constant over the step),
  // so a failing cell sheds its load smoothly over Death Steps rather than
  // in a single iterate. The internal state (Fp, backstress, kappa) is
  // computed and stored UNSCALED, so the cell keeps a valid material state
  // right up until it is fully dead.
  ScalarT const decay = gradual_death_ ? ScalarT(death_decay_old_(cell, pt)) : ScalarT(1.0);
  for (int i = 0; i < num_dims_; ++i) {
    for (int j = 0; j < num_dims_; ++j) {
      stress_(cell, pt, i, j)     = decay * sigmaVal(i, j);
      backStress_(cell, pt, i, j) = alphaVal(i, j);
    }
  }

  capParameter_(cell, pt)     = kappaVal;
  eqps_(cell, pt)             = eqps_old_(cell, pt) + deqps;
  volPlasticStrain_(cell, pt) = volPlasticStrain_old_(cell, pt) + devolps;
}

}  // namespace LCM
