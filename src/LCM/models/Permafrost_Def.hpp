// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
//
// Permafrost constitutive model: the three-invariant cap plasticity
// model specialized for frozen/thawing sediment in the ACE Arctic
// coastal erosion application, intended to replace J2Erosion. The
// integration and constitutive functions live in the shared
// CapIntegrator (verified against an independent reference); this
// kernel supplies parameters that vary per integration point with the
// ice saturation f. The design is recorded in
// doc/developersGuide/cap_plasticity.tex, Section "Planned Extension:
// Permafrost and Erosion".
//
// Saturation-to-parameter map (Phase 1):
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
// Remaining phases per the documented plan: (2) failure indicators and
// element-death plumbing; (3) ACE workflow integration.

#include <MiniTensor.h>

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

  // define the dependent fields. Elasticity is computed internally from
  // the end-member (K, G) pairs, so there is no dependence on the
  // Elastic Modulus / Poissons Ratio fields.
  if (have_ice_field_) setDependentField("ACE_Ice_Saturation", dl->qp_scalar);
  if (have_temperature_) setDependentField("Temperature", dl->qp_scalar);

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

  // define the state variables
  addStateVariable(cauchy_string, dl->qp_tensor, "scalar", 0.0, true, true);

  addStateVariable(backStress_string, dl->qp_tensor, "scalar", 0.0, true, true);

  // The cap-parameter state is initialized at the frozen kappa0:
  // simulations are assumed to start from the frozen state.
  addStateVariable(capParameter_string, dl->qp_scalar, "scalar", frozen_.kappa0, true, true);

  addStateVariable(eqps_string, dl->qp_scalar, "scalar", 0.0, true, true);

  addStateVariable(volPlasticStrain_string, dl->qp_scalar, "scalar", 0.0, true, true);
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

  // extract dependent MDFields
  if (have_ice_field_) ice_saturation_ = *dep_fields["ACE_Ice_Saturation"];
  if (have_temperature_) temperature_ = *dep_fields["Temperature"];
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

  current_time_ = workset.current_time;
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
PermafrostKernel<EvalT, Traits>::operator()(int cell, int pt) const
{
  using Tensor  = minitensor::Tensor<ScalarT>;
  using Tensor4 = minitensor::Tensor4<ScalarT>;

  Tensor const I(minitensor::eye<ScalarT>(num_dims_));

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
    // Thermal strain increment: the small-strain path requires the old
    // temperature state and is deferred; Phase 1 supports temperature in
    // the finite-deformation path (the ACE configuration).
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

  // Store results
  for (int i = 0; i < num_dims_; ++i) {
    for (int j = 0; j < num_dims_; ++j) {
      stress_(cell, pt, i, j)     = sigmaVal(i, j);
      backStress_(cell, pt, i, j) = alphaVal(i, j);
    }
  }

  capParameter_(cell, pt)     = kappaVal;
  eqps_(cell, pt)             = eqps_old_(cell, pt) + deqps;
  volPlasticStrain_(cell, pt) = volPlasticStrain_old_(cell, pt) + devolps;
}

}  // namespace LCM
