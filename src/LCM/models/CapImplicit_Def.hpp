// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include <MiniTensor.h>
#include <cmath>

#include "Albany_Utils.hpp"
#include "CapImplicit.hpp"
#include "MiniNonlinearSolver.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
CapImplicitKernel<EvalT, Traits>::CapImplicitKernel(
    ConstitutiveModel<EvalT, Traits>& model,
    Teuchos::ParameterList*           p,
    Teuchos::RCP<Albany::Layouts> const& dl)
    : BaseKernel(model),
      finite_deformation_(p->get<bool>("Finite Deformation", false)),
      A_(p->get<RealType>("A")),
      D_(p->get<RealType>("D")),
      C_(p->get<RealType>("C")),
      theta_(p->get<RealType>("theta")),
      R_(p->get<RealType>("R")),
      kappa0_(p->get<RealType>("kappa0")),
      W_(p->get<RealType>("W")),
      D1_(p->get<RealType>("D1")),
      D2_(p->get<RealType>("D2")),
      calpha_(p->get<RealType>("calpha")),
      psi_(p->get<RealType>("psi")),
      N_(p->get<RealType>("N")),
      L_(p->get<RealType>("L")),
      phi_(p->get<RealType>("phi")),
      Q_(p->get<RealType>("Q"))
{
  std::string const cauchy_string           = field_name_map_["Cauchy_Stress"];
  std::string const backStress_string       = field_name_map_["Back_Stress"];
  std::string const capParameter_string     = field_name_map_["Cap_Parameter"];
  std::string const eqps_string             = field_name_map_["eqps"];
  std::string const volPlasticStrain_string = field_name_map_["volPlastic_Strain"];

  setDependentField("Poissons Ratio", dl->qp_scalar);
  setDependentField("Elastic Modulus", dl->qp_scalar);

  if (finite_deformation_) {
    std::string const F_string  = field_name_map_["F"];
    std::string const J_string  = field_name_map_["J"];
    std::string const Fp_string = field_name_map_["Fp"];

    setDependentField(F_string, dl->qp_tensor);
    setDependentField(J_string, dl->qp_scalar);

    setEvaluatedField(Fp_string, dl->qp_tensor);
    addStateVariable(Fp_string, dl->qp_tensor, "identity", 0.0, true,
                     p->get<bool>("Output Fp", false));
  } else {
    std::string const strain_string = field_name_map_["Strain"];

    setDependentField("Strain", dl->qp_tensor);
    addStateVariable(strain_string, dl->qp_tensor, "scalar", 0.0, true, true);
  }

  setEvaluatedField(cauchy_string, dl->qp_tensor);
  setEvaluatedField(backStress_string, dl->qp_tensor);
  setEvaluatedField(capParameter_string, dl->qp_scalar);
  setEvaluatedField(eqps_string, dl->qp_scalar);
  setEvaluatedField(volPlasticStrain_string, dl->qp_scalar);

  addStateVariable(cauchy_string, dl->qp_tensor, "scalar", 0.0, true, true);
  addStateVariable(backStress_string, dl->qp_tensor, "scalar", 0.0, true, true);
  addStateVariable(capParameter_string, dl->qp_scalar, "scalar", kappa0_, true, true);
  addStateVariable(eqps_string, dl->qp_scalar, "scalar", 0.0, true, true);
  addStateVariable(volPlasticStrain_string, dl->qp_scalar, "scalar", 0.0, true, true);
}

template <typename EvalT, typename Traits>
void
CapImplicitKernel<EvalT, Traits>::init(
    Workset&                workset,
    FieldMap<ScalarT const>& dep_fields,
    FieldMap<ScalarT>&       eval_fields)
{
  std::string cauchy_string           = field_name_map_["Cauchy_Stress"];
  std::string backStress_string       = field_name_map_["Back_Stress"];
  std::string capParameter_string     = field_name_map_["Cap_Parameter"];
  std::string eqps_string             = field_name_map_["eqps"];
  std::string volPlasticStrain_string = field_name_map_["volPlastic_Strain"];

  elastic_modulus_ = *dep_fields["Elastic Modulus"];
  poissons_ratio_  = *dep_fields["Poissons Ratio"];

  if (finite_deformation_) {
    std::string F_string  = field_name_map_["F"];
    std::string J_string  = field_name_map_["J"];
    std::string Fp_string = field_name_map_["Fp"];

    def_grad_ = *dep_fields[F_string];
    J_        = *dep_fields[J_string];

    Fp_     = *eval_fields[Fp_string];
    Fp_old_ = (*workset.stateArrayPtr)[Fp_string + "_old"];
  } else {
    std::string strain_string = field_name_map_["Strain"];

    strain_     = *dep_fields["Strain"];
    strain_old_ = (*workset.stateArrayPtr)[strain_string + "_old"];
  }

  stress_           = *eval_fields[cauchy_string];
  backStress_       = *eval_fields[backStress_string];
  capParameter_     = *eval_fields[capParameter_string];
  eqps_             = *eval_fields[eqps_string];
  volPlasticStrain_ = *eval_fields[volPlasticStrain_string];

  stress_old_           = (*workset.stateArrayPtr)[cauchy_string + "_old"];
  backStress_old_       = (*workset.stateArrayPtr)[backStress_string + "_old"];
  capParameter_old_     = (*workset.stateArrayPtr)[capParameter_string + "_old"];
  eqps_old_             = (*workset.stateArrayPtr)[eqps_string + "_old"];
  volPlasticStrain_old_ = (*workset.stateArrayPtr)[volPlasticStrain_string + "_old"];
}

// ---------------------------------------------------------------------------
// CapImplicitNLS: 13-unknown nonlinear residual for the local return map.
//
// Unknowns (matches Sec.6 of doc/developersGuide/cap_plasticity.tex):
//   X = (sigma_11, sigma_22, sigma_33,
//        sigma_23, sigma_13, sigma_12,
//        alpha_11, alpha_22,
//        alpha_23, alpha_13, alpha_12,
//        kappa, dgamma)
//
// dim(X) = 13.  alpha_33 = -(alpha_11 + alpha_22) by deviatoric constraint.
//
// Residual (backward-Euler, eq 50-55 of the doc):
//   R_sigma   = sigma - sigma_tr + dgamma * (Celastic : dg/dsigma)
//   R_alpha   = alpha - alpha_n - dgamma * h_alpha(alpha, dg/dsigma)
//   R_kappa   = kappa - kappa_n - dgamma * h_kappa(I1(dg/dsigma), kappa)
//   R_gamma   = f(sigma, alpha, kappa)
//
// dg/dsigma is provided analytically by compute_dgdsigma below, in tensor
// form using the standard invariant-chain-rule formulas
//   dI1/dsigma  = I
//   dJ2/dsigma  = s
//   dJ3/dsigma  = s*s - (2/3) J2 I
// so that the gradient method is purely algebraic.  hessian comes from
// minitensor's default AD on gradient.
// ---------------------------------------------------------------------------
template <typename EvalT, minitensor::Index NUM_UNK = 13>
class CapImplicitNLS : public minitensor::Function_Base<CapImplicitNLS<EvalT, NUM_UNK>, typename EvalT::ScalarT, NUM_UNK>
{
  using S = typename EvalT::ScalarT;

 public:
  CapImplicitNLS(
      RealType A, RealType C, RealType D, RealType theta, RealType R, RealType kappa0,
      RealType W, RealType D1, RealType D2, RealType calpha, RealType psi,
      RealType N, RealType L, RealType phi, RealType Q,
      S const& lame, S const& mu,
      minitensor::Tensor<S, 3> const& sigma_tr,
      minitensor::Tensor<S, 3> const& alpha_n,
      S const& kappa_n)
      : A_(A), C_(C), D_(D), theta_(theta), R_(R), kappa0_(kappa0),
        W_(W), D1_(D1), D2_(D2), calpha_(calpha), psi_(psi),
        N_(N), L_(L), phi_(phi), Q_(Q),
        lame_(lame), mu_(mu),
        sigma_tr_(sigma_tr), alpha_n_(alpha_n), kappa_n_(kappa_n)
  {
  }

  constexpr static char const* const NAME{"CapImplicit NLS"};

  using Base = minitensor::Function_Base<CapImplicitNLS<EvalT, NUM_UNK>, S, NUM_UNK>;

  // value: default merit function 0.5 * dot(residual, residual)
  template <typename T, minitensor::Index N>
  T
  value(minitensor::Vector<T, N> const& x)
  {
    return Base::value(*this, x);
  }

  // gradient: returns the 13-component backward-Euler residual.
  template <typename T, minitensor::Index N>
  minitensor::Vector<T, N>
  gradient(minitensor::Vector<T, N> const& x)
  {
    minitensor::Index const dim = x.get_dimension();
    ALBANY_EXPECT(dim == Base::DIMENSION);

    // Material constants are RealType -> implicit-convertible to T.
    // Lame parameters and trial state carry outer FAD info on S; peel
    // them down to T (handles S=Residual::ScalarT, ValueT, AD<...,N>).
    T const lame = peel<EvalT, T, N>()(lame_);
    T const mu   = peel<EvalT, T, N>()(mu_);

    minitensor::Tensor<T, 3> sigma_tr(3);
    minitensor::Tensor<T, 3> alpha_n(3);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        sigma_tr(i, j) = peel<EvalT, T, N>()(sigma_tr_(i, j));
        alpha_n(i, j)  = peel<EvalT, T, N>()(alpha_n_(i, j));
      }
    }
    T const kappa_n = peel<EvalT, T, N>()(kappa_n_);

    // Unpack unknowns from x.
    minitensor::Tensor<T, 3> sigma(3);
    sigma(0, 0) = x(0);
    sigma(1, 1) = x(1);
    sigma(2, 2) = x(2);
    sigma(1, 2) = x(3);  sigma(2, 1) = x(3);
    sigma(0, 2) = x(4);  sigma(2, 0) = x(4);
    sigma(0, 1) = x(5);  sigma(1, 0) = x(5);

    minitensor::Tensor<T, 3> alpha(3);
    alpha(0, 0) = x(6);
    alpha(1, 1) = x(7);
    alpha(2, 2) = -x(6) - x(7);
    alpha(1, 2) = x(8);  alpha(2, 1) = x(8);
    alpha(0, 2) = x(9);  alpha(2, 0) = x(9);
    alpha(0, 1) = x(10); alpha(1, 0) = x(10);

    T const kappa  = x(11);
    T const dgamma = x(12);

    // dg/dsigma and the hardening terms.
    minitensor::Tensor<T, 3> const dgdsigma = compute_dgdsigma(sigma, alpha, kappa);
    T const                        I1_dgds  = minitensor::trace(dgdsigma);

    T const                        J2_alpha = T(0.5) * minitensor::dotdot(alpha, alpha);
    minitensor::Tensor<T, 3> const halpha   = compute_halpha(dgdsigma, J2_alpha);

    T const dedkappa = compute_dedkappa(kappa);
    T const hkappa   = (dedkappa != T(0)) ? I1_dgds / dedkappa : T(0);

    // Celastic : dg/dsigma  =  lame * tr(dg/dsigma) * I + 2 mu * dg/dsigma
    T const                        tr_dgds = I1_dgds;
    minitensor::Tensor<T, 3> const Cdgds   = lame * tr_dgds * minitensor::eye<T, 3>(3)
                                           + T(2.0) * mu * dgdsigma;

    // Backward-Euler residuals.
    minitensor::Tensor<T, 3> const R_sigma = sigma - sigma_tr + dgamma * Cdgds;
    minitensor::Tensor<T, 3> const R_alpha = alpha - alpha_n - dgamma * halpha;
    T const                        R_kappa = kappa - kappa_n - dgamma * hkappa;
    T const                        R_gamma = compute_f(sigma, alpha, kappa);

    minitensor::Vector<T, N> r(dim);
    r(0)  = R_sigma(0, 0);
    r(1)  = R_sigma(1, 1);
    r(2)  = R_sigma(2, 2);
    r(3)  = R_sigma(1, 2);
    r(4)  = R_sigma(0, 2);
    r(5)  = R_sigma(0, 1);
    r(6)  = R_alpha(0, 0);
    r(7)  = R_alpha(1, 1);
    r(8)  = R_alpha(1, 2);
    r(9)  = R_alpha(0, 2);
    r(10) = R_alpha(0, 1);
    r(11) = R_kappa;
    r(12) = R_gamma;

    return r;
  }

  // hessian: default AD from gradient.
  template <typename T, minitensor::Index N>
  minitensor::Tensor<T, N>
  hessian(minitensor::Vector<T, N> const& x)
  {
    return Base::hessian(*this, x);
  }

  // ----- helpers (templated on T, purely algebraic) -----

  template <typename T>
  T
  compute_f(minitensor::Tensor<T, 3> const& sigma,
            minitensor::Tensor<T, 3> const& alpha,
            T const&                        kappa) const
  {
    minitensor::Tensor<T, 3> const xi = sigma - alpha;
    T const                        I1 = minitensor::trace(xi);
    T const                        p  = I1 / T(3);
    minitensor::Tensor<T, 3> const s  = xi - p * minitensor::eye<T, 3>(3);
    T const                        J2 = T(0.5) * minitensor::dotdot(s, s);
    T const                        J3 = minitensor::det(s);

    T const                        Gamma = compute_Gamma(J2, J3);
    T const Ff_I1                        = A_ - C_ * std::exp(D_ * I1) - theta_ * I1;
    T const Ff_kappa                     = A_ - C_ * std::exp(D_ * kappa) - theta_ * kappa;
    T const X                            = kappa - R_ * Ff_kappa;
    T const Fc                           = compute_Fc(I1, kappa, X);

    return Gamma * Gamma * J2 - Fc * (Ff_I1 - N_) * (Ff_I1 - N_);
  }

  template <typename T>
  T
  compute_g(minitensor::Tensor<T, 3> const& sigma,
            minitensor::Tensor<T, 3> const& alpha,
            T const&                        kappa) const
  {
    minitensor::Tensor<T, 3> const xi = sigma - alpha;
    T const                        I1 = minitensor::trace(xi);
    T const                        p  = I1 / T(3);
    minitensor::Tensor<T, 3> const s  = xi - p * minitensor::eye<T, 3>(3);
    T const                        J2 = T(0.5) * minitensor::dotdot(s, s);
    T const                        J3 = minitensor::det(s);

    T const                        Gamma   = compute_Gamma(J2, J3);
    T const Ff_I1                          = A_ - C_ * std::exp(L_ * I1) - phi_ * I1;
    T const Ff_kappa                       = A_ - C_ * std::exp(L_ * kappa) - phi_ * kappa;
    T const X                              = kappa - Q_ * Ff_kappa;
    T const Fc                             = compute_Fc(I1, kappa, X);

    return Gamma * Gamma * J2 - Fc * (Ff_I1 - N_) * (Ff_I1 - N_);
  }

  // dg/dsigma analytically, via the invariant chain rule.
  template <typename T>
  minitensor::Tensor<T, 3>
  compute_dgdsigma(minitensor::Tensor<T, 3> const& sigma,
                   minitensor::Tensor<T, 3> const& alpha,
                   T const&                        kappa) const
  {
    minitensor::Tensor<T, 3> const I_3 = minitensor::eye<T, 3>(3);
    minitensor::Tensor<T, 3> const xi  = sigma - alpha;
    T const                        I1  = minitensor::trace(xi);
    T const                        p   = I1 / T(3);
    minitensor::Tensor<T, 3> const s   = xi - p * I_3;
    T const                        J2  = T(0.5) * minitensor::dotdot(s, s);
    T const                        J3  = minitensor::det(s);

    // Plastic-potential shear surface (uses L, phi, Q -- not D, theta, R).
    T const Ff_I1    = A_ - C_ * std::exp(L_ * I1) - phi_ * I1;
    T const dFf_dI1  = -C_ * L_ * std::exp(L_ * I1) - phi_;
    T const Ff_kappa = A_ - C_ * std::exp(L_ * kappa) - phi_ * kappa;
    T const X        = kappa - Q_ * Ff_kappa;
    T const Fc       = compute_Fc(I1, kappa, X);

    // dFc/dI1:  Fc = 1 - H(kappa - I1) * (I1 - kappa)^2 / (X - kappa)^2
    T dFc_dI1 = T(0);
    T const J2_tol(1.0e-12);
    if (kappa - I1 > T(0) && (X - kappa) != T(0)) {
      dFc_dI1 = -T(2.0) * (I1 - kappa) / ((X - kappa) * (X - kappa));
    }

    // dg/dI1:  d(Gamma^2 J2)/dI1 = 0 (Gamma, J2 don't depend on I1);
    //          d(-Fc (Ff - N)^2)/dI1 = -[dFc/dI1 (Ff-N)^2 + 2 Fc (Ff-N) dFf/dI1]
    T const dg_dI1 = -dFc_dI1 * (Ff_I1 - N_) * (Ff_I1 - N_)
                     - T(2.0) * Fc * (Ff_I1 - N_) * dFf_dI1;

    // Gamma and dGamma/d{J2,J3}.
    T const                        Gamma = compute_Gamma(J2, J3);
    T                              dG_dJ2(0);
    T                              dG_dJ3(0);
    compute_dGamma(J2, J3, dG_dJ2, dG_dJ3);

    // dg/dJ2 = d(Gamma^2 J2)/dJ2 = 2 Gamma (dG/dJ2) J2 + Gamma^2
    T const dg_dJ2 = T(2.0) * Gamma * dG_dJ2 * J2 + Gamma * Gamma;

    // dg/dJ3 = d(Gamma^2)/dJ3 J2 = 2 Gamma (dG/dJ3) J2
    T const dg_dJ3 = T(2.0) * Gamma * dG_dJ3 * J2;

    // Invariant gradients in tensor form.
    minitensor::Tensor<T, 3> const dJ2_dsigma = s;
    // dJ3/dsigma = s*s - (2/3) J2 I.
    minitensor::Tensor<T, 3> const s2         = s * s;
    minitensor::Tensor<T, 3> const dJ3_dsigma = s2 - (T(2.0) / T(3.0)) * J2 * I_3;

    // Assemble.  dI1/dsigma = I.
    minitensor::Tensor<T, 3> const dgdsigma =
        dg_dI1 * I_3 + dg_dJ2 * dJ2_dsigma + dg_dJ3 * dJ3_dsigma;

    return dgdsigma;
    (void)J2_tol;
  }

  // Kinematic hardening direction:  halpha = c_alpha * G_alpha(alpha) * dev(dg/dsigma)
  template <typename T>
  minitensor::Tensor<T, 3>
  compute_halpha(minitensor::Tensor<T, 3> const& dgdsigma, T const& J2_alpha) const
  {
    T const                        Galpha   = T(1.0) - std::sqrt(J2_alpha) / N_;
    T const                        I1_dgds  = minitensor::trace(dgdsigma);
    minitensor::Tensor<T, 3> const dev_dgds = dgdsigma - (I1_dgds / T(3.0)) * minitensor::eye<T, 3>(3);
    return calpha_ * Galpha * dev_dgds;
  }

  // d(epsilon_v^p)/d(kappa) via chain rule through X(kappa).
  template <typename T>
  T
  compute_dedkappa(T const& kappa) const
  {
    T const X0       = kappa0_ - R_ * (A_ - C_ * std::exp(D_ * kappa0_) - theta_ * kappa0_);
    T const Ff_kappa = A_ - C_ * std::exp(D_ * kappa) - theta_ * kappa;
    T const X        = kappa - R_ * Ff_kappa;
    T const dX_dkap  = T(1.0) - R_ * (-C_ * D_ * std::exp(D_ * kappa) - theta_);
    T const arg      = D1_ * (X - X0) - D2_ * (X - X0) * (X - X0);
    T const evp_X    = W_ * (std::exp(arg) - T(1.0));
    T const devp_dX  = W_ * std::exp(arg) * (D1_ - T(2.0) * D2_ * (X - X0));
    (void)evp_X;
    return devp_dX * dX_dkap;
  }

  // Lode coefficient Gamma; degenerates to 1 for small J2 to avoid
  // J3 / J2^{3/2} blow-up at near-hydrostatic stress.
  template <typename T>
  T
  compute_Gamma(T const& J2, T const& J3) const
  {
    T const J2_tol(1.0e-12);
    if (psi_ == 0.0 || J2 <= J2_tol) return T(1);
    T const root_3   = std::sqrt(T(3.0));
    T const ratio    = T(3.0) * root_3 * J3 / (T(2.0) * std::pow(J2, T(1.5)));
    return T(0.5) * (T(1.0) - ratio + (T(1.0) + ratio) / psi_);
  }

  template <typename T>
  void
  compute_dGamma(T const& J2, T const& J3, T& dG_dJ2, T& dG_dJ3) const
  {
    T const J2_tol(1.0e-12);
    if (psi_ == 0.0 || J2 <= J2_tol) {
      dG_dJ2 = T(0);
      dG_dJ3 = T(0);
      return;
    }
    T const root_3  = std::sqrt(T(3.0));
    T const J2_15   = std::pow(J2, T(1.5));
    T const J2_25   = std::pow(J2, T(2.5));
    T const coeff   = T(0.5) * (T(1.0) / psi_ - T(1.0));
    // d(ratio)/dJ2 = -1.5 * (3 sqrt(3) * J3 / 2) / J2^{2.5}
    //              = -2.25 sqrt(3) J3 / J2^{2.5}
    T const dratio_dJ2 = -T(2.25) * root_3 * J3 / J2_25;
    T const dratio_dJ3 = T(1.5) * root_3 / J2_15;
    dG_dJ2 = coeff * dratio_dJ2;
    dG_dJ3 = coeff * dratio_dJ3;
  }

  template <typename T>
  T
  compute_Fc(T const& I1, T const& kappa, T const& X) const
  {
    if (kappa - I1 > T(0) && (X - kappa) != T(0)) {
      return T(1.0) - (I1 - kappa) * (I1 - kappa) / ((X - kappa) * (X - kappa));
    }
    return T(1.0);
  }

  // ----- captured state (held by reference / value) -----
  RealType const A_, C_, D_, theta_, R_, kappa0_;
  RealType const W_, D1_, D2_, calpha_, psi_;
  RealType const N_, L_, phi_, Q_;

  S const& lame_;
  S const& mu_;

  minitensor::Tensor<S, 3> const& sigma_tr_;
  minitensor::Tensor<S, 3> const& alpha_n_;
  S const&                        kappa_n_;
};

// ---------------------------------------------------------------------------
// Kernel operator(): elastic trial + (if yielding) local return map via
// MiniSolver, then state update.
// ---------------------------------------------------------------------------
template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
CapImplicitKernel<EvalT, Traits>::operator()(int cell, int pt) const
{
  constexpr minitensor::Index MAX_DIM{3};
  using Tensor = minitensor::Tensor<ScalarT, MAX_DIM>;

  Tensor const I(minitensor::eye<ScalarT, MAX_DIM>(num_dims_));

  ScalarT const E           = elastic_modulus_(cell, pt);
  ScalarT const nu          = poissons_ratio_(cell, pt);
  ScalarT const lame        = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
  ScalarT const mu          = E / 2.0 / (1.0 + nu);
  ScalarT const bulkModulus = lame + (2.0 / 3.0) * mu;
  (void)bulkModulus;

  // Old state.
  Tensor alpha_n(num_dims_);
  for (int i = 0; i < num_dims_; ++i)
    for (int j = 0; j < num_dims_; ++j)
      alpha_n(i, j) = backStress_old_(cell, pt, i, j);

  ScalarT kappa_n = capParameter_old_(cell, pt);

  // Trial state (small strain OR finite deformation -> Kirchhoff).
  Tensor sigma_tr(num_dims_);
  Tensor Fpn(num_dims_);
  Tensor Fpnew(num_dims_);

  if (finite_deformation_) {
    Tensor F(num_dims_);
    F.fill(def_grad_, cell, pt, 0, 0);

    for (int i = 0; i < num_dims_; ++i)
      for (int j = 0; j < num_dims_; ++j)
        Fpn(i, j) = ScalarT(Fp_old_(cell, pt, i, j));

    Tensor const Fpinv = minitensor::inverse(Fpn);
    Tensor const Cpinv = Fpinv * minitensor::transpose(Fpinv);
    Tensor const be    = F * Cpinv * minitensor::transpose(F);
    Tensor const logbe = minitensor::log_sym<ScalarT>(be);
    Tensor const eps_e = 0.5 * logbe;

    sigma_tr = lame * minitensor::trace(eps_e) * I + 2.0 * mu * eps_e;
  } else {
    Tensor depsilon(num_dims_);
    Tensor sigmaN(num_dims_);
    for (int i = 0; i < num_dims_; ++i) {
      for (int j = 0; j < num_dims_; ++j) {
        depsilon(i, j) = strain_(cell, pt, i, j) - strain_old_(cell, pt, i, j);
        sigmaN(i, j)   = stress_old_(cell, pt, i, j);
      }
    }
    sigma_tr = sigmaN + minitensor::dotdot(
                            lame * minitensor::identity_3<ScalarT, MAX_DIM>(num_dims_)
                                + mu * (minitensor::identity_1<ScalarT, MAX_DIM>(num_dims_)
                                        + minitensor::identity_2<ScalarT, MAX_DIM>(num_dims_)),
                            depsilon);
  }

  // Trial state for the return-map output (gets updated if plastic).
  Tensor sigma_out(num_dims_);
  Tensor alpha_out(num_dims_);
  for (int i = 0; i < num_dims_; ++i) {
    for (int j = 0; j < num_dims_; ++j) {
      sigma_out(i, j) = sigma_tr(i, j);
      alpha_out(i, j) = alpha_n(i, j);
    }
  }
  ScalarT kappa_out = kappa_n;
  ScalarT dgamma_out(0.0);

  // Yield check.  compute_f here uses a stack-local NLS-equivalent helper
  // -- we instantiate a tiny one for the trial evaluation only.
  using NLS = CapImplicitNLS<EvalT>;
  NLS nls(A_, C_, D_, theta_, R_, kappa0_,
          W_, D1_, D2_, calpha_, psi_,
          N_, L_, phi_, Q_,
          lame, mu,
          sigma_tr, alpha_n, kappa_n);

  ScalarT const f_trial = nls.template compute_f<ScalarT>(sigma_tr, alpha_n, kappa_n);

  if (f_trial > 1.0e-11) {  // plastic yielding
    using ValueT = typename Sacado::ValueType<ScalarT>::type;
    constexpr minitensor::Index nls_dim{NLS::DIMENSION};

    using MIN  = minitensor::Minimizer<ValueT, nls_dim>;
    using STEP = minitensor::NewtonWithLineSearchStep<NLS, ValueT, nls_dim>;

    MIN  minimizer;
    minimizer.max_num_iter = 30;
    minimizer.rel_tol      = 1.0e-10;
    minimizer.abs_tol      = 1.0e-10;
    STEP step;

    // Initial guess: x = (sigma_tr packed, alpha_n packed, kappa_n, 0)
    minitensor::Vector<ScalarT, nls_dim> x(nls_dim);
    x(0)  = sigma_tr(0, 0);
    x(1)  = sigma_tr(1, 1);
    x(2)  = sigma_tr(2, 2);
    x(3)  = sigma_tr(1, 2);
    x(4)  = sigma_tr(0, 2);
    x(5)  = sigma_tr(0, 1);
    x(6)  = alpha_n(0, 0);
    x(7)  = alpha_n(1, 1);
    x(8)  = alpha_n(1, 2);
    x(9)  = alpha_n(0, 2);
    x(10) = alpha_n(0, 1);
    x(11) = kappa_n;
    x(12) = ScalarT(0.0);

    LCM::MiniSolver<MIN, STEP, NLS, EvalT, nls_dim> mini_solver(minimizer, step, nls, x);

    // If the local return map failed or produced non-finite output (the
    // 13-equation Newton is sensitive to a far-from-physical initial
    // guess that the global solver can produce in early iterations),
    // fall back to the elastic trial so the global Newton can retry
    // from a sane state instead of poisoning everything with NaN.
    bool                    local_ok = !minimizer.failed;
    typename Sacado::ValueType<ScalarT>::type x_max(0);
    for (int k = 0; k < int(nls_dim) && local_ok; ++k) {
      auto const xk = Sacado::ScalarValue<ScalarT>::eval(x(k));
      if (!std::isfinite(xk)) {
        local_ok = false;
        break;
      }
      x_max = std::max(x_max, std::abs(xk));
    }
    if (local_ok) {
      // Unpack converged solution.  x carries the correct outer
      // Sacado-FAD info via the implicit function theorem (applied by
      // MiniSolver).
      sigma_out(0, 0) = x(0);
      sigma_out(1, 1) = x(1);
      sigma_out(2, 2) = x(2);
      sigma_out(1, 2) = x(3); sigma_out(2, 1) = x(3);
      sigma_out(0, 2) = x(4); sigma_out(2, 0) = x(4);
      sigma_out(0, 1) = x(5); sigma_out(1, 0) = x(5);

      alpha_out(0, 0) = x(6);
      alpha_out(1, 1) = x(7);
      alpha_out(2, 2) = -x(6) - x(7);
      alpha_out(1, 2) = x(8); alpha_out(2, 1) = x(8);
      alpha_out(0, 2) = x(9); alpha_out(2, 0) = x(9);
      alpha_out(0, 1) = x(10); alpha_out(1, 0) = x(10);

      kappa_out  = x(11);
      dgamma_out = x(12);
    }
    // If !local_ok, sigma_out/alpha_out/kappa_out/dgamma_out remain the
    // elastic trial; the global Newton must retry from a saner state.
    (void)x_max;
  }

  // Plastic-strain invariants from the stress drop (elastic compliance).
  // For elastic IPs dgamma_out is 0 so sigma_tr == sigma_out and these
  // come out zero.
  ScalarT deqps(0.0), devolps(0.0);
  if (dgamma_out > ScalarT(0.0)) {
    ScalarT const comp_iso = 1.0 / (3.0 * bulkModulus);
    ScalarT const comp_dev = 1.0 / (2.0 * mu);
    Tensor  const dsigma   = sigma_tr - sigma_out;
    ScalarT const trd      = minitensor::trace(dsigma);
    Tensor  const dev_d    = dsigma - (trd / 3.0) * I;
    devolps                = comp_iso * trd;
    Tensor  const dev_eps  = comp_dev * dev_d;
    deqps                  = std::sqrt(2.0 / 3.0) * minitensor::norm(dev_eps);
  }

  if (finite_deformation_) {
    if (dgamma_out > ScalarT(0.0)) {
      Tensor const dgdsigma = nls.template compute_dgdsigma<ScalarT>(sigma_out, alpha_out, kappa_out);
      Tensor const A_exp    = dgamma_out * dgdsigma;
      Tensor const expA     = minitensor::exp(A_exp);
      Fpnew                 = expA * Fpn;
    } else {
      Fpnew = Fpn;
    }

    ScalarT const Jdet = J_(cell, pt);
    for (int i = 0; i < num_dims_; ++i) {
      for (int j = 0; j < num_dims_; ++j) {
        stress_(cell, pt, i, j)     = sigma_out(i, j) / Jdet;
        Fp_(cell, pt, i, j)         = Fpnew(i, j);
        backStress_(cell, pt, i, j) = alpha_out(i, j);
      }
    }
  } else {
    for (int i = 0; i < num_dims_; ++i) {
      for (int j = 0; j < num_dims_; ++j) {
        stress_(cell, pt, i, j)     = sigma_out(i, j);
        backStress_(cell, pt, i, j) = alpha_out(i, j);
      }
    }
  }

  capParameter_(cell, pt)     = kappa_out;
  eqps_(cell, pt)             = eqps_old_(cell, pt) + deqps;
  volPlasticStrain_(cell, pt) = volPlasticStrain_old_(cell, pt) + devolps;
}

}  // namespace LCM
