// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
//
// Shared integrator for the three-invariant isotropic/kinematic hardening
// cap plasticity model: yield/potential functions, hardening laws, and the
// substepped explicit integration with drift correction. Extracted from
// the CapModel kernel so that the verified physics has exactly one
// implementation; CapModel supplies constant material parameters, while
// the Permafrost model supplies parameters that vary per integration
// point with ice saturation (which is why the parameters are ScalarT, not
// RealType). See doc/developers-guide/cap-plasticity.tex for the
// formulation, its sources, and the verification.

#if !defined(LCM_CapIntegrator_hpp)
#define LCM_CapIntegrator_hpp

#include <MiniTensor.h>

namespace LCM {

// Full parameter set of the cap model, including the elastic Lame
// constants (see the developers-guide parameter table). ScalarT so that
// callers may make them functions of other fields (e.g. ice saturation
// in the Permafrost model).
template <typename ScalarT>
struct CapParameters
{
  ScalarT A{0.0};
  ScalarT D{0.0};
  ScalarT C{0.0};
  ScalarT theta{0.0};
  ScalarT R{0.0};
  ScalarT kappa0{0.0};
  ScalarT W{0.0};
  ScalarT D1{0.0};
  ScalarT D2{0.0};
  ScalarT calpha{0.0};
  ScalarT psi{1.0};
  ScalarT N{0.0};
  ScalarT L{0.0};
  ScalarT phi{0.0};
  ScalarT Q{0.0};
  ScalarT lame{0.0};
  ScalarT mu{0.0};
};

template <typename ScalarT>
struct CapIntegrator
{
  using Tensor  = minitensor::Tensor<ScalarT>;
  using Tensor4 = minitensor::Tensor4<ScalarT>;
  using Params  = CapParameters<ScalarT>;

  // Parameter sets at the beginning and end of the strain increment.
  // When the material parameters are constant (CapModel) the two sets
  // are equal; when they evolve with an external field (Permafrost ice
  // saturation), the integrator ramps linearly between them across the
  // substeps, so that the yield surface and elastic moduli move in many
  // small increments rather than one jump. A parameter jump produces a
  // large drift projection with knife-edge consistent/normal branching,
  // which makes trajectories hypersensitive to roundoff; ramping removes
  // the jump.
  Params p0;
  Params p1;

  // Substepping controls (numerical, not material).
  double substep_tolerance{1.0e-4};
  int    max_substeps{200};

  static Params
  blend(Params const& a, Params const& b, double T)
  {
    Params r;
    r.A      = (1.0 - T) * a.A + T * b.A;
    r.D      = (1.0 - T) * a.D + T * b.D;
    r.C      = (1.0 - T) * a.C + T * b.C;
    r.theta  = (1.0 - T) * a.theta + T * b.theta;
    r.R      = (1.0 - T) * a.R + T * b.R;
    r.kappa0 = (1.0 - T) * a.kappa0 + T * b.kappa0;
    r.W      = (1.0 - T) * a.W + T * b.W;
    r.D1     = (1.0 - T) * a.D1 + T * b.D1;
    r.D2     = (1.0 - T) * a.D2 + T * b.D2;
    r.calpha = (1.0 - T) * a.calpha + T * b.calpha;
    r.psi    = (1.0 - T) * a.psi + T * b.psi;
    r.N      = (1.0 - T) * a.N + T * b.N;
    r.L      = (1.0 - T) * a.L + T * b.L;
    r.phi    = (1.0 - T) * a.phi + T * b.phi;
    r.Q      = (1.0 - T) * a.Q + T * b.Q;
    r.lame   = (1.0 - T) * a.lame + T * b.lame;
    r.mu     = (1.0 - T) * a.mu + T * b.mu;
    return r;
  }

  static Tensor4
  elastic_tangent(Params const& P)
  {
    Tensor4 const id1 = minitensor::identity_1<ScalarT>(3);
    Tensor4 const id2 = minitensor::identity_2<ScalarT>(3);
    Tensor4 const id3 = minitensor::identity_3<ScalarT>(3);
    return P.lame * id3 + P.mu * (id1 + id2);
  }

  //
  // Yield function f = Gamma^2 J2 - Fc (Ff - N)^2 on the relative
  // stress xi = sigma - alpha.
  //
  ScalarT
  compute_f(Params const& P, Tensor const& sigma, Tensor const& alpha_in, ScalarT const& kappa) const
  {
    Tensor xi = sigma - alpha_in;

    ScalarT I1 = minitensor::trace(xi), p = I1 / 3;

    Tensor s = xi - p * minitensor::identity<ScalarT>(3);

    ScalarT J2 = 0.5 * minitensor::dotdot(s, s);

    ScalarT J3 = minitensor::det(s);

    ScalarT Gamma = 1.0;
    if (P.psi != 0 && J2 != 0)
      Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) +
              (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / P.psi);

    ScalarT Ff_I1 = P.A - P.C * std::exp(P.D * I1) - P.theta * I1;

    ScalarT Ff_kappa = P.A - P.C * std::exp(P.D * kappa) - P.theta * kappa;

    ScalarT X = kappa - P.R * Ff_kappa;

    ScalarT Fc = 1.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

    return Gamma * Gamma * J2 - Fc * (Ff_I1 - P.N) * (Ff_I1 - P.N);
  }

  Tensor
  compute_dfdsigma(Params const& P, Tensor const& sigma, Tensor const& alpha_in, ScalarT const& kappa) const
  {
    Tensor const I(minitensor::eye<ScalarT>(3));

    Tensor xi = sigma - alpha_in;

    ScalarT I1 = minitensor::trace(xi), p = I1 / 3;

    Tensor s = xi - p * minitensor::identity<ScalarT>(3);

    ScalarT J2 = 0.5 * minitensor::dotdot(s, s);

    ScalarT J3 = minitensor::det(s);

    // dJ3/dsigma = s.s - (2/3) J2 I  -- matrix product s.s, NOT the
    // elementwise square (they coincide only in principal axes).
    Tensor const ss = minitensor::dot(s, s);
    Tensor dJ3dsigma(3);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        dJ3dsigma(i, j) = ss(i, j) - 2 * J2 * I(i, j) / 3;

    ScalarT Ff_I1 = P.A - P.C * std::exp(P.D * I1) - P.theta * I1;

    ScalarT Ff_kappa = P.A - P.C * std::exp(P.D * kappa) - P.theta * kappa;

    ScalarT X = kappa - P.R * Ff_kappa;

    ScalarT Fc = 1.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

    ScalarT Gamma = 1.0;
    if (P.psi != 0 && J2 != 0)
      Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) +
              (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / P.psi);

    // derivatives
    ScalarT dFfdI1 = -(P.D * P.C * std::exp(P.D * I1) + P.theta);

    ScalarT dFcdI1 = 0.0;
    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      dFcdI1 = -2.0 * (I1 - kappa) / (X - kappa) / (X - kappa);

    ScalarT dfdI1 = -(Ff_I1 - P.N) * (2 * Fc * dFfdI1 + (Ff_I1 - P.N) * dFcdI1);

    ScalarT dGammadJ2 = 0.0;
    if (J2 != 0)
      dGammadJ2 = 9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0 * (1.0 - 1.0 / P.psi);

    ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;

    ScalarT dGammadJ3 = 0.0;
    if (J2 != 0)
      dGammadJ3 = -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0 * (1.0 - 1.0 / P.psi);

    ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

    return dfdI1 * I + dfdJ2 * s + dfdJ3 * dJ3dsigma;
  }

  Tensor
  compute_dgdsigma(Params const& P, Tensor const& sigma, Tensor const& alpha_in, ScalarT const& kappa) const
  {
    Tensor const I(minitensor::eye<ScalarT>(3));

    Tensor xi = sigma - alpha_in;

    ScalarT I1 = minitensor::trace(xi), p = I1 / 3;

    Tensor s = xi - p * minitensor::identity<ScalarT>(3);

    ScalarT J2 = 0.5 * minitensor::dotdot(s, s);

    ScalarT J3 = minitensor::det(s);

    // dJ3/dsigma = s.s - (2/3) J2 I -- matrix product, see compute_dfdsigma.
    Tensor const ss = minitensor::dot(s, s);
    Tensor dJ3dsigma(3);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        dJ3dsigma(i, j) = ss(i, j) - 2 * J2 * I(i, j) / 3;

    // The plastic potential's shear surface F_f^g uses its own parameters
    // (P.L, P.phi) ...
    ScalarT Ff_I1 = P.A - P.C * std::exp(P.L * I1) - P.phi * I1;

    // ... but the potential cap position X^g(kappa) = kappa - P.Q*F_f(kappa)
    // is built from the YIELD failure function F_f (parameters P.D, P.theta):
    // Regueiro & Foster (2011) eq. (14); only the aspect ratio P.Q
    // distinguishes it from the yield cap position X.
    ScalarT Ff_kappa = P.A - P.C * std::exp(P.D * kappa) - P.theta * kappa;

    ScalarT X = kappa - P.Q * Ff_kappa;

    ScalarT Fc = 1.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

    ScalarT Gamma = 1.0;
    if (P.psi != 0 && J2 != 0)
      Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) +
              (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / P.psi);

    // derivatives
    ScalarT dFfdI1 = -(P.L * P.C * std::exp(P.L * I1) + P.phi);

    ScalarT dFcdI1 = 0.0;
    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      dFcdI1 = -2.0 * (I1 - kappa) / (X - kappa) / (X - kappa);

    ScalarT dfdI1 = -(Ff_I1 - P.N) * (2 * Fc * dFfdI1 + (Ff_I1 - P.N) * dFcdI1);

    ScalarT dGammadJ2 = 0.0;
    if (J2 != 0)
      dGammadJ2 = 9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0 * (1.0 - 1.0 / P.psi);

    ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;

    ScalarT dGammadJ3 = 0.0;
    if (J2 != 0)
      dGammadJ3 = -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0 * (1.0 - 1.0 / P.psi);

    ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

    return dfdI1 * I + dfdJ2 * s + dfdJ3 * dJ3dsigma;
  }

  ScalarT
  compute_dfdkappa(Params const& P, Tensor const& sigma, Tensor const& alpha_in, ScalarT const& kappa) const
  {
    Tensor xi = sigma - alpha_in;

    ScalarT I1 = minitensor::trace(xi);

    ScalarT Ff_I1 = P.A - P.C * std::exp(P.D * I1) - P.theta * I1;

    ScalarT Ff_kappa = P.A - P.C * std::exp(P.D * kappa) - P.theta * kappa;

    ScalarT X = kappa - P.R * Ff_kappa;

    ScalarT dFcdkappa = 0.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0)) {
      dFcdkappa = 2 * (I1 - kappa) *
          ((X - kappa) + P.R * (I1 - kappa) * (P.theta + P.D * P.C * std::exp(P.D * kappa))) /
          (X - kappa) / (X - kappa) / (X - kappa);
    }

    return -dFcdkappa * (Ff_I1 - P.N) * (Ff_I1 - P.N);
  }

  ScalarT
  compute_Galpha(Params const& P, ScalarT const& J2_alpha) const
  {
    // Clamped at zero, as in the LAME GeoModel reference (GFUN is floored
    // at 0): once the backstress reaches its limit surface, kinematic
    // hardening stops. Without the clamp, explicit-integration overshoot
    // (sqrt(J2_alpha) slightly > P.N) makes Galpha negative, REVERSING the
    // hardening direction -- which destabilizes the homogeneous solution
    // of multi-element problems (observed as a bifurcation in the
    // verification study). The papers omit the clamp.
    if (P.N != 0) {
      ScalarT Galpha = 1.0 - std::pow(J2_alpha, 0.5) / P.N;
      if (Galpha < 0.0) Galpha = 0.0;
      return Galpha;
    } else
      return 0.0;
  }

  Tensor
  compute_halpha(Params const& P, Tensor const& dgdsigma, ScalarT const& J2_alpha) const
  {
    Tensor const I(minitensor::eye<ScalarT>(3));

    ScalarT Galpha = compute_Galpha(P, J2_alpha);

    ScalarT I1 = minitensor::trace(dgdsigma), p = I1 / 3;

    Tensor s_loc = dgdsigma - p * I;

    Tensor result(3);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        result(i, j) = P.calpha * Galpha * s_loc(i, j);

    return result;
  }

  ScalarT
  compute_dedkappa(Params const& P, ScalarT const& kappa) const
  {
    // The crush curve eps_v^p(X) is calibrated against the YIELD cap
    // position X(kappa) = kappa - P.R*F_f(kappa) (parameters P.R, P.D, P.theta),
    // not the plastic-potential cap X^g: the LAME GeoModel reference uses
    // DXDKP = 1 - P.R*dF_f/dkappa and evaluates DVDX at X - X0.
    ScalarT Ff_kappa0 = P.A - P.C * std::exp(P.D * P.kappa0) - P.theta * P.kappa0;

    ScalarT X0 = P.kappa0 - P.R * Ff_kappa0;

    ScalarT Ff_kappa = P.A - P.C * std::exp(P.D * kappa) - P.theta * kappa;

    ScalarT X = kappa - P.R * Ff_kappa;

    ScalarT dedX = (P.D1 - 2 * P.D2 * (X - X0)) *
        std::exp((P.D1 - P.D2 * (X - X0)) * (X - X0)) * P.W;

    ScalarT dXdkappa = 1 + P.R * (P.D * P.C * std::exp(P.D * kappa) + P.theta);

    return dedX * dXdkappa;
  }

  ScalarT
  compute_evp(Params const& P, ScalarT const& kappa) const
  {
    // Plastic volumetric strain on the crush curve,
    //   eps_v^p = P.W (exp[P.D1 (X - X0) - P.D2 (X - X0)^2] - 1),
    // evaluated at the yield cap position X(kappa); see compute_dedkappa.
    ScalarT Ff_kappa0 = P.A - P.C * std::exp(P.D * P.kappa0) - P.theta * P.kappa0;

    ScalarT X0 = P.kappa0 - P.R * Ff_kappa0;

    ScalarT Ff_kappa = P.A - P.C * std::exp(P.D * kappa) - P.theta * kappa;

    ScalarT X = kappa - P.R * Ff_kappa;

    ScalarT dX = X - X0;

    return P.W * (std::exp(P.D1 * dX - P.D2 * dX * dX) - 1.0);
  }

  //
  // Integrate one strain increment from the converged state
  // {sigma_n, alpha, kappa}: elastic trial, then Sloan-style adaptive
  // substepping (modified-Euler RK1/RK2 pairs with relative stress-error
  // control), each accepted substep followed by the drift correction with
  // the Sloan normal-correction fallback. The parameters ramp linearly
  // from p0 to p1 across the substep pseudo-time (no-op when p0 == p1):
  // RK1 rates use the parameters at the substep start, RK2 rates and the
  // drift correction those at the substep end. Returns sigma_{n+1} and
  // advances alpha and kappa in place. The caller is responsible for
  // expressing sigma_n through the CURRENT elastic moduli when they
  // change between steps. f_tolerance should be 1.0e-12 * E^2
  // (unit-system invariant; f has units of stress^2).
  //
  Tensor
  integrate(
      Tensor const&  sigmaN,
      Tensor&        alphaVal,
      ScalarT&       kappaVal,
      Tensor const&  depsilon,
      ScalarT const& f_tolerance) const
  {
    Tensor4 const Celastic1 = elastic_tangent(p1);

    Tensor sigmaVal = sigmaN + minitensor::dotdot(Celastic1, depsilon);

    ScalarT f = compute_f(p1, sigmaVal, alphaVal, kappaVal);

    if (f <= 0.0) return sigmaVal;

    // Rate evaluation: forward-Euler increments at the given state and
    // parameter set for a substrain deps_sub. dgamma is clamped at zero
    // (elastic unloading within a substep), dkappa at zero from above
    // (no cap contraction, LAME reference).
    auto rates = [&](Params const& P, Tensor const& sig, Tensor const& alp,
                     ScalarT const& kap, Tensor const& deps_sub,
                     Tensor& dsig, Tensor& dalp, ScalarT& dkap) {
      Tensor4 const Ce = elastic_tangent(P);
      Tensor dfds = compute_dfdsigma(P, sig, alp, kap);
      Tensor dgds = compute_dgdsigma(P, sig, alp, kap);
      Tensor dfda = -dfds;
      ScalarT dfdk = compute_dfdkappa(P, sig, alp, kap);
      ScalarT J2a = 0.5 * minitensor::dotdot(alp, alp);
      Tensor ha = compute_halpha(P, dgds, J2a);
      ScalarT I1dg = minitensor::trace(dgds);
      ScalarT dedk = compute_dedkappa(P, kap);

      ScalarT hk;
      if (dedk != 0.0)
        hk = I1dg / dedk;
      else
        hk = 0.0;

      // Cap-lock saturation (LAME GeoModel reference): once the crush
      // curve is exhausted the cap must stop evolving.
      ScalarT evp_now = compute_evp(P, kap);
      ScalarT const bulk = P.lame + 2.0 * P.mu / 3.0;
      if (std::abs(evp_now) >= std::abs(P.W))
        hk = -0.01 * bulk * bulk;

      ScalarT kai = minitensor::dotdot(dfds, minitensor::dotdot(Ce, dgds))
                    - minitensor::dotdot(dfda, ha) - dfdk * hk;

      ScalarT dgam;
      if (kai != 0.0)
        dgam = minitensor::dotdot(minitensor::dotdot(dfds, Ce), deps_sub) / kai;
      else
        dgam = 0.0;
      if (dgam < 0.0) dgam = 0.0;

      dsig = minitensor::dotdot(Ce, deps_sub) - dgam * minitensor::dotdot(Ce, dgds);
      dalp = dgam * ha;
      dkap = dgam * hk;
      if (dkap > 0.0) dkap = 0.0;
    };

    // Drift correction (Algorithm 2 of the source paper): consistent
    // correction with a fallback to the Sloan et al. (2001) normal
    // correction when the consistent step fails to reduce |f|.
    auto drift_correct = [&](Params const& P, Tensor& sig, Tensor& alp, ScalarT& kap) {
      Tensor4 const Ce = elastic_tangent(P);
      int       iteration     = 0;
      int const max_iteration = 20;
      while (true) {
        ScalarT fd = compute_f(P, sig, alp, kap);
        if (std::abs(fd) < f_tolerance) break;
        if (iteration > max_iteration) break;

        Tensor dfds = compute_dfdsigma(P, sig, alp, kap);
        Tensor dgds = compute_dgdsigma(P, sig, alp, kap);
        Tensor dfda = -dfds;
        ScalarT dfdk = compute_dfdkappa(P, sig, alp, kap);
        ScalarT J2a = 0.5 * minitensor::dotdot(alp, alp);
        Tensor ha = compute_halpha(P, dgds, J2a);
        ScalarT I1dg = minitensor::trace(dgds);
        ScalarT dedk = compute_dedkappa(P, kap);

        ScalarT hk;
        if (dedk != 0.0)
          hk = I1dg / dedk;
        else
          hk = 0.0;

        ScalarT evp_now = compute_evp(P, kap);
        ScalarT const bulk = P.lame + 2.0 * P.mu / 3.0;
        if (std::abs(evp_now) >= std::abs(P.W))
          hk = -0.01 * bulk * bulk;

        ScalarT kai = minitensor::dotdot(dfds, minitensor::dotdot(Ce, dgds));
        kai = kai - minitensor::dotdot(dfda, ha) - dfdk * hk;

        ScalarT dg;
        if (kai != 0.0)
          dg = fd / kai;
        else
          dg = 0.0;

        ScalarT dkap = dg * hk;
        if (dkap > 0.0) dkap = 0.0;

        Tensor sigK = sig - dg * minitensor::dotdot(Ce, dgds);
        Tensor alpK = alp + dg * ha;
        ScalarT kapK = kap + dkap;

        ScalarT fK = compute_f(P, sigK, alpK, kapK);

        if (std::abs(fK) > std::abs(fd)) {
          // Normal correction: pure geometric projection along df/dsigma
          // with frozen internal variables. (The as-printed Algorithm 2
          // mixes C^e into the update while omitting it from chi-tilde,
          // which is dimensionally inconsistent; this is the Sloan form.)
          ScalarT dfdotdf = minitensor::dotdot(dfds, dfds);
          if (dfdotdf != 0.0)
            dg = fd / dfdotdf;
          else
            dg = 0.0;

          sigK = sig - dg * dfds;
          alpK = alp;
          kapK = kap;
        }

        sig = sigK;
        alp = alpK;
        kap = kapK;
        iteration++;
      }
    };

    // Substepping driver. Control variables are plain doubles (Sacado
    // values) so the Residual and Jacobian evaluations take identical
    // branch sequences.
    double const STOL   = substep_tolerance;
    double const dT_min = 1.0 / static_cast<double>(max_substeps);

    sigmaVal = sigmaN;  // integrate from the converged state, not the trial
    double T  = 0.0;
    double dT = 1.0;
    int    nsub = 0;
    int const nsub_max = 4 * max_substeps;  // rejected attempts included

    while (T < 1.0 && nsub < nsub_max) {
      Tensor const deps_sub = dT * depsilon;

      Params const Pa = blend(p0, p1, T);
      Params const Pb = blend(p0, p1, T + dT);

      // RK1 increments at the current state and substep-start parameters
      Tensor dsig1(3), dalp1(3);
      ScalarT dkap1;
      rates(Pa, sigmaVal, alphaVal, kappaVal, deps_sub, dsig1, dalp1, dkap1);

      // RK2 increments at the RK1-advanced state and substep-end parameters
      Tensor const sig1 = sigmaVal + dsig1;
      Tensor const alp1 = alphaVal + dalp1;
      ScalarT const kap1 = kappaVal + dkap1;
      Tensor dsig2(3), dalp2(3);
      ScalarT dkap2;
      rates(Pb, sig1, alp1, kap1, deps_sub, dsig2, dalp2, dkap2);

      // Modified-Euler state and relative stress-error estimate
      Tensor const sig_new = sigmaVal + 0.5 * (dsig1 + dsig2);
      Tensor const alp_new = alphaVal + 0.5 * (dalp1 + dalp2);
      ScalarT const kap_new = kappaVal + 0.5 * (dkap1 + dkap2);

      double const err_abs = Sacado::ScalarValue<ScalarT>::eval(
          minitensor::norm(Tensor(dsig2 - dsig1)));
      double const sig_scale = std::max(
          Sacado::ScalarValue<ScalarT>::eval(minitensor::norm(sig_new)),
          1.0e-12);
      double const R_err = 0.5 * err_abs / sig_scale;

      if (R_err > STOL && dT > dT_min) {
        // reject: shrink the substep and retry
        double q = 0.9 * std::sqrt(STOL / R_err);
        if (q < 0.1) q = 0.1;
        dT = std::max(q * dT, dT_min);
        nsub++;
        continue;
      }

      // accept: drift-correct against the substep-end surface, advance
      // pseudo-time, grow the substep
      sigmaVal = sig_new;
      alphaVal = alp_new;
      kappaVal = kap_new;
      drift_correct(Pb, sigmaVal, alphaVal, kappaVal);

      T += dT;
      double q = 0.9 * std::sqrt(STOL / std::max(R_err, 1.0e-30));
      if (q > 1.1) q = 1.1;
      dT = std::min(q * dT, 1.0 - T);
      if (dT < dT_min) dT = std::min(dT_min, 1.0 - T);
      nsub++;
    }

    return sigmaVal;
  }
};

}  // namespace LCM

#endif  // LCM_CapIntegrator_hpp
