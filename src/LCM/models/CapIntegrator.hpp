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
// RealType). See doc/developersGuide/cap_plasticity.tex for the
// formulation, its sources, and the verification.

#if !defined(LCM_CapIntegrator_hpp)
#define LCM_CapIntegrator_hpp

#include <MiniTensor.h>

namespace LCM {

template <typename ScalarT>
struct CapIntegrator
{
  using Tensor  = minitensor::Tensor<ScalarT>;
  using Tensor4 = minitensor::Tensor4<ScalarT>;

  // Material parameters (see the developers-guide parameter table).
  // ScalarT so that callers may make them functions of other fields
  // (e.g. ice saturation in the Permafrost model).
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

  // Substepping controls (numerical, not material).
  double substep_tolerance{1.0e-4};
  int    max_substeps{200};

  //
  // Yield function f = Gamma^2 J2 - Fc (Ff - N)^2 on the relative
  // stress xi = sigma - alpha.
  //
  ScalarT
  compute_f(Tensor const& sigma, Tensor const& alpha_in, ScalarT const& kappa) const
  {
    Tensor xi = sigma - alpha_in;

    ScalarT I1 = minitensor::trace(xi), p = I1 / 3;

    Tensor s = xi - p * minitensor::identity<ScalarT>(3);

    ScalarT J2 = 0.5 * minitensor::dotdot(s, s);

    ScalarT J3 = minitensor::det(s);

    ScalarT Gamma = 1.0;
    if (psi != 0 && J2 != 0)
      Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) +
              (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

    ScalarT Ff_I1 = A - C * std::exp(D * I1) - theta * I1;

    ScalarT Ff_kappa = A - C * std::exp(D * kappa) - theta * kappa;

    ScalarT X = kappa - R * Ff_kappa;

    ScalarT Fc = 1.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

    return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
  }

  Tensor
  compute_dfdsigma(Tensor const& sigma, Tensor const& alpha_in, ScalarT const& kappa) const
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

    ScalarT Ff_I1 = A - C * std::exp(D * I1) - theta * I1;

    ScalarT Ff_kappa = A - C * std::exp(D * kappa) - theta * kappa;

    ScalarT X = kappa - R * Ff_kappa;

    ScalarT Fc = 1.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

    ScalarT Gamma = 1.0;
    if (psi != 0 && J2 != 0)
      Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) +
              (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

    // derivatives
    ScalarT dFfdI1 = -(D * C * std::exp(D * I1) + theta);

    ScalarT dFcdI1 = 0.0;
    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      dFcdI1 = -2.0 * (I1 - kappa) / (X - kappa) / (X - kappa);

    ScalarT dfdI1 = -(Ff_I1 - N) * (2 * Fc * dFfdI1 + (Ff_I1 - N) * dFcdI1);

    ScalarT dGammadJ2 = 0.0;
    if (J2 != 0)
      dGammadJ2 = 9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0 * (1.0 - 1.0 / psi);

    ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;

    ScalarT dGammadJ3 = 0.0;
    if (J2 != 0)
      dGammadJ3 = -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0 * (1.0 - 1.0 / psi);

    ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

    return dfdI1 * I + dfdJ2 * s + dfdJ3 * dJ3dsigma;
  }

  Tensor
  compute_dgdsigma(Tensor const& sigma, Tensor const& alpha_in, ScalarT const& kappa) const
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
    // (L, phi) ...
    ScalarT Ff_I1 = A - C * std::exp(L * I1) - phi * I1;

    // ... but the potential cap position X^g(kappa) = kappa - Q*F_f(kappa)
    // is built from the YIELD failure function F_f (parameters D, theta):
    // Regueiro & Foster (2011) eq. (14); only the aspect ratio Q
    // distinguishes it from the yield cap position X.
    ScalarT Ff_kappa = A - C * std::exp(D * kappa) - theta * kappa;

    ScalarT X = kappa - Q * Ff_kappa;

    ScalarT Fc = 1.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

    ScalarT Gamma = 1.0;
    if (psi != 0 && J2 != 0)
      Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) +
              (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

    // derivatives
    ScalarT dFfdI1 = -(L * C * std::exp(L * I1) + phi);

    ScalarT dFcdI1 = 0.0;
    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      dFcdI1 = -2.0 * (I1 - kappa) / (X - kappa) / (X - kappa);

    ScalarT dfdI1 = -(Ff_I1 - N) * (2 * Fc * dFfdI1 + (Ff_I1 - N) * dFcdI1);

    ScalarT dGammadJ2 = 0.0;
    if (J2 != 0)
      dGammadJ2 = 9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0 * (1.0 - 1.0 / psi);

    ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;

    ScalarT dGammadJ3 = 0.0;
    if (J2 != 0)
      dGammadJ3 = -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0 * (1.0 - 1.0 / psi);

    ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

    return dfdI1 * I + dfdJ2 * s + dfdJ3 * dJ3dsigma;
  }

  ScalarT
  compute_dfdkappa(Tensor const& sigma, Tensor const& alpha_in, ScalarT const& kappa) const
  {
    Tensor xi = sigma - alpha_in;

    ScalarT I1 = minitensor::trace(xi);

    ScalarT Ff_I1 = A - C * std::exp(D * I1) - theta * I1;

    ScalarT Ff_kappa = A - C * std::exp(D * kappa) - theta * kappa;

    ScalarT X = kappa - R * Ff_kappa;

    ScalarT dFcdkappa = 0.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0)) {
      dFcdkappa = 2 * (I1 - kappa) *
          ((X - kappa) + R * (I1 - kappa) * (theta + D * C * std::exp(D * kappa))) /
          (X - kappa) / (X - kappa) / (X - kappa);
    }

    return -dFcdkappa * (Ff_I1 - N) * (Ff_I1 - N);
  }

  ScalarT
  compute_Galpha(ScalarT const& J2_alpha) const
  {
    // Clamped at zero, as in the LAME GeoModel reference (GFUN is floored
    // at 0): once the backstress reaches its limit surface, kinematic
    // hardening stops. Without the clamp, explicit-integration overshoot
    // (sqrt(J2_alpha) slightly > N) makes Galpha negative, REVERSING the
    // hardening direction -- which destabilizes the homogeneous solution
    // of multi-element problems (observed as a bifurcation in the
    // verification study). The papers omit the clamp.
    if (N != 0) {
      ScalarT Galpha = 1.0 - std::pow(J2_alpha, 0.5) / N;
      if (Galpha < 0.0) Galpha = 0.0;
      return Galpha;
    } else
      return 0.0;
  }

  Tensor
  compute_halpha(Tensor const& dgdsigma, ScalarT const& J2_alpha) const
  {
    Tensor const I(minitensor::eye<ScalarT>(3));

    ScalarT Galpha = compute_Galpha(J2_alpha);

    ScalarT I1 = minitensor::trace(dgdsigma), p = I1 / 3;

    Tensor s_loc = dgdsigma - p * I;

    Tensor result(3);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        result(i, j) = calpha * Galpha * s_loc(i, j);

    return result;
  }

  ScalarT
  compute_dedkappa(ScalarT const& kappa) const
  {
    // The crush curve eps_v^p(X) is calibrated against the YIELD cap
    // position X(kappa) = kappa - R*F_f(kappa) (parameters R, D, theta),
    // not the plastic-potential cap X^g: the LAME GeoModel reference uses
    // DXDKP = 1 - R*dF_f/dkappa and evaluates DVDX at X - X0.
    ScalarT Ff_kappa0 = A - C * std::exp(D * kappa0) - theta * kappa0;

    ScalarT X0 = kappa0 - R * Ff_kappa0;

    ScalarT Ff_kappa = A - C * std::exp(D * kappa) - theta * kappa;

    ScalarT X = kappa - R * Ff_kappa;

    ScalarT dedX = (D1 - 2 * D2 * (X - X0)) *
        std::exp((D1 - D2 * (X - X0)) * (X - X0)) * W;

    ScalarT dXdkappa = 1 + R * (D * C * std::exp(D * kappa) + theta);

    return dedX * dXdkappa;
  }

  ScalarT
  compute_evp(ScalarT const& kappa) const
  {
    // Plastic volumetric strain on the crush curve,
    //   eps_v^p = W (exp[D1 (X - X0) - D2 (X - X0)^2] - 1),
    // evaluated at the yield cap position X(kappa); see compute_dedkappa.
    ScalarT Ff_kappa0 = A - C * std::exp(D * kappa0) - theta * kappa0;

    ScalarT X0 = kappa0 - R * Ff_kappa0;

    ScalarT Ff_kappa = A - C * std::exp(D * kappa) - theta * kappa;

    ScalarT X = kappa - R * Ff_kappa;

    ScalarT dX = X - X0;

    return W * (std::exp(D1 * dX - D2 * dX * dX) - 1.0);
  }

  //
  // Integrate one strain increment from the converged state
  // {sigma_n, alpha, kappa}: elastic trial, then Sloan-style adaptive
  // substepping (modified-Euler RK1/RK2 pairs with relative stress-error
  // control), each accepted substep followed by the drift correction with
  // the Sloan normal-correction fallback. Returns sigma_{n+1} and
  // advances alpha and kappa in place. f_tolerance should be
  // 1.0e-12 * E^2 (unit-system invariant; f has units of stress^2).
  //
  Tensor
  integrate(
      Tensor const&  sigmaN,
      Tensor&        alphaVal,
      ScalarT&       kappaVal,
      Tensor const&  depsilon,
      Tensor4 const& Celastic,
      ScalarT const& bulkModulus,
      ScalarT const& f_tolerance) const
  {
    Tensor sigmaVal = sigmaN + minitensor::dotdot(Celastic, depsilon);

    ScalarT f = compute_f(sigmaVal, alphaVal, kappaVal);

    if (f <= 0.0) return sigmaVal;

    // Rate evaluation: forward-Euler increments at the given state for a
    // substrain deps_sub. dgamma is clamped at zero (elastic unloading
    // within a substep), dkappa at zero from above (no cap contraction,
    // LAME reference).
    auto rates = [&](Tensor const& sig, Tensor const& alp, ScalarT const& kap,
                     Tensor const& deps_sub,
                     Tensor& dsig, Tensor& dalp, ScalarT& dkap) {
      Tensor dfds = compute_dfdsigma(sig, alp, kap);
      Tensor dgds = compute_dgdsigma(sig, alp, kap);
      Tensor dfda = -dfds;
      ScalarT dfdk = compute_dfdkappa(sig, alp, kap);
      ScalarT J2a = 0.5 * minitensor::dotdot(alp, alp);
      Tensor ha = compute_halpha(dgds, J2a);
      ScalarT I1dg = minitensor::trace(dgds);
      ScalarT dedk = compute_dedkappa(kap);

      ScalarT hk;
      if (dedk != 0.0)
        hk = I1dg / dedk;
      else
        hk = 0.0;

      // Cap-lock saturation (LAME GeoModel reference): once the crush
      // curve is exhausted the cap must stop evolving.
      ScalarT evp_now = compute_evp(kap);
      if (std::abs(evp_now) >= std::abs(W))
        hk = -0.01 * bulkModulus * bulkModulus;

      ScalarT kai = minitensor::dotdot(dfds, minitensor::dotdot(Celastic, dgds))
                    - minitensor::dotdot(dfda, ha) - dfdk * hk;

      ScalarT dgam;
      if (kai != 0.0)
        dgam = minitensor::dotdot(minitensor::dotdot(dfds, Celastic), deps_sub) / kai;
      else
        dgam = 0.0;
      if (dgam < 0.0) dgam = 0.0;

      dsig = minitensor::dotdot(Celastic, deps_sub) - dgam * minitensor::dotdot(Celastic, dgds);
      dalp = dgam * ha;
      dkap = dgam * hk;
      if (dkap > 0.0) dkap = 0.0;
    };

    // Drift correction (Algorithm 2 of the source paper): consistent
    // correction with a fallback to the Sloan et al. (2001) normal
    // correction when the consistent step fails to reduce |f|.
    auto drift_correct = [&](Tensor& sig, Tensor& alp, ScalarT& kap) {
      int       iteration     = 0;
      int const max_iteration = 20;
      while (true) {
        ScalarT fd = compute_f(sig, alp, kap);
        if (std::abs(fd) < f_tolerance) break;
        if (iteration > max_iteration) break;

        Tensor dfds = compute_dfdsigma(sig, alp, kap);
        Tensor dgds = compute_dgdsigma(sig, alp, kap);
        Tensor dfda = -dfds;
        ScalarT dfdk = compute_dfdkappa(sig, alp, kap);
        ScalarT J2a = 0.5 * minitensor::dotdot(alp, alp);
        Tensor ha = compute_halpha(dgds, J2a);
        ScalarT I1dg = minitensor::trace(dgds);
        ScalarT dedk = compute_dedkappa(kap);

        ScalarT hk;
        if (dedk != 0.0)
          hk = I1dg / dedk;
        else
          hk = 0.0;

        ScalarT evp_now = compute_evp(kap);
        if (std::abs(evp_now) >= std::abs(W))
          hk = -0.01 * bulkModulus * bulkModulus;

        ScalarT kai = minitensor::dotdot(dfds, minitensor::dotdot(Celastic, dgds));
        kai = kai - minitensor::dotdot(dfda, ha) - dfdk * hk;

        ScalarT dg;
        if (kai != 0.0)
          dg = fd / kai;
        else
          dg = 0.0;

        ScalarT dkap = dg * hk;
        if (dkap > 0.0) dkap = 0.0;

        Tensor sigK = sig - dg * minitensor::dotdot(Celastic, dgds);
        Tensor alpK = alp + dg * ha;
        ScalarT kapK = kap + dkap;

        ScalarT fK = compute_f(sigK, alpK, kapK);

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

      // RK1 increments at the current state
      Tensor dsig1(3), dalp1(3);
      ScalarT dkap1;
      rates(sigmaVal, alphaVal, kappaVal, deps_sub, dsig1, dalp1, dkap1);

      // RK2 increments at the RK1-advanced state
      Tensor const sig1 = sigmaVal + dsig1;
      Tensor const alp1 = alphaVal + dalp1;
      ScalarT const kap1 = kappaVal + dkap1;
      Tensor dsig2(3), dalp2(3);
      ScalarT dkap2;
      rates(sig1, alp1, kap1, deps_sub, dsig2, dalp2, dkap2);

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

      // accept: drift-correct, advance pseudo-time, grow the substep
      sigmaVal = sig_new;
      alphaVal = alp_new;
      kappaVal = kap_new;
      drift_correct(sigmaVal, alphaVal, kappaVal);

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
