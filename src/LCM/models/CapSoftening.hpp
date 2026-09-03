// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_CapSoftening_hpp)
#define LCM_CapSoftening_hpp

#include <cmath>

#include "CapIntegrator.hpp"

namespace LCM {

// Strain softening of the cap model's shear strength by loss of cohesion:
// the ice-bond breakage of frozen soil, or the cement breakage of a weak
// rock. Shared by the CapModel and Permafrost kernels, which own the two
// states it needs (Coherence and Damage_Strain) and call it in the same
// three places.
//
// The form is the one in the LAME Kayenta model (kayenta_model.F,
// KMM_UPDSFT), the production descendant of the GeoModel this cap model is
// transcribed from, with three choices made explicit:
//
//   * A scalar COHERENCE omega, 1 intact, falls toward a residual as
//     damage strain accumulates. Kayenta calls it COHER, "1.0 minus
//     damage", and blends an intact and a failed limit surface with it.
//     Here the failed surface is the intact one with its cohesion
//     reduced: friction (theta), the envelope curvature (C, D) and every
//     cap parameter are untouched. That is the reduction the frozen-sand
//     triaxial series supports (the strength lost is 70 to 80 per cent of
//     the cohesive share at every confining pressure, and none of the
//     frictional one), and it is exactly the algebra the Permafrost model
//     already uses for ocean-exposure weakening:
//         C -> A - omega (A - C)        so   A - C -> omega (A - C)
//         N -> omega N                  the hardening room shrinks with it
//
//   * Onset when the kinematic hardening is exhausted (Kayenta:
//     SV(KCRACK) set when GFUN <= 1e-6), i.e. when the back stress has
//     reached the failure envelope, sqrt(J2(alpha))/N >= 1 - 1e-6, and only
//     on the shear branch I1 >= kappa (Kayenta: .NOT.CONFINED). Nothing
//     softens during hardening or on the compaction cap, so the design rule
//     that environmental softening never flows through kappa holds here too.
//
//   * The driver is the equivalent plastic strain accumulated after onset
//     (Kayenta's strain mode, NSOFTSHEAR = 3, TGROW += EQDE/XC1), and the
//     law is Kayenta's logistic in that strain,
//         c(s) = (1 + exp(-2k)) / (1 + exp(-2k (1 - s/eps_f)))
//     which is 1 at s = 0, 1/2 at s = eps_f, and 0 as s -> infinity, with
//     k the "failure speed" setting how abrupt the fall is. The coherence
//     is omega = omega_res + (1 - omega_res) c(s) (Kayenta's COHEROLDB with
//     SFRATIO = omega_res), and it never heals.
//
// Left out on purpose, both available in Kayenta: the reduction of the
// elastic moduli by coherence (nothing in a monotonic triaxial test
// constrains it) and the time-based driver (rate dependence through a crack
// growth time; the natural hook when creep comes up).
//
// A material point softens; a boundary-value problem with a softening
// material localizes and its solution depends on the mesh until a length
// scale is introduced. That regularization is a separate decision and is
// not made here.
template <typename ScalarT>
struct CapSoftening
{
  using Params = CapParameters<ScalarT>;

  bool     enabled{false};
  RealType residual{1.0};        // omega_res, coherence at full damage, in (0, 1]
  RealType failure_strain{0.0};  // eps_f, damage strain at which c = 1/2
  RealType failure_speed{0.0};   // k, sharpness of the fall

  // Kayenta's onset criterion, GFUN <= 1e-6, written on the ratio it tests.
  static constexpr double onset_tolerance = 1.0e-6;

  // Coherence at a given accumulated damage strain.
  ScalarT
  coherence(ScalarT const& damage_strain) const
  {
    if (!enabled) return ScalarT(1.0);
    ScalarT const x = damage_strain / failure_strain;
    ScalarT const c = (1.0 + std::exp(-2.0 * failure_speed)) /
                      (1.0 + std::exp(-2.0 * failure_speed * (1.0 - x)));
    return residual + (1.0 - residual) * c;
  }

  // Reduce the cohesive group of a parameter set by the coherence. The
  // strength lost is omega (A - C) of the zero-pressure intercept and the
  // same fraction of the kinematic hardening room N.
  static void
  apply(Params& P, ScalarT const& omega)
  {
    P.C = P.A - omega * (P.A - P.C);
    P.N = omega * P.N;
  }

  // When N shrinks the back stress may sit outside its new bounding
  // surface sqrt(J2(alpha)) <= N. Scale it back onto that surface so the
  // total reach of the translated yield surface, (Ff - N) + |alpha|, is the
  // softened envelope Ff and not more. The clamp in the hardening law only
  // stops growth; it does not pull an existing back stress in.
  template <typename Tensor>
  static void
  project_backstress(Tensor& alpha, ScalarT const& N)
  {
    ScalarT const J2a = 0.5 * minitensor::dotdot(alpha, alpha);
    ScalarT const mag = std::sqrt(J2a);
    if (mag > N && mag > 0.0) alpha = (N / mag) * alpha;
  }

  // Advance the damage strain across a step. `started` is whether damage
  // had already begun before this step (damage_strain_old > 0); onset is
  // detected on the state at the END of the step, so the increment that
  // exhausts the hardening is the first one counted, as in Kayenta.
  template <typename Tensor>
  ScalarT
  advance(
      ScalarT const& damage_strain_old,
      ScalarT const& deqps,
      Tensor const&  sigma,
      Tensor const&  alpha,
      ScalarT const& kappa,
      ScalarT const& N) const
  {
    if (!enabled) return damage_strain_old;
    bool const started = damage_strain_old > 0.0;
    bool const on_shear_branch = minitensor::trace(sigma) >= kappa;
    ScalarT const mag = std::sqrt(0.5 * minitensor::dotdot(alpha, alpha));
    bool const exhausted = N > 0.0 && mag >= (1.0 - onset_tolerance) * N;
    if ((started || exhausted) && on_shear_branch) return damage_strain_old + deqps;
    return damage_strain_old;
  }
};

}  // namespace LCM

#endif  // LCM_CapSoftening_hpp
