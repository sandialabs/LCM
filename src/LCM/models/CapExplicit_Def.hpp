// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include <MiniTensor.h>

#include "Albany_Utils.hpp"
#include "CapExplicit.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
CapExplicitKernel<EvalT, Traits>::CapExplicitKernel(
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
  // retrieve appropriate field name strings
  std::string const cauchy_string           = field_name_map_["Cauchy_Stress"];
  std::string const backStress_string       = field_name_map_["Back_Stress"];
  std::string const capParameter_string     = field_name_map_["Cap_Parameter"];
  std::string const eqps_string             = field_name_map_["eqps"];
  std::string const volPlasticStrain_string = field_name_map_["volPlastic_Strain"];

  // define the dependent fields
  setDependentField("Poissons Ratio", dl->qp_scalar);
  setDependentField("Elastic Modulus", dl->qp_scalar);

  if (finite_deformation_) {
    std::string const F_string  = field_name_map_["F"];
    std::string const J_string  = field_name_map_["J"];
    std::string const Fp_string = field_name_map_["Fp"];

    setDependentField(F_string, dl->qp_tensor);
    setDependentField(J_string, dl->qp_scalar);

    // Fp is evaluated and a state variable
    setEvaluatedField(Fp_string, dl->qp_tensor);
    addStateVariable(Fp_string, dl->qp_tensor, "identity", 0.0, true,
                     p->get<bool>("Output Fp", false));
  } else {
    std::string const strain_string = field_name_map_["Strain"];

    setDependentField("Strain", dl->qp_tensor);

    // strain is a state variable (old state needed)
    addStateVariable(strain_string, dl->qp_tensor, "scalar", 0.0, true, true);
  }

  // define the evaluated fields
  setEvaluatedField(cauchy_string, dl->qp_tensor);
  setEvaluatedField(backStress_string, dl->qp_tensor);
  setEvaluatedField(capParameter_string, dl->qp_scalar);
  setEvaluatedField(eqps_string, dl->qp_scalar);
  setEvaluatedField(volPlasticStrain_string, dl->qp_scalar);

  // define the state variables
  // stress
  addStateVariable(cauchy_string, dl->qp_tensor, "scalar", 0.0, true, true);

  // backStress
  addStateVariable(backStress_string, dl->qp_tensor, "scalar", 0.0, true, true);

  // capParameter
  addStateVariable(capParameter_string, dl->qp_scalar, "scalar", kappa0_, true, true);

  // eqps
  addStateVariable(eqps_string, dl->qp_scalar, "scalar", 0.0, true, true);

  // volPlasticStrain
  addStateVariable(volPlasticStrain_string, dl->qp_scalar, "scalar", 0.0, true, true);
}

template <typename EvalT, typename Traits>
void
CapExplicitKernel<EvalT, Traits>::init(
    Workset&                workset,
    FieldMap<ScalarT const>& dep_fields,
    FieldMap<ScalarT>&       eval_fields)
{
  std::string cauchy_string           = field_name_map_["Cauchy_Stress"];
  std::string backStress_string       = field_name_map_["Back_Stress"];
  std::string capParameter_string     = field_name_map_["Cap_Parameter"];
  std::string eqps_string             = field_name_map_["eqps"];
  std::string volPlasticStrain_string = field_name_map_["volPlastic_Strain"];

  // extract dependent MDFields
  elastic_modulus_ = *dep_fields["Elastic Modulus"];
  poissons_ratio_  = *dep_fields["Poissons Ratio"];

  if (finite_deformation_) {
    std::string F_string  = field_name_map_["F"];
    std::string J_string  = field_name_map_["J"];
    std::string Fp_string = field_name_map_["Fp"];

    def_grad_ = *dep_fields[F_string];
    J_        = *dep_fields[J_string];

    // extract evaluated MDFields
    Fp_ = *eval_fields[Fp_string];

    // get old state
    Fp_old_ = (*workset.stateArrayPtr)[Fp_string + "_old"];
  } else {
    std::string strain_string = field_name_map_["Strain"];

    strain_ = *dep_fields["Strain"];

    // get old state
    strain_old_ = (*workset.stateArrayPtr)[strain_string + "_old"];
  }

  // extract evaluated MDFields
  stress_           = *eval_fields[cauchy_string];
  backStress_       = *eval_fields[backStress_string];
  capParameter_     = *eval_fields[capParameter_string];
  eqps_             = *eval_fields[eqps_string];
  volPlasticStrain_ = *eval_fields[volPlasticStrain_string];

  // get old state variables
  stress_old_           = (*workset.stateArrayPtr)[cauchy_string + "_old"];
  backStress_old_       = (*workset.stateArrayPtr)[backStress_string + "_old"];
  capParameter_old_     = (*workset.stateArrayPtr)[capParameter_string + "_old"];
  eqps_old_             = (*workset.stateArrayPtr)[eqps_string + "_old"];
  volPlasticStrain_old_ = (*workset.stateArrayPtr)[volPlasticStrain_string + "_old"];
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
CapExplicitKernel<EvalT, Traits>::operator()(int cell, int pt) const
{
  using Tensor  = minitensor::Tensor<ScalarT>;
  using Tensor4 = minitensor::Tensor4<ScalarT>;

  Tensor const I(minitensor::eye<ScalarT>(num_dims_));

  // local parameters
  ScalarT const E  = elastic_modulus_(cell, pt);
  ScalarT const nu = poissons_ratio_(cell, pt);
  ScalarT const lame        = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
  ScalarT const mu          = E / 2.0 / (1.0 + nu);
  ScalarT const bulkModulus = lame + (2.0 / 3.0) * mu;

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

  // Trial stress computation - depends on kinematics
  Tensor sigmaVal(num_dims_);
  Tensor sigmaTr(num_dims_);
  Tensor sigmaN(num_dims_);
  Tensor depsilon(num_dims_);

  // Variables for FD path
  Tensor Fpn(num_dims_);
  Tensor Fpnew(num_dims_);

  if (finite_deformation_) {
    // Finite deformation path
    Tensor F(num_dims_);
    F.fill(def_grad_, cell, pt, 0, 0);

    for (int i = 0; i < num_dims_; ++i)
      for (int j = 0; j < num_dims_; ++j)
        Fpn(i, j) = ScalarT(Fp_old_(cell, pt, i, j));

    // Compute trial elastic log strain
    Tensor const Fpinv = minitensor::inverse(Fpn);
    Tensor const Cpinv = Fpinv * minitensor::transpose(Fpinv);
    Tensor const be    = F * Cpinv * minitensor::transpose(F);
    Tensor const logbe = minitensor::log_sym<ScalarT>(be);
    Tensor const eps_e = 0.5 * logbe;

    // Trial Kirchhoff stress
    sigmaTr  = lame * minitensor::trace(eps_e) * I + 2.0 * mu * eps_e;
    sigmaVal = sigmaTr;
    sigmaN   = sigmaTr;
  } else {
    // Small strain path
    for (int i = 0; i < num_dims_; ++i) {
      for (int j = 0; j < num_dims_; ++j) {
        depsilon(i, j) = strain_(cell, pt, i, j) - strain_old_(cell, pt, i, j);
        sigmaN(i, j)   = stress_old_(cell, pt, i, j);
      }
    }

    sigmaVal = sigmaN + minitensor::dotdot(Celastic, depsilon);
    sigmaTr  = sigmaVal;
  }

  // Store previous back stress for later use
  Tensor alphaTr = alphaVal;

  // Plastic strain increment invariants
  ScalarT deqps(0.0), devolps(0.0);

  // Check yielding
  ScalarT f = compute_f(sigmaVal, alphaVal, kappaVal);

  // Plastic correction
  ScalarT dgamma = 0.0;
  if (f > 0.0) {
    Tensor dfdsigma_loc(num_dims_);
    Tensor dgdsigma_loc(num_dims_);
    Tensor dfdalpha_loc(num_dims_);
    Tensor halpha_loc(num_dims_);
    Tensor dfdotCe(num_dims_);

    if (!finite_deformation_) {
      // Small strain: initial predictor step
      dfdsigma_loc = compute_dfdsigma(sigmaN, alphaVal, kappaVal);
      dgdsigma_loc = compute_dgdsigma(sigmaN, alphaVal, kappaVal);
      dfdalpha_loc = -dfdsigma_loc;
      ScalarT dfdkappa = compute_dfdkappa(sigmaN, alphaVal, kappaVal);
      ScalarT J2_alpha = 0.5 * minitensor::dotdot(alphaVal, alphaVal);
      halpha_loc = compute_halpha(dgdsigma_loc, J2_alpha);
      ScalarT I1_dgdsigma = minitensor::trace(dgdsigma_loc);
      ScalarT dedkappa = compute_dedkappa(kappaVal);

      ScalarT hkappa;
      if (dedkappa != 0.0)
        hkappa = I1_dgdsigma / dedkappa;
      else
        hkappa = 0.0;

      ScalarT kai = minitensor::dotdot(dfdsigma_loc, minitensor::dotdot(Celastic, dgdsigma_loc))
                    - minitensor::dotdot(dfdalpha_loc, halpha_loc) - dfdkappa * hkappa;

      dfdotCe = minitensor::dotdot(dfdsigma_loc, Celastic);

      if (kai != 0.0)
        dgamma = minitensor::dotdot(dfdotCe, depsilon) / kai;
      else
        dgamma = 0.0;

      // initial update
      sigmaVal -= dgamma * minitensor::dotdot(Celastic, dgdsigma_loc);
      alphaVal += dgamma * halpha_loc;

      // restrictions on kappa
      ScalarT dkappa = dgamma * hkappa;
      if (dkappa > 0.0) dkappa = 0.0;
      kappaVal += dkappa;
    }

    // Stress correction loop (used by both paths)
    bool     condition     = false;
    int      iteration     = 0;
    int      max_iteration = finite_deformation_ ? 100 : 20;
    RealType tolerance     = 1.0e-10;

    while (condition == false) {
      f = compute_f(sigmaVal, alphaVal, kappaVal);
      if (std::abs(f) < tolerance) break;
      if (iteration > max_iteration) break;

      dfdsigma_loc = compute_dfdsigma(sigmaVal, alphaVal, kappaVal);
      dgdsigma_loc = compute_dgdsigma(sigmaVal, alphaVal, kappaVal);
      dfdalpha_loc = -dfdsigma_loc;
      ScalarT dfdkappa = compute_dfdkappa(sigmaVal, alphaVal, kappaVal);
      ScalarT J2_alpha = 0.5 * minitensor::dotdot(alphaVal, alphaVal);
      halpha_loc = compute_halpha(dgdsigma_loc, J2_alpha);
      ScalarT I1_dgdsigma = minitensor::trace(dgdsigma_loc);
      ScalarT dedkappa = compute_dedkappa(kappaVal);

      ScalarT hkappa;
      if (dedkappa != 0.0)
        hkappa = I1_dgdsigma / dedkappa;
      else
        hkappa = 0.0;

      ScalarT kai = minitensor::dotdot(dfdsigma_loc, minitensor::dotdot(Celastic, dgdsigma_loc));
      kai = kai - minitensor::dotdot(dfdalpha_loc, halpha_loc) - dfdkappa * hkappa;

      ScalarT delta_gamma;
      if (kai != 0.0)
        delta_gamma = f / kai;
      else
        delta_gamma = 0.0;

      // restrictions on kappa
      ScalarT dkappa = delta_gamma * hkappa;
      if (dkappa > 0.0) dkappa = 0.0;

      // trial update
      Tensor sigmaK = sigmaVal - delta_gamma * minitensor::dotdot(Celastic, dgdsigma_loc);
      Tensor alphaK = alphaVal + delta_gamma * halpha_loc;
      ScalarT kappaK = kappaVal + dkappa;

      if (!finite_deformation_) {
        // Check if correction improves the yield function
        ScalarT fK = compute_f(sigmaK, alphaK, kappaK);

        if (std::abs(fK) > std::abs(f)) {
          // Normal correction
          ScalarT dfdotdf = minitensor::dotdot(dfdsigma_loc, dfdsigma_loc);
          if (dfdotdf != 0.0)
            delta_gamma = f / dfdotdf;
          else
            delta_gamma = 0.0;

          sigmaK = sigmaVal - delta_gamma * dfdsigma_loc;
          alphaK = alphaVal;
          kappaK = kappaVal;
        }
      }

      sigmaVal = sigmaK;
      alphaVal = alphaK;
      kappaVal = kappaK;
      dgamma += delta_gamma;

      iteration++;
    }  // end stress correction loop

    // Compute plastic strain increment
    Tensor dsigma       = sigmaTr - sigmaVal;
    Tensor deps_plastic = minitensor::dotdot(compliance, dsigma);
    devolps             = minitensor::trace(deps_plastic);
    Tensor dev_plastic  = deps_plastic - (1.0 / 3.0) * devolps * I;
    deqps = std::sqrt(2.0 / 3.0) * minitensor::norm(dev_plastic);

    if (finite_deformation_) {
      // Update Fp via exponential map
      dgdsigma_loc = compute_dgdsigma(sigmaVal, alphaVal, kappaVal);
      Tensor const expA = minitensor::exp(dgamma * dgdsigma_loc);
      Fpnew = expA * Fpn;
    }
  } else {
    // Elastic step
    if (finite_deformation_) {
      Fpnew = Fpn;
    }
  }

  // Store results
  if (finite_deformation_) {
    // Cauchy stress = Kirchhoff stress / J
    ScalarT const Jdet = J_(cell, pt);
    for (int i = 0; i < num_dims_; ++i) {
      for (int j = 0; j < num_dims_; ++j) {
        stress_(cell, pt, i, j)     = sigmaVal(i, j) / Jdet;
        Fp_(cell, pt, i, j)         = Fpnew(i, j);
        backStress_(cell, pt, i, j) = alphaVal(i, j);
      }
    }
  } else {
    for (int i = 0; i < num_dims_; ++i) {
      for (int j = 0; j < num_dims_; ++j) {
        stress_(cell, pt, i, j)     = sigmaVal(i, j);
        backStress_(cell, pt, i, j) = alphaVal(i, j);
      }
    }
  }

  capParameter_(cell, pt)     = kappaVal;
  eqps_(cell, pt)             = eqps_old_(cell, pt) + deqps;
  volPlasticStrain_(cell, pt) = volPlasticStrain_old_(cell, pt) + devolps;
}

//
// Helper functions
//
template <typename EvalT, typename Traits>
typename CapExplicitKernel<EvalT, Traits>::ScalarT
CapExplicitKernel<EvalT, Traits>::compute_f(
    minitensor::Tensor<ScalarT>& sigma,
    minitensor::Tensor<ScalarT>& alpha,
    ScalarT& kappa) const
{
  minitensor::Tensor<ScalarT> xi = sigma - alpha;

  ScalarT I1 = minitensor::trace(xi), p = I1 / 3;

  minitensor::Tensor<ScalarT> s = xi - p * minitensor::identity<ScalarT>(3);

  ScalarT J2 = 0.5 * minitensor::dotdot(s, s);

  ScalarT J3 = minitensor::det(s);

  ScalarT Gamma = 1.0;
  if (psi_ != 0 && J2 != 0)
    Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) +
            (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi_);

  ScalarT Ff_I1 = A_ - C_ * std::exp(D_ * I1) - theta_ * I1;

  ScalarT Ff_kappa = A_ - C_ * std::exp(D_ * kappa) - theta_ * kappa;

  ScalarT X = kappa - R_ * Ff_kappa;

  ScalarT Fc = 1.0;

  if ((kappa - I1) > 0 && ((X - kappa) != 0))
    Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

  return Gamma * Gamma * J2 - Fc * (Ff_I1 - N_) * (Ff_I1 - N_);
}

template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapExplicitKernel<EvalT, Traits>::ScalarT>
CapExplicitKernel<EvalT, Traits>::compute_dfdsigma(
    minitensor::Tensor<ScalarT>& sigma,
    minitensor::Tensor<ScalarT>& alpha,
    ScalarT& kappa) const
{
  minitensor::Tensor<ScalarT> const I(minitensor::eye<ScalarT>(3));

  minitensor::Tensor<ScalarT> xi = sigma - alpha;

  ScalarT I1 = minitensor::trace(xi), p = I1 / 3;

  minitensor::Tensor<ScalarT> s = xi - p * minitensor::identity<ScalarT>(3);

  ScalarT J2 = 0.5 * minitensor::dotdot(s, s);

  ScalarT J3 = minitensor::det(s);

  minitensor::Tensor<ScalarT> dJ3dsigma(3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      dJ3dsigma(i, j) = s(i, j) * s(i, j) - 2 * J2 * I(i, j) / 3;

  ScalarT Ff_I1 = A_ - C_ * std::exp(D_ * I1) - theta_ * I1;

  ScalarT Ff_kappa = A_ - C_ * std::exp(D_ * kappa) - theta_ * kappa;

  ScalarT X = kappa - R_ * Ff_kappa;

  ScalarT Fc = 1.0;

  if ((kappa - I1) > 0) Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

  ScalarT Gamma = 1.0;
  if (psi_ != 0 && J2 != 0)
    Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) +
            (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi_);

  // derivatives
  ScalarT dFfdI1 = -(D_ * C_ * std::exp(D_ * I1) + theta_);

  ScalarT dFcdI1 = 0.0;
  if ((kappa - I1) > 0 && ((X - kappa) != 0))
    dFcdI1 = -2.0 * (I1 - kappa) / (X - kappa) / (X - kappa);

  ScalarT dfdI1 = -(Ff_I1 - N_) * (2 * Fc * dFfdI1 + (Ff_I1 - N_) * dFcdI1);

  ScalarT dGammadJ2 = 0.0;
  if (J2 != 0)
    dGammadJ2 = 9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0 * (1.0 - 1.0 / psi_);

  ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;

  ScalarT dGammadJ3 = 0.0;
  if (J2 != 0)
    dGammadJ3 = -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0 * (1.0 - 1.0 / psi_);

  ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

  return dfdI1 * I + dfdJ2 * s + dfdJ3 * dJ3dsigma;
}

template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapExplicitKernel<EvalT, Traits>::ScalarT>
CapExplicitKernel<EvalT, Traits>::compute_dgdsigma(
    minitensor::Tensor<ScalarT>& sigma,
    minitensor::Tensor<ScalarT>& alpha,
    ScalarT& kappa) const
{
  minitensor::Tensor<ScalarT> const I(minitensor::eye<ScalarT>(3));

  minitensor::Tensor<ScalarT> xi = sigma - alpha;

  ScalarT I1 = minitensor::trace(xi), p = I1 / 3;

  minitensor::Tensor<ScalarT> s = xi - p * minitensor::identity<ScalarT>(3);

  ScalarT J2 = 0.5 * minitensor::dotdot(s, s);

  ScalarT J3 = minitensor::det(s);

  minitensor::Tensor<ScalarT> dJ3dsigma(3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      dJ3dsigma(i, j) = s(i, j) * s(i, j) - 2 * J2 * I(i, j) / 3;

  ScalarT Ff_I1 = A_ - C_ * std::exp(L_ * I1) - phi_ * I1;

  ScalarT Ff_kappa = A_ - C_ * std::exp(L_ * kappa) - phi_ * kappa;

  ScalarT X = kappa - Q_ * Ff_kappa;

  ScalarT Fc = 1.0;

  if ((kappa - I1) > 0) Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

  ScalarT Gamma = 1.0;
  if (psi_ != 0 && J2 != 0)
    Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) +
            (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi_);

  // derivatives
  ScalarT dFfdI1 = -(L_ * C_ * std::exp(L_ * I1) + phi_);

  ScalarT dFcdI1 = 0.0;
  if ((kappa - I1) > 0 && ((X - kappa) != 0))
    dFcdI1 = -2.0 * (I1 - kappa) / (X - kappa) / (X - kappa);

  ScalarT dfdI1 = -(Ff_I1 - N_) * (2 * Fc * dFfdI1 + (Ff_I1 - N_) * dFcdI1);

  ScalarT dGammadJ2 = 0.0;
  if (J2 != 0)
    dGammadJ2 = 9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0 * (1.0 - 1.0 / psi_);

  ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;

  ScalarT dGammadJ3 = 0.0;
  if (J2 != 0)
    dGammadJ3 = -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0 * (1.0 - 1.0 / psi_);

  ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

  return dfdI1 * I + dfdJ2 * s + dfdJ3 * dJ3dsigma;
}

template <typename EvalT, typename Traits>
typename CapExplicitKernel<EvalT, Traits>::ScalarT
CapExplicitKernel<EvalT, Traits>::compute_dfdkappa(
    minitensor::Tensor<ScalarT>& sigma,
    minitensor::Tensor<ScalarT>& alpha,
    ScalarT& kappa) const
{
  minitensor::Tensor<ScalarT> xi = sigma - alpha;

  ScalarT I1 = minitensor::trace(xi);

  ScalarT Ff_I1 = A_ - C_ * std::exp(D_ * I1) - theta_ * I1;

  ScalarT Ff_kappa = A_ - C_ * std::exp(D_ * kappa) - theta_ * kappa;

  ScalarT X = kappa - R_ * Ff_kappa;

  ScalarT dFcdkappa = 0.0;

  if ((kappa - I1) > 0 && ((X - kappa) != 0)) {
    dFcdkappa = 2 * (I1 - kappa) *
        ((X - kappa) + R_ * (I1 - kappa) * (theta_ + D_ * C_ * std::exp(D_ * kappa))) /
        (X - kappa) / (X - kappa) / (X - kappa);
  }

  return -dFcdkappa * (Ff_I1 - N_) * (Ff_I1 - N_);
}

template <typename EvalT, typename Traits>
typename CapExplicitKernel<EvalT, Traits>::ScalarT
CapExplicitKernel<EvalT, Traits>::compute_Galpha(ScalarT& J2_alpha) const
{
  if (N_ != 0)
    return 1.0 - std::pow(J2_alpha, 0.5) / N_;
  else
    return 0.0;
}

template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapExplicitKernel<EvalT, Traits>::ScalarT>
CapExplicitKernel<EvalT, Traits>::compute_halpha(
    minitensor::Tensor<ScalarT>& dgdsigma,
    ScalarT& J2_alpha) const
{
  minitensor::Tensor<ScalarT> const I(minitensor::eye<ScalarT>(3));

  ScalarT Galpha = compute_Galpha(J2_alpha);

  ScalarT I1 = minitensor::trace(dgdsigma), p = I1 / 3;

  minitensor::Tensor<ScalarT> s_loc = dgdsigma - p * I;

  minitensor::Tensor<ScalarT> result(3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      result(i, j) = calpha_ * Galpha * s_loc(i, j);

  return result;
}

template <typename EvalT, typename Traits>
typename CapExplicitKernel<EvalT, Traits>::ScalarT
CapExplicitKernel<EvalT, Traits>::compute_dedkappa(ScalarT& kappa) const
{
  ScalarT Ff_kappa0 = A_ - C_ * std::exp(L_ * kappa0_) - phi_ * kappa0_;

  ScalarT X0 = kappa0_ - Q_ * Ff_kappa0;

  ScalarT Ff_kappa = A_ - C_ * std::exp(L_ * kappa) - phi_ * kappa;

  ScalarT X = kappa - Q_ * Ff_kappa;

  ScalarT dedX = (D1_ - 2 * D2_ * (X - X0)) *
      std::exp((D1_ - D2_ * (X - X0)) * (X - X0)) * W_;

  ScalarT dXdkappa = 1 + Q_ * C_ * L_ * std::exp(L_ * kappa) + Q_ * phi_;

  return dedX * dXdkappa;
}

}  // namespace LCM
