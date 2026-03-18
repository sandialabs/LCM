// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include <MiniTensor.h>

#include "Albany_Utils.hpp"
#include "CapImplicit.hpp"
#include "LocalNonlinearSolver.hpp"

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

  if (compute_tangent_) {
    std::string const tangent_string = field_name_map_["Material Tangent"];
    setEvaluatedField(tangent_string, dl->qp_tensor4);
  }

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

  if (compute_tangent_) {
    std::string tangent_string = field_name_map_["Material Tangent"];
    tangent_ = *eval_fields[tangent_string];
  }

  // get old state variables
  stress_old_           = (*workset.stateArrayPtr)[cauchy_string + "_old"];
  backStress_old_       = (*workset.stateArrayPtr)[backStress_string + "_old"];
  capParameter_old_     = (*workset.stateArrayPtr)[capParameter_string + "_old"];
  eqps_old_             = (*workset.stateArrayPtr)[eqps_string + "_old"];
  volPlasticStrain_old_ = (*workset.stateArrayPtr)[volPlasticStrain_string + "_old"];
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
CapImplicitKernel<EvalT, Traits>::operator()(int cell, int pt) const
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
  Tensor4 const Celastic =
      lame * minitensor::identity_3<ScalarT>(num_dims_) +
      mu * (minitensor::identity_1<ScalarT>(num_dims_) + minitensor::identity_2<ScalarT>(num_dims_));

  // elastic compliance tangent matrix
  Tensor4 const compliance =
      (1.0 / bulkModulus / 9.0) * minitensor::identity_3<ScalarT>(num_dims_) +
      (1.0 / mu / 2.0) * (0.5 * (minitensor::identity_1<ScalarT>(num_dims_) + minitensor::identity_2<ScalarT>(num_dims_)) -
      (1.0 / 3.0) * minitensor::identity_3<ScalarT>(num_dims_));

  // Load old back stress and cap parameter
  Tensor alphaVal(num_dims_);
  for (int i = 0; i < num_dims_; ++i)
    for (int j = 0; j < num_dims_; ++j)
      alphaVal(i, j) = backStress_old_(cell, pt, i, j);

  ScalarT kappaVal = capParameter_old_(cell, pt);

  // Trial stress computation - depends on kinematics
  Tensor sigmaVal(num_dims_);
  Tensor sigmaTr(num_dims_);
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
  } else {
    // Small strain path
    Tensor sigmaN(num_dims_);
    for (int i = 0; i < num_dims_; ++i) {
      for (int j = 0; j < num_dims_; ++j) {
        depsilon(i, j) = strain_(cell, pt, i, j) - strain_old_(cell, pt, i, j);
        sigmaN(i, j)   = stress_old_(cell, pt, i, j);
      }
    }

    sigmaVal = sigmaN + minitensor::dotdot(Celastic, depsilon);
    sigmaTr  = sigmaVal;
  }

  ScalarT dgammaVal = 0.0;

  // define plastic strain increment, its two invariants: dev, and vol
  Tensor  deps_plastic(num_dims_, minitensor::Filler::ZEROS);
  ScalarT deqps(0.0), devolps(0.0);

  std::vector<ScalarT> XXVal(13);

  // check yielding
  ScalarT f = compute_f(sigmaVal, alphaVal, kappaVal);
  XXVal     = initialize(sigmaVal, alphaVal, kappaVal, dgammaVal);

  // local Newton loop
  if (f > 1.e-11) {  // plastic yielding

    ScalarT normR, normR0, conv;
    bool    kappa_flag = false;
    bool    converged  = false;
    int     iter       = 0;

    std::vector<ScalarT>                R(13);
    std::vector<ScalarT>                dRdX(13 * 13);
    LocalNonlinearSolver<EvalT, Traits> solver;

    while (!converged) {
      // assemble residual vector and local Jacobian
      // For FD, pass sigmaTr (trial Kirchhoff stress); for small strain, pass sigmaVal (trial stress)
      compute_ResidJacobian(XXVal, R, dRdX, sigmaTr, alphaVal, kappaVal, Celastic, kappa_flag);

      normR = 0.0;
      for (int i = 0; i < 13; i++) normR += R[i] * R[i];

      normR = std::sqrt(normR);

      if (iter == 0) normR0 = normR;
      if (normR0 != 0)
        conv = normR / normR0;
      else
        conv = normR0;

      if (conv < 1.e-11 || normR < 1.e-11) break;

      if (iter > 20) break;

      std::vector<ScalarT> XXValK = XXVal;
      solver.solve(dRdX, XXValK, R);

      // put restrictions on kappa: only allows monotonic decreasing (cap
      // hardening)
      if (XXValK[11] > XXVal[11]) {
        kappa_flag = true;
      } else {
        XXVal      = XXValK;
        kappa_flag = false;
      }

      iter++;
    }  // end local NR

    // compute sensitivity information, and pack back to X.
    solver.computeFadInfo(dRdX, XXVal, R);

  }  // end of plasticity

  // update
  sigmaVal(0, 0) = XXVal[0];
  sigmaVal(0, 1) = XXVal[5];
  sigmaVal(0, 2) = XXVal[4];
  sigmaVal(1, 0) = XXVal[5];
  sigmaVal(1, 1) = XXVal[1];
  sigmaVal(1, 2) = XXVal[3];
  sigmaVal(2, 0) = XXVal[4];
  sigmaVal(2, 1) = XXVal[3];
  sigmaVal(2, 2) = XXVal[2];

  alphaVal(0, 0) = XXVal[6];
  alphaVal(0, 1) = XXVal[10];
  alphaVal(0, 2) = XXVal[9];
  alphaVal(1, 0) = XXVal[10];
  alphaVal(1, 1) = XXVal[7];
  alphaVal(1, 2) = XXVal[8];
  alphaVal(2, 0) = XXVal[9];
  alphaVal(2, 1) = XXVal[8];
  alphaVal(2, 2) = -XXVal[6] - XXVal[7];

  kappaVal  = XXVal[11];
  dgammaVal = XXVal[12];

  // compute plastic strain increment deps_plastic = compliance ( sigma_tr - sigma_(n+1))
  Tensor dsigma    = sigmaTr - sigmaVal;
  deps_plastic     = minitensor::dotdot(compliance, dsigma);
  devolps          = minitensor::trace(deps_plastic);
  Tensor dev_plastic = deps_plastic - (1.0 / 3.0) * devolps * I;
  deqps = std::sqrt(2.0 / 3.0) * minitensor::norm(dev_plastic);

  if (finite_deformation_) {
    // Update Fp via exponential map
    if (dgammaVal > 0) {
      Tensor dgdsigma_loc = compute_dgdsigma(XXVal);
      Tensor const expA   = minitensor::exp(dgammaVal * dgdsigma_loc);
      Fpnew = expA * Fpn;
    } else {
      Fpnew = Fpn;
    }

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

  if (compute_tangent_) {
    Tensor4 Cep = compute_Cep(const_cast<Tensor4&>(Celastic), sigmaVal, alphaVal, kappaVal, dgammaVal);
    for (int i(0); i < num_dims_; ++i) {
      for (int j(0); j < num_dims_; ++j) {
        for (int k(0); k < num_dims_; ++k) {
          for (int l(0); l < num_dims_; ++l) {
            tangent_(cell, pt, i, j, k, l) = Cep(i, j, k, l);
          }
        }
      }
    }
  }
}

//
// Helper functions
//

//------------------------------ yield function ------------------------------//
template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitKernel<EvalT, Traits>::compute_f(
    minitensor::Tensor<T>& sigma,
    minitensor::Tensor<T>& alpha,
    T& kappa) const
{
  minitensor::Tensor<T> xi = sigma - alpha;

  T I1 = minitensor::trace(xi), p = I1 / 3.;

  minitensor::Tensor<T> s = xi - p * minitensor::identity<T>(3);

  T J2 = 0.5 * minitensor::dotdot(s, s);

  T J3 = minitensor::det(s);

  T Gamma = 1.0;

  if (psi_ != 0 && J2 != 0)
    Gamma = 0.5 * (1. - 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5) +
            (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)) / psi_);

  T Ff_I1 = A_ - C_ * std::exp(D_ * I1) - theta_ * I1;

  T Ff_kappa = A_ - C_ * std::exp(D_ * kappa) - theta_ * kappa;

  T X = kappa - R_ * Ff_kappa;

  T Fc = 1.0;

  if ((kappa - I1) > 0 && ((X - kappa) != 0))
    Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

  return Gamma * Gamma * J2 - Fc * (Ff_I1 - N_) * (Ff_I1 - N_);
}

//---------------------------- plastic potential -----------------------------//
template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitKernel<EvalT, Traits>::compute_g(
    minitensor::Tensor<T>& sigma,
    minitensor::Tensor<T>& alpha,
    T& kappa) const
{
  minitensor::Tensor<T> xi = sigma - alpha;

  T I1 = minitensor::trace(xi);

  T p = I1 / 3.;

  minitensor::Tensor<T> s = xi - p * minitensor::identity<T>(3);

  T J2 = 0.5 * minitensor::dotdot(s, s);

  T J3 = minitensor::det(s);

  T Gamma = 1.0;

  if (psi_ != 0 && J2 != 0)
    Gamma = 0.5 * (1. - 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5) +
            (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)) / psi_);

  T Ff_I1 = A_ - C_ * std::exp(L_ * I1) - phi_ * I1;

  T Ff_kappa = A_ - C_ * std::exp(L_ * kappa) - phi_ * kappa;

  T X = kappa - Q_ * Ff_kappa;

  T Fc = 1.0;

  if ((kappa - I1) > 0 && ((X - kappa) != 0))
    Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

  return Gamma * Gamma * J2 - Fc * (Ff_I1 - N_) * (Ff_I1 - N_);
}

//------------------------ unknown variable value list ------------------------//
template <typename EvalT, typename Traits>
std::vector<typename CapImplicitKernel<EvalT, Traits>::ScalarT>
CapImplicitKernel<EvalT, Traits>::initialize(
    minitensor::Tensor<ScalarT>& sigmaVal,
    minitensor::Tensor<ScalarT>& alphaVal,
    ScalarT& kappaVal,
    ScalarT& dgammaVal) const
{
  std::vector<ScalarT> XX(13);

  XX[0]  = sigmaVal(0, 0);
  XX[1]  = sigmaVal(1, 1);
  XX[2]  = sigmaVal(2, 2);
  XX[3]  = sigmaVal(1, 2);
  XX[4]  = sigmaVal(0, 2);
  XX[5]  = sigmaVal(0, 1);
  XX[6]  = alphaVal(0, 0);
  XX[7]  = alphaVal(1, 1);
  XX[8]  = alphaVal(1, 2);
  XX[9]  = alphaVal(0, 2);
  XX[10] = alphaVal(0, 1);
  XX[11] = kappaVal;
  XX[12] = dgammaVal;

  return XX;
}

//----------------------- local iteration jacobian ---------------------------//
template <typename EvalT, typename Traits>
void
CapImplicitKernel<EvalT, Traits>::compute_ResidJacobian(
    std::vector<ScalarT> const&         XXVal,
    std::vector<ScalarT>&               R,
    std::vector<ScalarT>&               dRdX,
    const minitensor::Tensor<ScalarT>&  sigmaVal,
    const minitensor::Tensor<ScalarT>&  alphaVal,
    ScalarT const&                      kappaVal,
    minitensor::Tensor4<ScalarT> const& Celastic,
    bool                                kappa_flag) const
{
  std::vector<DFadType> Rfad(13);
  std::vector<DFadType> XX(13);
  std::vector<ScalarT>  XXtmp(13);

  // initialize DFadType local unknown vector
  for (int i = 0; i < 13; ++i) {
    XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XXVal[i]);
    XX[i]    = DFadType(13, i, XXtmp[i]);
  }

  minitensor::Tensor<DFadType> sigma(3), alpha(3);

  sigma(0, 0) = XX[0];
  sigma(0, 1) = XX[5];
  sigma(0, 2) = XX[4];
  sigma(1, 0) = XX[5];
  sigma(1, 1) = XX[1];
  sigma(1, 2) = XX[3];
  sigma(2, 0) = XX[4];
  sigma(2, 1) = XX[3];
  sigma(2, 2) = XX[2];

  alpha(0, 0) = XX[6];
  alpha(0, 1) = XX[10];
  alpha(0, 2) = XX[9];
  alpha(1, 0) = XX[10];
  alpha(1, 1) = XX[7];
  alpha(1, 2) = XX[8];
  alpha(2, 0) = XX[9];
  alpha(2, 1) = XX[8];
  alpha(2, 2) = -XX[6] - XX[7];

  DFadType kappa = XX[11];

  DFadType dgamma = XX[12];

  DFadType f = compute_f(sigma, alpha, kappa);

  minitensor::Tensor<DFadType> dgdsigma = compute_dgdsigma(XX);

  DFadType J2_alpha = 0.5 * minitensor::dotdot(alpha, alpha);

  minitensor::Tensor<DFadType> halpha = compute_halpha(dgdsigma, J2_alpha);

  DFadType I1_dgdsigma = minitensor::trace(dgdsigma);

  DFadType dedkappa = compute_dedkappa(kappa);

  DFadType hkappa = compute_hkappa(I1_dgdsigma, dedkappa);

  DFadType t;

  t = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      t = t + Celastic(0, 0, i, j) * dgdsigma(i, j);
    }
  }
  Rfad[0] = dgamma * t + sigma(0, 0) - sigmaVal(0, 0);

  t = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      t = t + Celastic(1, 1, i, j) * dgdsigma(i, j);
    }
  }
  Rfad[1] = dgamma * t + sigma(1, 1) - sigmaVal(1, 1);

  t = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      t = t + Celastic(2, 2, i, j) * dgdsigma(i, j);
    }
  }
  Rfad[2] = dgamma * t + sigma(2, 2) - sigmaVal(2, 2);

  t = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      t = t + Celastic(1, 2, i, j) * dgdsigma(i, j);
    }
  }
  Rfad[3] = dgamma * t + sigma(1, 2) - sigmaVal(1, 2);

  t = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      t = t + Celastic(0, 2, i, j) * dgdsigma(i, j);
    }
  }
  Rfad[4] = dgamma * t + sigma(0, 2) - sigmaVal(0, 2);

  t = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      t = t + Celastic(0, 1, i, j) * dgdsigma(i, j);
    }
  }
  Rfad[5] = dgamma * t + sigma(0, 1) - sigmaVal(0, 1);

  Rfad[6] = dgamma * halpha(0, 0) - alpha(0, 0) + alphaVal(0, 0);

  Rfad[7] = dgamma * halpha(1, 1) - alpha(1, 1) + alphaVal(1, 1);

  Rfad[8] = dgamma * halpha(1, 2) - alpha(1, 2) + alphaVal(1, 2);

  Rfad[9] = dgamma * halpha(0, 2) - alpha(0, 2) + alphaVal(0, 2);

  Rfad[10] = dgamma * halpha(0, 1) - alpha(0, 1) + alphaVal(0, 1);

  if (kappa_flag == false)
    Rfad[11] = dgamma * hkappa - kappa + kappaVal;
  else
    Rfad[11] = 0;

  Rfad[12] = f;

  // get ScalarT Residual
  for (int i = 0; i < 13; i++) R[i] = Rfad[i].val();

  // get Jacobian
  for (int i = 0; i < 13; i++)
    for (int j = 0; j < 13; j++) dRdX[i + 13 * j] = Rfad[i].dx(j);

  if (kappa_flag == true) {
    for (int j = 0; j < 13; j++) dRdX[11 + 13 * j] = 0.0;

    dRdX[11 + 13 * 11] = 1.0;
  }
}

//----------------------------- derivative -----------------------------------//
template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapImplicitKernel<EvalT, Traits>::ScalarT>
CapImplicitKernel<EvalT, Traits>::compute_dfdsigma(std::vector<ScalarT> const& XX) const
{
  std::vector<DFadType> XXFad(13);
  std::vector<ScalarT>  XXtmp(13);

  for (int i = 0; i < 13; ++i) {
    XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XX[i]);
    XXFad[i] = DFadType(13, i, XXtmp[i]);
  }

  minitensor::Tensor<DFadType> sigma(3), alpha(3);

  sigma(0, 0) = XXFad[0];
  sigma(0, 1) = XXFad[5];
  sigma(0, 2) = XXFad[4];
  sigma(1, 0) = XXFad[5];
  sigma(1, 1) = XXFad[1];
  sigma(1, 2) = XXFad[3];
  sigma(2, 0) = XXFad[4];
  sigma(2, 1) = XXFad[3];
  sigma(2, 2) = XXFad[2];

  alpha(0, 0) = XXFad[6];
  alpha(0, 1) = XXFad[10];
  alpha(0, 2) = XXFad[9];
  alpha(1, 0) = XXFad[10];
  alpha(1, 1) = XXFad[7];
  alpha(1, 2) = XXFad[8];
  alpha(2, 0) = XXFad[9];
  alpha(2, 1) = XXFad[8];
  alpha(2, 2) = -XXFad[6] - XXFad[7];

  DFadType kappa = XXFad[11];

  DFadType f = compute_f(sigma, alpha, kappa);

  minitensor::Tensor<ScalarT> dfdsigma(3);

  dfdsigma(0, 0) = f.dx(0);
  dfdsigma(0, 1) = f.dx(5);
  dfdsigma(0, 2) = f.dx(4);
  dfdsigma(1, 0) = f.dx(5);
  dfdsigma(1, 1) = f.dx(1);
  dfdsigma(1, 2) = f.dx(3);
  dfdsigma(2, 0) = f.dx(4);
  dfdsigma(2, 1) = f.dx(3);
  dfdsigma(2, 2) = f.dx(2);

  return dfdsigma;
}

template <typename EvalT, typename Traits>
typename CapImplicitKernel<EvalT, Traits>::ScalarT
CapImplicitKernel<EvalT, Traits>::compute_dfdkappa(std::vector<ScalarT> const& XX) const
{
  std::vector<DFadType> XXFad(13);
  std::vector<ScalarT>  XXtmp(13);

  for (int i = 0; i < 13; ++i) {
    XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XX[i]);
    XXFad[i] = DFadType(13, i, XXtmp[i]);
  }

  minitensor::Tensor<DFadType> sigma(3), alpha(3);

  sigma(0, 0) = XXFad[0];
  sigma(0, 1) = XXFad[5];
  sigma(0, 2) = XXFad[4];
  sigma(1, 0) = XXFad[5];
  sigma(1, 1) = XXFad[1];
  sigma(1, 2) = XXFad[3];
  sigma(2, 0) = XXFad[4];
  sigma(2, 1) = XXFad[3];
  sigma(2, 2) = XXFad[2];

  alpha(0, 0) = XXFad[6];
  alpha(0, 1) = XXFad[10];
  alpha(0, 2) = XXFad[9];
  alpha(1, 0) = XXFad[10];
  alpha(1, 1) = XXFad[7];
  alpha(1, 2) = XXFad[8];
  alpha(2, 0) = XXFad[9];
  alpha(2, 1) = XXFad[8];
  alpha(2, 2) = -XXFad[6] - XXFad[7];

  DFadType kappa = XXFad[11];

  DFadType f = compute_f(sigma, alpha, kappa);

  ScalarT dfdkappa;

  dfdkappa = f.dx(11);

  return dfdkappa;
}

template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapImplicitKernel<EvalT, Traits>::ScalarT>
CapImplicitKernel<EvalT, Traits>::compute_dgdsigma(std::vector<ScalarT> const& XX) const
{
  std::vector<DFadType> XXFad(13);
  std::vector<ScalarT>  XXtmp(13);

  for (int i = 0; i < 13; ++i) {
    XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XX[i]);
    XXFad[i] = DFadType(13, i, XXtmp[i]);
  }

  minitensor::Tensor<DFadType> sigma(3), alpha(3);

  sigma(0, 0) = XXFad[0];
  sigma(0, 1) = XXFad[5];
  sigma(0, 2) = XXFad[4];
  sigma(1, 0) = XXFad[5];
  sigma(1, 1) = XXFad[1];
  sigma(1, 2) = XXFad[3];
  sigma(2, 0) = XXFad[4];
  sigma(2, 1) = XXFad[3];
  sigma(2, 2) = XXFad[2];

  alpha(0, 0) = XXFad[6];
  alpha(0, 1) = XXFad[10];
  alpha(0, 2) = XXFad[9];
  alpha(1, 0) = XXFad[10];
  alpha(1, 1) = XXFad[7];
  alpha(1, 2) = XXFad[8];
  alpha(2, 0) = XXFad[9];
  alpha(2, 1) = XXFad[8];
  alpha(2, 2) = -XXFad[6] - XXFad[7];

  DFadType kappa = XXFad[11];

  DFadType g = compute_g(sigma, alpha, kappa);

  minitensor::Tensor<ScalarT> dgdsigma(3);

  dgdsigma(0, 0) = g.dx(0);
  dgdsigma(0, 1) = g.dx(5);
  dgdsigma(0, 2) = g.dx(4);
  dgdsigma(1, 0) = g.dx(5);
  dgdsigma(1, 1) = g.dx(1);
  dgdsigma(1, 2) = g.dx(3);
  dgdsigma(2, 0) = g.dx(4);
  dgdsigma(2, 1) = g.dx(3);
  dgdsigma(2, 2) = g.dx(2);

  return dgdsigma;
}

template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapImplicitKernel<EvalT, Traits>::DFadType>
CapImplicitKernel<EvalT, Traits>::compute_dgdsigma(std::vector<DFadType> const& XX) const
{
  std::vector<D2FadType> D2XX(13);
  std::vector<DFadType>  XXFadtmp(13);
  std::vector<ScalarT>   XXtmp(13);

  for (int i = 0; i < 13; ++i) {
    XXtmp[i]    = Sacado::ScalarValue<ScalarT>::eval(XX[i].val());
    XXFadtmp[i] = DFadType(13, i, XXtmp[i]);
    D2XX[i]     = D2FadType(13, i, XXFadtmp[i]);
  }

  minitensor::Tensor<D2FadType> sigma(3), alpha(3);

  sigma(0, 0) = D2XX[0];
  sigma(0, 1) = D2XX[5];
  sigma(0, 2) = D2XX[4];
  sigma(1, 0) = D2XX[5];
  sigma(1, 1) = D2XX[1];
  sigma(1, 2) = D2XX[3];
  sigma(2, 0) = D2XX[4];
  sigma(2, 1) = D2XX[3];
  sigma(2, 2) = D2XX[2];

  alpha(0, 0) = D2XX[6];
  alpha(0, 1) = D2XX[10];
  alpha(0, 2) = D2XX[9];
  alpha(1, 0) = D2XX[10];
  alpha(1, 1) = D2XX[7];
  alpha(1, 2) = D2XX[8];
  alpha(2, 0) = D2XX[9];
  alpha(2, 1) = D2XX[8];
  alpha(2, 2) = -D2XX[6] - D2XX[7];

  D2FadType kappa = D2XX[11];

  D2FadType g = compute_g(sigma, alpha, kappa);

  minitensor::Tensor<DFadType> dgdsigma(3);

  dgdsigma(0, 0) = g.dx(0);
  dgdsigma(0, 1) = g.dx(5);
  dgdsigma(0, 2) = g.dx(4);
  dgdsigma(1, 0) = g.dx(5);
  dgdsigma(1, 1) = g.dx(1);
  dgdsigma(1, 2) = g.dx(3);
  dgdsigma(2, 0) = g.dx(4);
  dgdsigma(2, 1) = g.dx(3);
  dgdsigma(2, 2) = g.dx(2);

  return dgdsigma;
}

//--------------------------- hardening functions ----------------------------//
template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitKernel<EvalT, Traits>::compute_Galpha(T J2_alpha) const
{
  if (N_ != 0)
    return 1.0 - pow(J2_alpha, 0.5) / N_;
  else
    return 0.0;
}

template <typename EvalT, typename Traits>
template <typename T>
minitensor::Tensor<T>
CapImplicitKernel<EvalT, Traits>::compute_halpha(
    minitensor::Tensor<T> const& dgdsigma,
    T const J2_alpha) const
{
  T Galpha = compute_Galpha(J2_alpha);

  T I1 = minitensor::trace(dgdsigma), p = I1 / 3.0;

  minitensor::Tensor<T> s = dgdsigma - p * minitensor::identity<T>(3);

  minitensor::Tensor<T> halpha(3);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      halpha(i, j) = calpha_ * Galpha * s(i, j);
    }
  }

  return halpha;
}

template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitKernel<EvalT, Traits>::compute_dedkappa(T const kappa) const
{
  T Ff_kappa0 = A_ - C_ * std::exp(L_ * kappa0_) - phi_ * kappa0_;

  T X0 = kappa0_ - Q_ * Ff_kappa0;

  T Ff_kappa = A_ - C_ * std::exp(L_ * kappa) - phi_ * kappa;

  T X = kappa - Q_ * Ff_kappa;

  T dedX = (D1_ - 2. * D2_ * (X - X0)) *
      std::exp((D1_ - D2_ * (X - X0)) * (X - X0)) * W_;

  T dXdkappa = 1. + Q_ * C_ * L_ * exp(L_ * kappa) + Q_ * phi_;

  return dedX * dXdkappa;
}

template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitKernel<EvalT, Traits>::compute_hkappa(T const I1_dgdsigma, T const dedkappa) const
{
  if (dedkappa != 0)
    return I1_dgdsigma / dedkappa;
  else
    return 0;
}

//------------------------ elasto-plastic tangent modulus --------------------//
template <typename EvalT, typename Traits>
minitensor::Tensor4<typename CapImplicitKernel<EvalT, Traits>::ScalarT>
CapImplicitKernel<EvalT, Traits>::compute_Cep(
    minitensor::Tensor4<ScalarT>& Celastic,
    minitensor::Tensor<ScalarT>&  sigma,
    minitensor::Tensor<ScalarT>&  alpha,
    ScalarT&                      kappa,
    ScalarT&                      dgamma) const
{
  if (dgamma == 0) return Celastic;

  // define variable
  minitensor::Tensor4<ScalarT> Cep(num_dims_);

  std::vector<ScalarT> XX(13);

  minitensor::Tensor<ScalarT> dfdsigma;
  minitensor::Tensor<ScalarT> dfdalpha;
  minitensor::Tensor<ScalarT> dgdsigma;
  minitensor::Tensor<ScalarT> halpha_loc;
  ScalarT                     hkappa_loc;
  ScalarT                     dfdkappa;
  ScalarT                     chi;

  // compute variable
  XX = initialize(sigma, alpha, kappa, dgamma);

  dfdsigma = compute_dfdsigma(XX);
  dfdalpha = dfdsigma * (-1.0);
  dfdkappa = compute_dfdkappa(XX);
  dgdsigma = compute_dgdsigma(XX);

  ScalarT J2_alpha = 0.5 * minitensor::dotdot(alpha, alpha);
  halpha_loc       = compute_halpha(dgdsigma, J2_alpha);

  ScalarT I1_dgdsigma = minitensor::trace(dgdsigma);
  ScalarT dedkappa    = compute_dedkappa(kappa);
  hkappa_loc          = compute_hkappa(I1_dgdsigma, dedkappa);

  chi = minitensor::dotdot(minitensor::dotdot(dfdsigma, Celastic), dgdsigma) -
        minitensor::dotdot(dfdalpha, halpha_loc) - dfdkappa * hkappa_loc;

  if (chi == 0) {
    chi = 1e-16;
  }

  // compute tangent
  Cep = Celastic - 1.0 / chi * minitensor::dotdot(Celastic,
      minitensor::tensor(dgdsigma, minitensor::dotdot(dfdsigma, Celastic)));

  return Cep;
}

}  // namespace LCM
