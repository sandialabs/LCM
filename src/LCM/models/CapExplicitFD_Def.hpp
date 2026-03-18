// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include <MiniTensor.h>

#include <Intrepid2_FunctionSpaceTools.hpp>
#include <Phalanx_DataLayout.hpp>
#include <typeinfo>

namespace LCM {

template <typename EvalT, typename Traits>
CapExplicitFD<EvalT, Traits>::CapExplicitFD(Teuchos::ParameterList* p, const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
      A(p->get<RealType>("A")),
      D(p->get<RealType>("D")),
      C(p->get<RealType>("C")),
      theta(p->get<RealType>("theta")),
      R(p->get<RealType>("R")),
      kappa0(p->get<RealType>("kappa0")),
      W(p->get<RealType>("W")),
      D1(p->get<RealType>("D1")),
      D2(p->get<RealType>("D2")),
      calpha(p->get<RealType>("calpha")),
      psi(p->get<RealType>("psi")),
      N(p->get<RealType>("N")),
      L(p->get<RealType>("L")),
      phi(p->get<RealType>("phi")),
      Q(p->get<RealType>("Q"))
{
  std::string F_string              = (*field_name_map_)["F"];
  std::string J_string              = (*field_name_map_)["J"];
  std::string cauchy_string         = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string             = (*field_name_map_)["Fp"];
  std::string backStress_string     = (*field_name_map_)["Back_Stress"];
  std::string capParameter_string   = (*field_name_map_)["Cap_Parameter"];
  std::string eqps_string           = (*field_name_map_)["eqps"];
  std::string volPlasticStrain_string = (*field_name_map_)["volPlastic_Strain"];

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair(F_string, dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair(J_string, dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Poissons Ratio", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Elastic Modulus", dl->qp_scalar));

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(Fp_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(backStress_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(capParameter_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(eqps_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(volPlasticStrain_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Material Tangent", dl->qp_tensor4));

  // define the state variables
  // Fp
  this->num_state_variables_++;
  this->state_var_names_.push_back(Fp_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("identity");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  // stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(cauchy_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  // backStress
  this->num_state_variables_++;
  this->state_var_names_.push_back(backStress_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  // capParameter
  this->num_state_variables_++;
  this->state_var_names_.push_back(capParameter_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(kappa0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  // eqps
  this->num_state_variables_++;
  this->state_var_names_.push_back(eqps_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  // volPlasticStrain
  this->num_state_variables_++;
  this->state_var_names_.push_back(volPlasticStrain_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);

  // initialize tensor
  I             = minitensor::eye<ScalarT>(num_dims_);
  id1           = minitensor::identity_1<ScalarT>(num_dims_);
  id2           = minitensor::identity_2<ScalarT>(num_dims_);
  id3           = minitensor::identity_3<ScalarT>(num_dims_);
  Celastic      = minitensor::Tensor4<ScalarT>(num_dims_);
  compliance    = minitensor::Tensor4<ScalarT>(num_dims_);
  sigmaN        = minitensor::Tensor<ScalarT>(num_dims_);
  sigmaVal      = minitensor::Tensor<ScalarT>(num_dims_);
  alphaVal      = minitensor::Tensor<ScalarT>(num_dims_);
  deps_plastic  = minitensor::Tensor<ScalarT>(num_dims_);
  sigmaTr       = minitensor::Tensor<ScalarT>(num_dims_);
  alphaTr       = minitensor::Tensor<ScalarT>(num_dims_);
  dfdsigma      = minitensor::Tensor<ScalarT>(num_dims_);
  dgdsigma      = minitensor::Tensor<ScalarT>(num_dims_);
  dfdalpha      = minitensor::Tensor<ScalarT>(num_dims_);
  halpha        = minitensor::Tensor<ScalarT>(num_dims_);
  dfdotCe       = minitensor::Tensor<ScalarT>(num_dims_);
  sigmaK        = minitensor::Tensor<ScalarT>(num_dims_);
  alphaK        = minitensor::Tensor<ScalarT>(num_dims_);
  dsigma        = minitensor::Tensor<ScalarT>(num_dims_);
  dev_plastic   = minitensor::Tensor<ScalarT>(num_dims_);
  xi            = minitensor::Tensor<ScalarT>(num_dims_);
  sN            = minitensor::Tensor<ScalarT>(num_dims_);
  s             = minitensor::Tensor<ScalarT>(num_dims_);
  dJ3dsigma     = minitensor::Tensor<ScalarT>(num_dims_);
  eps_dev       = minitensor::Tensor<ScalarT>(num_dims_);
}

template <typename EvalT, typename Traits>
void
CapExplicitFD<EvalT, Traits>::computeState(typename Traits::EvalData workset, DepFieldMap dep_fields, FieldMap eval_fields)
{
  std::string F_string              = (*field_name_map_)["F"];
  std::string J_string              = (*field_name_map_)["J"];
  std::string cauchy_string         = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string             = (*field_name_map_)["Fp"];
  std::string backStress_string     = (*field_name_map_)["Back_Stress"];
  std::string capParameter_string   = (*field_name_map_)["Cap_Parameter"];
  std::string eqps_string           = (*field_name_map_)["eqps"];
  std::string volPlasticStrain_string = (*field_name_map_)["volPlastic_Strain"];

  // extract dependent MDFields
  auto def_grad        = *dep_fields[F_string];
  auto jac_det         = *dep_fields[J_string];
  auto poissons_ratio  = *dep_fields["Poissons Ratio"];
  auto elastic_modulus = *dep_fields["Elastic Modulus"];

  // extract evaluated MDFields
  auto stress           = *eval_fields[cauchy_string];
  auto Fp               = *eval_fields[Fp_string];
  auto backStress       = *eval_fields[backStress_string];
  auto capParameter     = *eval_fields[capParameter_string];
  auto eqps             = *eval_fields[eqps_string];
  auto volPlasticStrain = *eval_fields[volPlasticStrain_string];
  auto tangent          = *eval_fields["Material Tangent"];

  // get State Variables
  Albany::MDArray Fpold               = (*workset.stateArrayPtr)[Fp_string + "_old"];
  Albany::MDArray stressold           = (*workset.stateArrayPtr)[cauchy_string + "_old"];
  Albany::MDArray backStressold       = (*workset.stateArrayPtr)[backStress_string + "_old"];
  Albany::MDArray capParameterold     = (*workset.stateArrayPtr)[capParameter_string + "_old"];
  Albany::MDArray eqpsold             = (*workset.stateArrayPtr)[eqps_string + "_old"];
  Albany::MDArray volPlasticStrainold = (*workset.stateArrayPtr)[volPlasticStrain_string + "_old"];

  ScalarT lame, mu, bulkModulus;
  minitensor::Tensor<ScalarT> F(num_dims_), Fpn(num_dims_), Fpinv(num_dims_), Cpinv(num_dims_), be(num_dims_), logbe(num_dims_), eps_e(num_dims_);
  minitensor::Tensor<ScalarT> expA(num_dims_), Fpnew(num_dims_);

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < num_pts_; ++qp) {
      // local parameters
      lame        = elastic_modulus(cell, qp) * poissons_ratio(cell, qp) / (1.0 + poissons_ratio(cell, qp)) / (1.0 - 2.0 * poissons_ratio(cell, qp));
      mu          = elastic_modulus(cell, qp) / 2.0 / (1.0 + poissons_ratio(cell, qp));
      bulkModulus = lame + (2. / 3.) * mu;

      // elastic matrix
      Celastic = lame * id3 + mu * (id1 + id2);

      // elastic compliance tangent matrix
      compliance = (1. / bulkModulus / 9.) * id3 + (1. / mu / 2.) * (0.5 * (id1 + id2) - (1. / 3.) * id3);

      F.fill(def_grad, cell, qp);
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          Fpn(i, j) = ScalarT(Fpold(cell, qp, i, j));
          alphaVal(i, j) = backStressold(cell, qp, i, j);
        }
      }

      // Compute trial elastic log strain
      Fpinv = minitensor::inverse(Fpn);
      Cpinv = Fpinv * minitensor::transpose(Fpinv);
      be    = F * Cpinv * minitensor::transpose(F);
      logbe = minitensor::log_sym<ScalarT>(be);
      eps_e = 0.5 * logbe;

      // Trial Kirchhoff stress
      sigmaTr = lame * minitensor::trace(eps_e) * I + 2.0 * mu * eps_e;
      sigmaVal = sigmaTr;

      ScalarT kappaVal = capParameterold(cell, qp);

      // define plastic strain increment, its two invariants: dev, and vol
      ScalarT deqps(0.0), devolps(0.0);

      // define a temporary tensor to store previous back stress
      alphaTr = alphaVal;

      // check yielding
      ScalarT f = compute_f(sigmaVal, alphaVal, kappaVal);

      // plastic correction
      ScalarT dgamma = 0.0;
      if (f > 0.0) {
        // use sigmaTr as a proxy for sigmaN in the return mapping logic
        sigmaN = sigmaTr; // In explicit, this is often done.

        dfdsigma = compute_dfdsigma(sigmaVal, alphaVal, kappaVal);
        dgdsigma = compute_dgdsigma(sigmaVal, alphaVal, kappaVal);
        dfdalpha = -dfdsigma;
        ScalarT dfdkappa = compute_dfdkappa(sigmaVal, alphaVal, kappaVal);
        ScalarT J2_alpha = 0.5 * minitensor::dotdot(alphaVal, alphaVal);
        halpha = compute_halpha(dgdsigma, J2_alpha);
        ScalarT I1_dgdsigma = minitensor::trace(dgdsigma);
        ScalarT dedkappa = compute_dedkappa(kappaVal);

        ScalarT hkappa;
        if (dedkappa != 0.0)
          hkappa = I1_dgdsigma / dedkappa;
        else
          hkappa = 0.0;

        ScalarT kai = minitensor::dotdot(dfdsigma, minitensor::dotdot(Celastic, dgdsigma)) - minitensor::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

        // we don't have "depsilon" here in the same way, but we can use the return mapping logic.
        // Actually, the small strain model uses: dgamma = dfdotCe : depsilon / kai
        // In FD, we can just do the return mapping iteratively.
        
        // Let's use the stress correction algorithm directly as the return mapping
        bool     condition     = false;
        int      iteration     = 0;
        int      max_iteration = 100;
        RealType tolerance     = 1.0e-10;
        
        while (condition == false) {
          f = compute_f(sigmaVal, alphaVal, kappaVal);
          if (std::abs(f) < tolerance) break;
          
          dfdsigma = compute_dfdsigma(sigmaVal, alphaVal, kappaVal);
          dgdsigma = compute_dgdsigma(sigmaVal, alphaVal, kappaVal);
          dfdalpha = -dfdsigma;
          dfdkappa = compute_dfdkappa(sigmaVal, alphaVal, kappaVal);
          J2_alpha = 0.5 * minitensor::dotdot(alphaVal, alphaVal);
          halpha = compute_halpha(dgdsigma, J2_alpha);
          I1_dgdsigma = minitensor::trace(dgdsigma);
          dedkappa = compute_dedkappa(kappaVal);

          if (dedkappa != 0)
            hkappa = I1_dgdsigma / dedkappa;
          else
            hkappa = 0;

          ScalarT kai = minitensor::dotdot(dfdsigma, minitensor::dotdot(Celastic, dgdsigma));
          kai = kai - minitensor::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

          if (iteration > max_iteration) break;

          ScalarT delta_gamma;
          if (kai != 0)
            delta_gamma = f / kai;
          else
            delta_gamma = 0;

          ScalarT dkappa = delta_gamma * hkappa;
          if (dkappa > 0.0) dkappa = 0.0;

          // update
          sigmaVal -= delta_gamma * minitensor::dotdot(Celastic, dgdsigma);
          alphaVal += delta_gamma * halpha;
          kappaVal += dkappa;
          dgamma += delta_gamma;

          iteration++;
        }

        // update Fp
        dgdsigma = compute_dgdsigma(sigmaVal, alphaVal, kappaVal);
        expA = minitensor::exp(dgamma * dgdsigma);
        Fpnew = expA * Fpn;
        
        // compute plastic strain increment
        dsigma       = sigmaTr - sigmaVal;
        deps_plastic = minitensor::dotdot(compliance, dsigma);
        devolps      = minitensor::trace(deps_plastic);
        dev_plastic  = deps_plastic - (1. / 3.) * devolps * I;
        deqps        = std::sqrt(2.0/3.0) * minitensor::norm(dev_plastic);

      } else {
        Fpnew = Fpn;
      }

      // Cauchy stress = Kirchhoff stress / J
      ScalarT J = jac_det(cell, qp);
      for (int i = 0; i < num_dims_; ++i) {
        for (int j = 0; j < num_dims_; ++j) {
          stress(cell, qp, i, j)     = sigmaVal(i, j) / J;
          Fp(cell, qp, i, j)         = Fpnew(i, j);
          backStress(cell, qp, i, j) = alphaVal(i, j);
        }
      }

      capParameter(cell, qp)     = kappaVal;
      eqps(cell, qp)             = eqpsold(cell, qp) + deqps;
      volPlasticStrain(cell, qp) = volPlasticStrainold(cell, qp) + devolps;

    }  // loop over qps
  }  // loop over cell
}

// Reuse the physics functions from CapExplicitModel
template <typename EvalT, typename Traits>
typename CapExplicitFD<EvalT, Traits>::ScalarT
CapExplicitFD<EvalT, Traits>::compute_f(minitensor::Tensor<ScalarT>& sigma, minitensor::Tensor<ScalarT>& alpha, ScalarT& kappa)
{
  xi = sigma - alpha;
  ScalarT I1 = minitensor::trace(xi), p = I1 / 3;
  s = xi - p * minitensor::identity<ScalarT>(3);
  ScalarT J2 = 0.5 * minitensor::dotdot(s, s);
  ScalarT J3 = minitensor::det(s);

  ScalarT Gamma = 1.0;
  if (psi != 0 && J2 != 0)
    Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) + (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

  ScalarT Ff_I1 = A - C * std::exp(D * I1) - theta * I1;
  ScalarT Ff_kappa = A - C * std::exp(D * kappa) - theta * kappa;
  ScalarT X = kappa - R * Ff_kappa;
  ScalarT Fc = 1.0;
  if ((kappa - I1) > 0 && ((X - kappa) != 0)) Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

  return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
}

template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapExplicitFD<EvalT, Traits>::ScalarT>
CapExplicitFD<EvalT, Traits>::compute_dfdsigma(minitensor::Tensor<ScalarT>& sigma, minitensor::Tensor<ScalarT>& alpha, ScalarT& kappa)
{
  xi = sigma - alpha;
  ScalarT I1 = minitensor::trace(xi), p = I1 / 3;
  s = xi - p * minitensor::identity<ScalarT>(3);
  ScalarT J2 = 0.5 * minitensor::dotdot(s, s);
  ScalarT J3 = minitensor::det(s);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) dJ3dsigma(i, j) = s(i, j) * s(i, j) - 2 * J2 * I(i, j) / 3;

  ScalarT Ff_I1 = A - C * std::exp(D * I1) - theta * I1;
  ScalarT Ff_kappa = A - C * std::exp(D * kappa) - theta * kappa;
  ScalarT X = kappa - R * Ff_kappa;
  ScalarT Fc = 1.0;
  if ((kappa - I1) > 0) Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

  ScalarT Gamma = 1.0;
  if (psi != 0 && J2 != 0)
    Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) + (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

  ScalarT dFfdI1 = -(D * C * std::exp(D * I1) + theta);
  ScalarT dFcdI1 = 0.0;
  if ((kappa - I1) > 0 && ((X - kappa) != 0)) dFcdI1 = -2.0 * (I1 - kappa) / (X - kappa) / (X - kappa);
  ScalarT dfdI1 = -(Ff_I1 - N) * (2 * Fc * dFfdI1 + (Ff_I1 - N) * dFcdI1);
  ScalarT dGammadJ2 = 0.0;
  if (J2 != 0) dGammadJ2 = 9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0 * (1.0 - 1.0 / psi);
  ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;
  ScalarT dGammadJ3 = 0.0;
  if (J2 != 0) dGammadJ3 = -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0 * (1.0 - 1.0 / psi);
  ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

  return dfdI1 * I + dfdJ2 * s + dfdJ3 * dJ3dsigma;
}

template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapExplicitFD<EvalT, Traits>::ScalarT>
CapExplicitFD<EvalT, Traits>::compute_dgdsigma(minitensor::Tensor<ScalarT>& sigma, minitensor::Tensor<ScalarT>& alpha, ScalarT& kappa)
{
  xi = sigma - alpha;
  ScalarT I1 = minitensor::trace(xi), p = I1 / 3;
  s = xi - p * minitensor::identity<ScalarT>(3);
  ScalarT J2 = 0.5 * minitensor::dotdot(s, s);
  ScalarT J3 = minitensor::det(s);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) dJ3dsigma(i, j) = s(i, j) * s(i, j) - 2 * J2 * I(i, j) / 3;

  ScalarT Ff_I1 = A - C * std::exp(L * I1) - phi * I1;
  ScalarT Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;
  ScalarT X = kappa - Q * Ff_kappa;
  ScalarT Fc = 1.0;
  if ((kappa - I1) > 0) Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

  ScalarT Gamma = 1.0;
  if (psi != 0 && J2 != 0)
    Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) + (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

  ScalarT dFfdI1 = -(L * C * std::exp(L * I1) + phi);
  ScalarT dFcdI1 = 0.0;
  if ((kappa - I1) > 0 && ((X - kappa) != 0)) dFcdI1 = -2.0 * (I1 - kappa) / (X - kappa) / (X - kappa);
  ScalarT dfdI1 = -(Ff_I1 - N) * (2 * Fc * dFfdI1 + (Ff_I1 - N) * dFcdI1);
  ScalarT dGammadJ2 = 0.0;
  if (J2 != 0) dGammadJ2 = 9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0 * (1.0 - 1.0 / psi);
  ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;
  ScalarT dGammadJ3 = 0.0;
  if (J2 != 0) dGammadJ3 = -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0 * (1.0 - 1.0 / psi);
  ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

  return dfdI1 * I + dfdJ2 * s + dfdJ3 * dJ3dsigma;
}

template <typename EvalT, typename Traits>
typename CapExplicitFD<EvalT, Traits>::ScalarT
CapExplicitFD<EvalT, Traits>::compute_dfdkappa(minitensor::Tensor<ScalarT>& sigma, minitensor::Tensor<ScalarT>& alpha, ScalarT& kappa)
{
  xi = sigma - alpha;
  ScalarT I1 = minitensor::trace(xi);
  ScalarT Ff_I1 = A - C * std::exp(D * I1) - theta * I1;
  ScalarT Ff_kappa = A - C * std::exp(D * kappa) - theta * kappa;
  ScalarT X = kappa - R * Ff_kappa;
  ScalarT dFcdkappa = 0.0;
  if ((kappa - I1) > 0 && ((X - kappa) != 0)) {
    dFcdkappa = 2 * (I1 - kappa) * ((X - kappa) + R * (I1 - kappa) * (theta + D * C * std::exp(D * kappa))) / (X - kappa) / (X - kappa) / (X - kappa);
  }
  return -dFcdkappa * (Ff_I1 - N) * (Ff_I1 - N);
}

template <typename EvalT, typename Traits>
typename CapExplicitFD<EvalT, Traits>::ScalarT
CapExplicitFD<EvalT, Traits>::compute_Galpha(ScalarT& J2_alpha)
{
  if (N != 0) return 1.0 - std::pow(J2_alpha, 0.5) / N;
  else return 0.0;
}

template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapExplicitFD<EvalT, Traits>::ScalarT>
CapExplicitFD<EvalT, Traits>::compute_halpha(minitensor::Tensor<ScalarT>& dgdsigma, ScalarT& J2_alpha)
{
  ScalarT Galpha = compute_Galpha(J2_alpha);
  ScalarT I1 = minitensor::trace(dgdsigma), p = I1 / 3;
  minitensor::Tensor<ScalarT> s_loc = dgdsigma - p * I;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) halpha(i, j) = calpha * Galpha * s_loc(i, j);
  return halpha;
}

template <typename EvalT, typename Traits>
typename CapExplicitFD<EvalT, Traits>::ScalarT
CapExplicitFD<EvalT, Traits>::compute_dedkappa(ScalarT& kappa)
{
  ScalarT Ff_kappa0 = A - C * std::exp(L * kappa0) - phi * kappa0;
  ScalarT X0 = kappa0 - Q * Ff_kappa0;
  ScalarT Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;
  ScalarT X = kappa - Q * Ff_kappa;
  ScalarT dedX = (D1 - 2 * D2 * (X - X0)) * std::exp((D1 - D2 * (X - X0)) * (X - X0)) * W;
  ScalarT dXdkappa = 1 + Q * C * L * std::exp(L * kappa) + Q * phi;
  return dedX * dXdkappa;
}

}  // namespace LCM
