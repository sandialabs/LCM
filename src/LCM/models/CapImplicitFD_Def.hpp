// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include <MiniTensor.h>

#include <Intrepid2_FunctionSpaceTools.hpp>
#include <Phalanx_DataLayout.hpp>
#include <typeinfo>

#include "LocalNonlinearSolver.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
CapImplicitFD<EvalT, Traits>::CapImplicitFD(Teuchos::ParameterList* p, const Teuchos::RCP<Albany::Layouts>& dl)
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
  std::string tangent_string        = (*field_name_map_)["Material Tangent"];

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
  if (compute_tangent_) {
    this->eval_field_map_.insert(std::make_pair(tangent_string, dl->qp_tensor4));
  }

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
}

template <typename EvalT, typename Traits>
void
CapImplicitFD<EvalT, Traits>::computeState(typename Traits::EvalData workset, DepFieldMap dep_fields, FieldMap eval_fields)
{
  std::string F_string              = (*field_name_map_)["F"];
  std::string J_string              = (*field_name_map_)["J"];
  std::string cauchy_string         = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string             = (*field_name_map_)["Fp"];
  std::string backStress_string     = (*field_name_map_)["Back_Stress"];
  std::string capParameter_string   = (*field_name_map_)["Cap_Parameter"];
  std::string eqps_string           = (*field_name_map_)["eqps"];
  std::string volPlasticStrain_string = (*field_name_map_)["volPlastic_Strain"];
  std::string tangent_string        = (*field_name_map_)["Material Tangent"];

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
  PHX::MDField<ScalarT> tangent;
  if (compute_tangent_) tangent = *eval_fields[tangent_string];

  // get State Variables
  Albany::MDArray Fpold               = (*workset.stateArrayPtr)[Fp_string + "_old"];
  Albany::MDArray stressold           = (*workset.stateArrayPtr)[cauchy_string + "_old"];
  Albany::MDArray backStressold       = (*workset.stateArrayPtr)[backStress_string + "_old"];
  Albany::MDArray capParameterold     = (*workset.stateArrayPtr)[capParameter_string + "_old"];
  Albany::MDArray eqpsold             = (*workset.stateArrayPtr)[eqps_string + "_old"];
  Albany::MDArray volPlasticStrainold = (*workset.stateArrayPtr)[volPlasticStrain_string + "_old"];

  minitensor::Tensor<ScalarT> F(num_dims_), Fpn(num_dims_), Fpinv(num_dims_), Cpinv(num_dims_), be(num_dims_), logbe(num_dims_), eps_e(num_dims_);
  minitensor::Tensor<ScalarT> expA(num_dims_), Fpnew(num_dims_);

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < num_pts_; ++qp) {
      // local parameters
      ScalarT lame        = elastic_modulus(cell, qp) * poissons_ratio(cell, qp) / (1.0 + poissons_ratio(cell, qp)) / (1.0 - 2.0 * poissons_ratio(cell, qp));
      ScalarT mu          = elastic_modulus(cell, qp) / 2.0 / (1.0 + poissons_ratio(cell, qp));
      ScalarT bulkModulus = lame + (2. / 3.) * mu;

      // elastic matrix
      minitensor::Tensor4<ScalarT> Celastic =
          lame * minitensor::identity_3<ScalarT>(3) + mu * (minitensor::identity_1<ScalarT>(3) + minitensor::identity_2<ScalarT>(3));

      // elastic compliance tangent matrix
      minitensor::Tensor4<ScalarT> compliance =
          (1. / bulkModulus / 9.) * minitensor::identity_3<ScalarT>(3) +
          (1. / mu / 2.) * (0.5 * (minitensor::identity_1<ScalarT>(3) + minitensor::identity_2<ScalarT>(3)) - (1. / 3.) * minitensor::identity_3<ScalarT>(3));

      F.fill(def_grad, cell, qp);
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          Fpn(i, j) = ScalarT(Fpold(cell, qp, i, j));
        }
      }

      // Compute trial elastic log strain
      Fpinv = minitensor::inverse(Fpn);
      Cpinv = Fpinv * minitensor::transpose(Fpinv);
      be    = F * Cpinv * minitensor::transpose(F);
      logbe = minitensor::log_sym<ScalarT>(be);
      eps_e = 0.5 * logbe;

      // Trial Kirchhoff stress
      minitensor::Tensor<ScalarT> sigmaTr = lame * minitensor::trace(eps_e) * minitensor::identity<ScalarT>(3) + 2.0 * mu * eps_e;
      minitensor::Tensor<ScalarT> sigmaVal = sigmaTr;
      minitensor::Tensor<ScalarT> alphaVal(3);
      for (int i = 0; i < num_dims_; ++i) {
        for (int j = 0; j < num_dims_; ++j) {
          alphaVal(i, j) = backStressold(cell, qp, i, j);
        }
      }

      ScalarT kappaVal  = capParameterold(cell, qp);
      ScalarT dgammaVal = 0.0;

      // define plastic strain increment, its two invariants: dev, and vol
      minitensor::Tensor<ScalarT> deps_plastic(3, minitensor::Filler::ZEROS);
      ScalarT                     deqps(0.0), devolps(0.0);

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
          compute_ResidJacobian(XXVal, R, dRdX, sigmaTr, alphaVal, kappaVal, Celastic, kappa_flag);
          normR = 0.0;
          for (int i = 0; i < 13; i++) normR += R[i] * R[i];
          normR = std::sqrt(normR);
          if (iter == 0) normR0 = normR;
          if (normR0 != 0) conv = normR / normR0;
          else conv = normR0;
          if (conv < 1.e-11 || normR < 1.e-11) break;
          if (iter > 20) break;

          std::vector<ScalarT> XXValK = XXVal;
          solver.solve(dRdX, XXValK, R);
          if (XXValK[11] > XXVal[11]) kappa_flag = true;
          else {
            XXVal      = XXValK;
            kappa_flag = false;
          }
          iter++;
        }
        solver.computeFadInfo(dRdX, XXVal, R);
      }

      // update stress and state
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

      kappaVal = XXVal[11];
      dgammaVal = XXVal[12];

      // Update Fp
      if (dgammaVal > 0) {
        minitensor::Tensor<ScalarT> dgdsigma_loc = compute_dgdsigma(XXVal);
        expA = minitensor::exp(dgammaVal * dgdsigma_loc);
        Fpnew = expA * Fpn;
      } else {
        Fpnew = Fpn;
      }

      // plastic strain invariants
      minitensor::Tensor<ScalarT> dsigma = sigmaTr - sigmaVal;
      deps_plastic                       = minitensor::dotdot(compliance, dsigma);
      devolps                                 = minitensor::trace(deps_plastic);
      minitensor::Tensor<ScalarT> dev_plastic = deps_plastic - (1.0 / 3.0) * devolps * minitensor::identity<ScalarT>(3);
      deqps = std::sqrt(2.0/3.0) * minitensor::norm(dev_plastic);

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

      if (compute_tangent_) {
        minitensor::Tensor4<ScalarT> Cep = compute_Cep(Celastic, sigmaVal, alphaVal, kappaVal, dgammaVal);
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            for (int k(0); k < num_dims_; ++k) {
              for (int l(0); l < num_dims_; ++l) {
                tangent(cell, qp, i, j, k, l) = Cep(i, j, k, l);
              }
            }
          }
        }
      }
    }
  }
}

// Local functions for CapImplicitFD (Reuse logic from CapImplicitModel)
template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitFD<EvalT, Traits>::compute_f(minitensor::Tensor<T>& sigma, minitensor::Tensor<T>& alpha, T& kappa)
{
  minitensor::Tensor<T> xi_loc = sigma - alpha;
  T I1 = minitensor::trace(xi_loc), p = I1 / 3.;
  minitensor::Tensor<T> s_loc = xi_loc - p * minitensor::identity<T>(3);
  T J2 = 0.5 * minitensor::dotdot(s_loc, s_loc);
  T J3 = minitensor::det(s_loc);
  T Gamma = 1.0;
  if (psi != 0 && J2 != 0)
    Gamma = 0.5 * (1. - 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5) + (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)) / psi);
  T Ff_I1 = A - C * std::exp(D * I1) - theta * I1;
  T Ff_kappa = A - C * std::exp(D * kappa) - theta * kappa;
  T X = kappa - R * Ff_kappa;
  T Fc = 1.0;
  if ((kappa - I1) > 0 && ((X - kappa) != 0)) Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);
  return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
}

template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitFD<EvalT, Traits>::compute_g(minitensor::Tensor<T>& sigma, minitensor::Tensor<T>& alpha, T& kappa)
{
  minitensor::Tensor<T> xi_loc = sigma - alpha;
  T I1 = minitensor::trace(xi_loc), p = I1 / 3.;
  minitensor::Tensor<T> s_loc = xi_loc - p * minitensor::identity<T>(3);
  T J2 = 0.5 * minitensor::dotdot(s_loc, s_loc);
  T J3 = minitensor::det(s_loc);
  T Gamma = 1.0;
  if (psi != 0 && J2 != 0)
    Gamma = 0.5 * (1. - 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5) + (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)) / psi);
  T Ff_I1 = A - C * std::exp(L * I1) - phi * I1;
  T Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;
  T X = kappa - Q * Ff_kappa;
  T Fc = 1.0;
  if ((kappa - I1) > 0 && ((X - kappa) != 0)) Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);
  return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
}

template <typename EvalT, typename Traits>
std::vector<typename CapImplicitFD<EvalT, Traits>::ScalarT>
CapImplicitFD<EvalT, Traits>::initialize(minitensor::Tensor<ScalarT>& sigmaVal, minitensor::Tensor<ScalarT>& alphaVal, ScalarT& kappaVal, ScalarT& dgammaVal)
{
  std::vector<ScalarT> XX(13);
  XX[0] = sigmaVal(0, 0); XX[1] = sigmaVal(1, 1); XX[2] = sigmaVal(2, 2);
  XX[3] = sigmaVal(1, 2); XX[4] = sigmaVal(0, 2); XX[5] = sigmaVal(0, 1);
  XX[6] = alphaVal(0, 0); XX[7] = alphaVal(1, 1); XX[8] = alphaVal(1, 2);
  XX[9] = alphaVal(0, 2); XX[10] = alphaVal(0, 1);
  XX[11] = kappaVal; XX[12] = dgammaVal;
  return XX;
}

template <typename EvalT, typename Traits>
void
CapImplicitFD<EvalT, Traits>::compute_ResidJacobian(
    std::vector<ScalarT> const&         XXVal,
    std::vector<ScalarT>&               R,
    std::vector<ScalarT>&               dRdX,
    const minitensor::Tensor<ScalarT>&  sigmaTr,
    const minitensor::Tensor<ScalarT>&  alphaTr,
    ScalarT const&                      kappaTr,
    minitensor::Tensor4<ScalarT> const& Celastic,
    bool                                kappa_flag)
{
  std::vector<DFadType> Rfad(13), XX(13);
  std::vector<ScalarT>  XXtmp(13);
  for (int i = 0; i < 13; ++i) {
    XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XXVal[i]);
    XX[i]    = DFadType(13, i, XXtmp[i]);
  }
  minitensor::Tensor<DFadType> sigma(3), alpha(3);
  sigma(0, 0) = XX[0]; sigma(1, 1) = XX[1]; sigma(2, 2) = XX[2];
  sigma(1, 2) = XX[3]; sigma(0, 2) = XX[4]; sigma(0, 1) = XX[5];
  sigma(2, 1) = XX[3]; sigma(2, 0) = XX[4]; sigma(1, 0) = XX[5];
  alpha(0, 0) = XX[6]; alpha(1, 1) = XX[7]; alpha(1, 2) = XX[8];
  alpha(0, 2) = XX[9]; alpha(0, 1) = XX[10];
  alpha(2, 1) = XX[8]; alpha(2, 0) = XX[9]; alpha(1, 0) = XX[10];
  alpha(2, 2) = -XX[6] - XX[7];

  DFadType kappa = XX[11], dgamma = XX[12];
  DFadType f = compute_f(sigma, alpha, kappa);
  minitensor::Tensor<DFadType> dgdsigma_loc = compute_dgdsigma(XX);
  DFadType J2_alpha = 0.5 * minitensor::dotdot(alpha, alpha);
  minitensor::Tensor<DFadType> halpha_loc = compute_halpha(dgdsigma_loc, J2_alpha);
  DFadType I1_dgdsigma = minitensor::trace(dgdsigma_loc);
  DFadType dedkappa = compute_dedkappa(kappa);
  DFadType hkappa_loc = compute_hkappa(I1_dgdsigma, dedkappa);

  minitensor::Tensor<DFadType> t_tens = minitensor::dotdot(Celastic, dgdsigma_loc);
  Rfad[0] = dgamma * t_tens(0, 0) + sigma(0, 0) - sigmaTr(0, 0);
  Rfad[1] = dgamma * t_tens(1, 1) + sigma(1, 1) - sigmaTr(1, 1);
  Rfad[2] = dgamma * t_tens(2, 2) + sigma(2, 2) - sigmaTr(2, 2);
  Rfad[3] = dgamma * t_tens(1, 2) + sigma(1, 2) - sigmaTr(1, 2);
  Rfad[4] = dgamma * t_tens(0, 2) + sigma(0, 2) - sigmaTr(0, 2);
  Rfad[5] = dgamma * t_tens(0, 1) + sigma(0, 1) - sigmaTr(0, 1);
  Rfad[6] = dgamma * halpha_loc(0, 0) - alpha(0, 0) + alphaTr(0, 0);
  Rfad[7] = dgamma * halpha_loc(1, 1) - alpha(1, 1) + alphaTr(1, 1);
  Rfad[8] = dgamma * halpha_loc(1, 2) - alpha(1, 2) + alphaTr(1, 2);
  Rfad[9] = dgamma * halpha_loc(0, 2) - alpha(0, 2) + alphaTr(0, 2);
  Rfad[10] = dgamma * halpha_loc(0, 1) - alpha(0, 1) + alphaTr(0, 1);
  if (!kappa_flag) Rfad[11] = dgamma * hkappa_loc - kappa + kappaTr;
  else Rfad[11] = 0;
  Rfad[12] = f;

  for (int i = 0; i < 13; i++) R[i] = Rfad[i].val();
  for (int i = 0; i < 13; i++)
    for (int j = 0; j < 13; j++) dRdX[i + 13 * j] = Rfad[i].dx(j);
  if (kappa_flag) {
    for (int j = 0; j < 13; j++) dRdX[11 + 13 * j] = 0.0;
    dRdX[11 + 13 * 11] = 1.0;
  }
}

template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapImplicitFD<EvalT, Traits>::ScalarT>
CapImplicitFD<EvalT, Traits>::compute_dfdsigma(std::vector<ScalarT> const& XX)
{
  std::vector<DFadType> XXFad(13);
  std::vector<ScalarT>  XXtmp(13);
  for (int i = 0; i < 13; ++i) {
    XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XX[i]);
    XXFad[i] = DFadType(13, i, XXtmp[i]);
  }
  minitensor::Tensor<DFadType> sigma(3), alpha(3);
  sigma(0, 0) = XXFad[0]; sigma(1, 1) = XXFad[1]; sigma(2, 2) = XXFad[2];
  sigma(1, 2) = XXFad[3]; sigma(0, 2) = XXFad[4]; sigma(0, 1) = XXFad[5];
  sigma(2, 1) = XXFad[3]; sigma(2, 0) = XXFad[4]; sigma(1, 0) = XXFad[5];
  alpha(0, 0) = XXFad[6]; alpha(1, 1) = XXFad[7]; alpha(1, 2) = XXFad[8];
  alpha(0, 2) = XXFad[9]; alpha(0, 1) = XXFad[10];
  alpha(2, 1) = XXFad[8]; alpha(2, 0) = XXFad[9]; alpha(1, 0) = XXFad[10];
  alpha(2, 2) = -XXFad[6] - XXFad[7];
  DFadType kappa = XXFad[11];
  DFadType f = compute_f(sigma, alpha, kappa);
  minitensor::Tensor<ScalarT> res(3);
  res(0, 0) = f.dx(0); res(1, 1) = f.dx(1); res(2, 2) = f.dx(2);
  res(1, 2) = f.dx(3); res(0, 2) = f.dx(4); res(0, 1) = f.dx(5);
  res(2, 1) = f.dx(3); res(2, 0) = f.dx(4); res(1, 0) = f.dx(5);
  return res;
}

template <typename EvalT, typename Traits>
typename CapImplicitFD<EvalT, Traits>::ScalarT
CapImplicitFD<EvalT, Traits>::compute_dfdkappa(std::vector<ScalarT> const& XX)
{
  std::vector<DFadType> XXFad(13);
  std::vector<ScalarT>  XXtmp(13);
  for (int i = 0; i < 13; ++i) {
    XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XX[i]);
    XXFad[i] = DFadType(13, i, XXtmp[i]);
  }
  minitensor::Tensor<DFadType> sigma(3), alpha(3);
  sigma(0, 0) = XXFad[0]; sigma(1, 1) = XXFad[1]; sigma(2, 2) = XXFad[2];
  sigma(1, 2) = XXFad[3]; sigma(0, 2) = XXFad[4]; sigma(0, 1) = XXFad[5];
  sigma(2, 1) = XXFad[3]; sigma(2, 0) = XXFad[4]; sigma(1, 0) = XXFad[5];
  alpha(0, 0) = XXFad[6]; alpha(1, 1) = XXFad[7]; alpha(1, 2) = XXFad[8];
  alpha(0, 2) = XXFad[9]; alpha(0, 1) = XXFad[10];
  alpha(2, 1) = XXFad[8]; alpha(2, 0) = XXFad[9]; alpha(1, 0) = XXFad[10];
  alpha(2, 2) = -XXFad[6] - XXFad[7];
  DFadType kappa = XXFad[11];
  return compute_f(sigma, alpha, kappa).dx(11);
}

template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapImplicitFD<EvalT, Traits>::ScalarT>
CapImplicitFD<EvalT, Traits>::compute_dgdsigma(std::vector<ScalarT> const& XX)
{
  std::vector<DFadType> XXFad(13);
  std::vector<ScalarT>  XXtmp(13);
  for (int i = 0; i < 13; ++i) {
    XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XX[i]);
    XXFad[i] = DFadType(13, i, XXtmp[i]);
  }
  minitensor::Tensor<DFadType> sigma(3), alpha(3);
  sigma(0, 0) = XXFad[0]; sigma(1, 1) = XXFad[1]; sigma(2, 2) = XXFad[2];
  sigma(1, 2) = XXFad[3]; sigma(0, 2) = XXFad[4]; sigma(0, 1) = XXFad[5];
  sigma(2, 1) = XXFad[3]; sigma(2, 0) = XXFad[4]; sigma(1, 0) = XXFad[5];
  alpha(0, 0) = XXFad[6]; alpha(1, 1) = XXFad[7]; alpha(1, 2) = XXFad[8];
  alpha(0, 2) = XXFad[9]; alpha(0, 1) = XXFad[10];
  alpha(2, 1) = XXFad[8]; alpha(2, 0) = XXFad[9]; alpha(1, 0) = XXFad[10];
  alpha(2, 2) = -XXFad[6] - XXFad[7];
  DFadType kappa = XXFad[11];
  DFadType g = compute_g(sigma, alpha, kappa);
  minitensor::Tensor<ScalarT> res(3);
  res(0, 0) = g.dx(0); res(1, 1) = g.dx(1); res(2, 2) = g.dx(2);
  res(1, 2) = g.dx(3); res(0, 2) = g.dx(4); res(0, 1) = g.dx(5);
  res(2, 1) = g.dx(3); res(2, 0) = g.dx(4); res(1, 0) = g.dx(5);
  return res;
}

template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapImplicitFD<EvalT, Traits>::DFadType>
CapImplicitFD<EvalT, Traits>::compute_dgdsigma(std::vector<DFadType> const& XX)
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
  sigma(0, 0) = D2XX[0]; sigma(1, 1) = D2XX[1]; sigma(2, 2) = D2XX[2];
  sigma(1, 2) = D2XX[3]; sigma(0, 2) = D2XX[4]; sigma(0, 1) = D2XX[5];
  sigma(2, 1) = D2XX[3]; sigma(2, 0) = D2XX[4]; sigma(1, 0) = D2XX[5];
  alpha(0, 0) = D2XX[6]; alpha(1, 1) = D2XX[7]; alpha(1, 2) = D2XX[8];
  alpha(0, 2) = D2XX[9]; alpha(0, 1) = D2XX[10];
  alpha(2, 1) = D2XX[8]; alpha(2, 0) = D2XX[9]; alpha(1, 0) = D2XX[10];
  alpha(2, 2) = -D2XX[6] - D2XX[7];
  D2FadType kappa = D2XX[11];
  D2FadType g = compute_g(sigma, alpha, kappa);
  minitensor::Tensor<DFadType> res(3);
  res(0, 0) = g.dx(0); res(1, 1) = g.dx(1); res(2, 2) = g.dx(2);
  res(1, 2) = g.dx(3); res(0, 2) = g.dx(4); res(0, 1) = g.dx(5);
  res(2, 1) = g.dx(3); res(2, 0) = g.dx(4); res(1, 0) = g.dx(5);
  return res;
}

template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitFD<EvalT, Traits>::compute_Galpha(T J2_alpha)
{
  if (N != 0) return 1.0 - pow(J2_alpha, 0.5) / N;
  else return 0.0;
}

template <typename EvalT, typename Traits>
template <typename T>
minitensor::Tensor<T>
CapImplicitFD<EvalT, Traits>::compute_halpha(minitensor::Tensor<T> const& dgdsigma, T const J2_alpha)
{
  T Galpha = compute_Galpha(J2_alpha);
  T I1 = minitensor::trace(dgdsigma), p = I1 / 3.0;
  minitensor::Tensor<T> s_loc = dgdsigma - p * minitensor::identity<T>(3);
  minitensor::Tensor<T> halpha_loc(3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) halpha_loc(i, j) = calpha * Galpha * s_loc(i, j);
  return halpha_loc;
}

template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitFD<EvalT, Traits>::compute_dedkappa(T const kappa)
{
  T Ff_kappa0 = A - C * std::exp(L * kappa0) - phi * kappa0;
  T X0 = kappa0 - Q * Ff_kappa0;
  T Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;
  T X = kappa - Q * Ff_kappa;
  T dedX = (D1 - 2. * D2 * (X - X0)) * std::exp((D1 - D2 * (X - X0)) * (X - X0)) * W;
  T dXdkappa = 1. + Q * C * L * exp(L * kappa) + Q * phi;
  return dedX * dXdkappa;
}

template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitFD<EvalT, Traits>::compute_hkappa(T const I1_dgdsigma, T const dedkappa)
{
  if (dedkappa != 0) return I1_dgdsigma / dedkappa;
  else return 0;
}

template <typename EvalT, typename Traits>
minitensor::Tensor4<typename CapImplicitFD<EvalT, Traits>::ScalarT>
CapImplicitFD<EvalT, Traits>::compute_Cep(
    minitensor::Tensor4<ScalarT>& Celastic,
    minitensor::Tensor<ScalarT>&  sigma,
    minitensor::Tensor<ScalarT>&  alpha,
    ScalarT&                      kappa,
    ScalarT&                      dgamma)
{
  if (dgamma == 0) return Celastic;
  std::vector<ScalarT> XX = initialize(sigma, alpha, kappa, dgamma);
  minitensor::Tensor<ScalarT> dfdsigma_loc = compute_dfdsigma(XX);
  minitensor::Tensor<ScalarT> dfdalpha_loc = dfdsigma_loc * (-1.0);
  ScalarT dfdkappa_loc = compute_dfdkappa(XX);
  minitensor::Tensor<ScalarT> dgdsigma_loc = compute_dgdsigma(XX);
  ScalarT J2_alpha = 0.5 * minitensor::dotdot(alpha, alpha);
  minitensor::Tensor<ScalarT> halpha_loc = compute_halpha(dgdsigma_loc, J2_alpha);
  ScalarT I1_dgdsigma = minitensor::trace(dgdsigma_loc);
  ScalarT dedkappa_loc = compute_dedkappa(kappa);
  ScalarT hkappa_loc = compute_hkappa(I1_dgdsigma, dedkappa_loc);
  ScalarT chi_loc = minitensor::dotdot(minitensor::dotdot(dfdsigma_loc, Celastic), dgdsigma_loc) - minitensor::dotdot(dfdalpha_loc, halpha_loc) - dfdkappa_loc * hkappa_loc;
  if (chi_loc == 0) chi_loc = 1e-16;
  return Celastic - 1.0 / chi_loc * minitensor::dotdot(Celastic, minitensor::tensor(dgdsigma_loc, minitensor::dotdot(dfdsigma_loc, Celastic)));
}

}  // namespace LCM
