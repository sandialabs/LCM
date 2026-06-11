// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
//
// Small-strain, three-invariant, isotropic/kinematic hardening cap
// plasticity model with refined explicit integration and drift correction.
//
// References ("the sources"):
//  [1] C.D. Foster, R.A. Regueiro, A.F. Fossum, R.I. Borja, "Implicit
//      numerical integration of a three-invariant, isotropic/kinematic
//      hardening cap plasticity model for geomaterials", CMAME 194 (2005)
//      5109-5138.
//  [2] R.A. Regueiro, C.D. Foster, "Bifurcation analysis for a
//      rate-sensitive, non-associative, three-invariant, isotropic/kinematic
//      hardening cap plasticity model for geomaterials: Part I. Small
//      strain", IJNAMG 35 (2011) 201-225.
//  [3] W. Sun, Q. Chen, J.T. Ostien, "Modeling the hydro-mechanical
//      responses of strip and circular punch loadings on water-saturated
//      collapsible geomaterials", Acta Geotechnica 9 (2014) 903-934.
//  [4] A.F. Fossum, R.M. Brannon, "The Sandia GeoModel: Theory and User's
//      Guide", SAND2004-3226 -- and its LAME reference implementation
//      (sierra/code/lame/src/models/development/iso_geomodel_model.F),
//      which arbitrates conventions the papers leave ambiguous.
#include <MiniTensor.h>

#include "Albany_Utils.hpp"
#include "CapIntegrator.hpp"
#include "CapModel.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
CapModelKernel<EvalT, Traits>::CapModelKernel(
    ConstitutiveModel<EvalT, Traits>& model,
    Teuchos::ParameterList*           p,
    Teuchos::RCP<Albany::Layouts> const& dl)
    : BaseKernel(model),
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
      Q_(p->get<RealType>("Q")),
      // Sloan-style adaptive substepping of the explicit integration:
      // modified-Euler (RK1/RK2) pairs with relative stress-error control.
      substep_tolerance_(p->get<RealType>("Substep Tolerance", 1.0e-4)),
      max_substeps_(p->get<int>("Maximum Substeps", 200))
{
  finite_deformation_ = p->get<bool>("Finite Deformation", false);

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

  // define the dependent fields
  setDependentField("Poissons Ratio", dl->qp_scalar);
  setDependentField("Elastic Modulus", dl->qp_scalar);

  if (finite_deformation_) {
    // Exponential/logarithmic-map kinematics: the verified small-strain
    // integrator runs unchanged in logarithmic elastic strain /
    // Kirchhoff stress space; only the kinematics wrap around it.
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
CapModelKernel<EvalT, Traits>::init(
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
  elastic_modulus_ = *dep_fields["Elastic Modulus"];
  poissons_ratio_  = *dep_fields["Poissons Ratio"];
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
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
CapModelKernel<EvalT, Traits>::operator()(int cell, int pt) const
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
  // of logarithmic elastic strain (exponential/logarithmic-map approach):
  // the verified integrator is reused unchanged in that space.
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

    Tensor const Fpinv  = minitensor::inverse(Fp_n);
    Tensor const Cpinv  = minitensor::dot(Fpinv, minitensor::transpose(Fpinv));

    // Trial elastic logarithmic strain from be_tr = F Cp^-1 F^T
    Tensor const be_tr  = minitensor::dot(Fval, minitensor::dot(Cpinv, minitensor::transpose(Fval)));
    Tensor const eps_tr = 0.5 * log_sym(be_tr);

    // Elastic logarithmic strain at t_n (same Cp^-1, old F); the
    // corresponding Kirchhoff stress tau_n = C : eps_e_n is the stress
    // the integrator starts from, so that the trial state
    // tau_n + C : (eps_tr - eps_e_n) = C : eps_tr is exact for the
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
  }

  Tensor const sigmaTr = sigmaN + minitensor::dotdot(Celastic, depsilon);

  // Plastic strain increment invariants
  ScalarT deqps(0.0), devolps(0.0);

  // f has units of stress^2; scale the drift tolerance by E^2 so the
  // convergence check is invariant to the unit system.
  ScalarT const f_tolerance = 1.0e-12 * E * E;

  // The verified physics lives in the shared CapIntegrator (also used by
  // the Permafrost model); this kernel supplies constant parameters, so
  // the integrator's begin/end parameter sets are equal.
  CapIntegrator<ScalarT> integ;
  CapParameters<ScalarT> P;
  P.A      = A_;
  P.D      = D_;
  P.C      = C_;
  P.theta  = theta_;
  P.R      = R_;
  P.kappa0 = kappa0_;
  P.W      = W_;
  P.D1     = D1_;
  P.D2     = D2_;
  P.calpha = calpha_;
  P.psi    = psi_;
  P.N      = N_;
  P.L      = L_;
  P.phi    = phi_;
  P.Q      = Q_;
  P.lame   = lame;
  P.mu     = mu;
  integ.p0 = P;
  integ.p1 = P;
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
