// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include "ACEcommon.hpp"
#include "Albany_STKDiscretization.hpp"
#include "J2Erosion.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
J2ErosionKernel<EvalT, Traits>::J2ErosionKernel(ConstitutiveModel<EvalT, Traits>& model, Teuchos::ParameterList* p, Teuchos::RCP<Albany::Layouts> const& dl)
    : BaseKernel(model)
{
  this->setIntegrationPointLocationFlag(true);

  // Baseline constants
  sat_mod_                  = p->get<RealType>("Saturation Modulus", 0.0);
  sat_exp_                  = p->get<RealType>("Saturation Exponent", 0.0);
  bulk_porosity_            = p->get<RealType>("ACE Bulk Porosity", 0.0);
  Y_weakening_factor_       = p->get<RealType>("ACE Y Weakening Factor", 1.0);
  E_weakening_factor_       = p->get<RealType>("ACE E Weakening Factor", 1.0);
  SL_weakening_factor_      = p->get<RealType>("ACE SL Weakening Factor", 1.0);
  soil_yield_strength_      = p->get<RealType>("ACE Soil Yield Strength", 0.0);
  residual_elastic_modulus_ = p->get<RealType>("ACE Residual Elastic Modulus", 0.0);
  critical_angle_           = p->get<RealType>("ACE Critical Angle", 0.0);
  tensile_strength_         = p->get<RealType>("ACE Tensile Strength", 0.0);
  ice_saturation_material_fit_truncation_ = p->get<RealType>("ACE Material Fit Truncation Ice Saturation", 0.0);
  disable_erosion_          = p->get<bool>("Disable Erosion", false);

  if (p->isParameter("ACE Strain Limit")) {
    strain_limit_ = p->get<RealType>("ACE Strain Limit");
  } else {
    ALBANY_ABORT("ACE Strain Limit not specified in mechanics material file!  To avoid strain failure criterion, set this value to 0.0.");
  }
  if (p->isParameter("ACE Maximum Displacement")) {
    maximum_displacement_ = p->get<RealType>("ACE Maximum Displacement");
  } else {
    ALBANY_ABORT("ACE Maximum Displacement not specified in mechanics material file!  To get the old default behavior, set this parameter to 0.35.");
  }

  if (p->isParameter("ACE Sea Level File") == true) {
    auto const filename = p->get<std::string>("ACE Sea Level File");
    sea_level_          = vectorFromFile(filename);
  }
  if (p->isParameter("ACE Time File") == true) {
    auto const filename = p->get<std::string>("ACE Time File");
    time_               = vectorFromFile(filename);
  }
  ALBANY_ASSERT(
      time_.size() == sea_level_.size(),
      "*** ERROR: Number of times and number of sea level values "
      "must match.");
  if (p->isParameter("ACE Z Depth File") == true) {
    auto const filename     = p->get<std::string>("ACE Z Depth File");
    z_above_mean_sea_level_ = vectorFromFile(filename);
  }
  if (p->isParameter("ACE_Porosity File") == true) {
    auto const filename = p->get<std::string>("ACE_Porosity File");
    porosity_from_file_ = vectorFromFile(filename);
    ALBANY_ASSERT(
        z_above_mean_sea_level_.size() == porosity_from_file_.size(),
        "*** ERROR: Number of z values and number of porosity values in "
        "ACE_Porosity File must match.");
  }
  if (p->isParameter("ACE Peat File") == true) {
    auto const filename = p->get<std::string>("ACE Peat File");
    peat_from_file_     = vectorFromFile(filename);
    ALBANY_ASSERT(
        z_above_mean_sea_level_.size() == peat_from_file_.size(),
        "*** ERROR: Number of z values and number of peat values in "
        "ACE Peat File must match.");
  }
  if (p->isParameter("ACE Air File") == true) {
    auto const filename = p->get<std::string>("ACE Air File");
    air_from_file_      = vectorFromFile(filename);
    ALBANY_ASSERT(
        z_above_mean_sea_level_.size() == air_from_file_.size(),
        "*** ERROR: Number of z values and number of air values in "
        "ACE Air File must match.");
  }

  // retrieve appropriate field name strings

  // retrieve appropriate field name strings
  std::string const cauchy_str     = field_name_map_["Cauchy_Stress"];
  std::string const Fp_str         = field_name_map_["Fp"];
  std::string const eqps_str       = field_name_map_["eqps"];
  std::string const yield_surf_str = field_name_map_["Yield_Surface"];
  std::string const j2_stress_str  = field_name_map_["J2_Stress"];
  std::string const tilt_angle_str = field_name_map_["Tilt_Angle"];
  std::string const source_str     = field_name_map_["Mechanical_Source"];
  std::string const F_str          = field_name_map_["F"];
  std::string const J_str          = field_name_map_["J"];

  // failure indicators just for output
  std::string const yield_indicator_str        = field_name_map_["Yield_Indicator"];
  std::string const tensile_indicator_str      = field_name_map_["Tensile_Indicator"];
  std::string const strain_indicator_str       = field_name_map_["Strain_Indicator"];
  std::string const angle_indicator_str        = field_name_map_["Angle_Indicator"];
  std::string const displacement_indicator_str = field_name_map_["Displacement_Indicator"];

  // Elastic modulus used (for output)
  std::string const elastic_modulus_str = field_name_map_["Elastic_Modulus_Used"];

  // define the dependent fields
  setDependentField(F_str, dl->qp_tensor);
  setDependentField(J_str, dl->qp_scalar);
  setDependentField("Poissons Ratio", dl->qp_scalar);
  setDependentField("Elastic Modulus", dl->qp_scalar);
  setDependentField("Yield Strength", dl->qp_scalar);
  setDependentField("Hardening Modulus", dl->qp_scalar);
  setDependentField("ACE_Ice_Saturation", dl->qp_scalar);
  setDependentField("Delta Time", dl->workset_scalar);
  setDependentField("Displacement", dl->qp_vector);

  // define the evaluated fields
  setEvaluatedField("failure_state", dl->cell_scalar2);
  setEvaluatedField("cell_death", dl->cell_scalar2);
  setEvaluatedField("failure_modes", dl->qp_scalar);
  setEvaluatedField(cauchy_str, dl->qp_tensor);
  setEvaluatedField(Fp_str, dl->qp_tensor);
  setEvaluatedField(eqps_str, dl->qp_scalar);
  setEvaluatedField(yield_surf_str, dl->qp_scalar);
  setEvaluatedField(j2_stress_str, dl->qp_scalar);
  setEvaluatedField(tilt_angle_str, dl->qp_scalar);
  setEvaluatedField(yield_indicator_str, dl->qp_scalar);
  setEvaluatedField(tensile_indicator_str, dl->qp_scalar);
  setEvaluatedField(strain_indicator_str, dl->qp_scalar);
  setEvaluatedField(angle_indicator_str, dl->qp_scalar);
  setEvaluatedField(displacement_indicator_str, dl->qp_scalar);
  setEvaluatedField(elastic_modulus_str, dl->qp_scalar);
  
  if (have_temperature_ == true) {
    setDependentField("Temperature", dl->qp_scalar);
    setEvaluatedField(source_str, dl->qp_scalar);
  }

  // define the state variables

  addStateVariable(cauchy_str, dl->qp_tensor, "scalar", 0.0, false, p->get<bool>("Output Cauchy Stress", false));
  addStateVariable(Fp_str, dl->qp_tensor, "identity", 0.0, true, p->get<bool>("Output Fp", false));
  addStateVariable(eqps_str, dl->qp_scalar, "scalar", 0.0, true, p->get<bool>("Output eqps", false));
  addStateVariable(yield_surf_str, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Yield Surface", false));
  addStateVariable(j2_stress_str, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output J2 Stress", false));
  addStateVariable(tilt_angle_str, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Tilt Angle", false));
  addStateVariable(yield_indicator_str, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Yield Indicator", false));
  addStateVariable(tensile_indicator_str, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Tensile Indicator", false));
  addStateVariable(strain_indicator_str, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Strain Indicator", false));
  addStateVariable(angle_indicator_str, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Angle Indicator", false));
  addStateVariable(displacement_indicator_str, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Displacement Indicator", false));
  addStateVariable(elastic_modulus_str, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Elastic Modulus", false));
  addStateVariable("failure_state", dl->cell_scalar2, "scalar", 0.0, false, p->get<bool>("Output Failure State", false));
  addStateVariable("cell_death", dl->cell_scalar2, "scalar", 0.0, false, p->get<bool>("Output Cell Death", false));
  addStateVariable("failure_modes", dl->qp_scalar, "scalar", 0.0, true, p->get<bool>("Output Failure Modes", false));

  if (have_temperature_ == true) {
    addStateVariable("Temperature", dl->qp_scalar, "scalar", 0.0, true, p->get<bool>("Output Temperature", false));
    addStateVariable(source_str, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Mechanical Source", false));
  }
}

template <typename EvalT, typename Traits>
void
J2ErosionKernel<EvalT, Traits>::init(Workset& workset, FieldMap<ScalarT const>& dep_fields, FieldMap<ScalarT>& eval_fields)
{
  std::string const cauchy_str                 = field_name_map_["Cauchy_Stress"];
  std::string const Fp_str                     = field_name_map_["Fp"];
  std::string const eqps_str                   = field_name_map_["eqps"];
  std::string const ct_str                     = field_name_map_["Cumulative_Time"];
  std::string const yield_surf_str             = field_name_map_["Yield_Surface"];
  std::string const j2_stress_str              = field_name_map_["J2_Stress"];
  std::string const tilt_angle_str             = field_name_map_["Tilt_Angle"];
  std::string const source_str                 = field_name_map_["Mechanical_Source"];
  std::string const F_str                      = field_name_map_["F"];
  std::string const J_str                      = field_name_map_["J"];
  std::string const yield_indicator_str        = field_name_map_["Yield_Indicator"];
  std::string const tensile_indicator_str      = field_name_map_["Tensile_Indicator"];
  std::string const strain_indicator_str       = field_name_map_["Strain_Indicator"];
  std::string const angle_indicator_str        = field_name_map_["Angle_Indicator"];
  std::string const displacement_indicator_str = field_name_map_["Displacement_Indicator"];
  std::string const elastic_modulus_str        = field_name_map_["Elastic_Modulus_Used"];

  // extract dependent MDFields
  def_grad_          = *dep_fields[F_str];
  J_                 = *dep_fields[J_str];
  poissons_ratio_    = *dep_fields["Poissons Ratio"];
  elastic_modulus_   = *dep_fields["Elastic Modulus"];
  yield_strength_    = *dep_fields["Yield Strength"];
  hardening_modulus_ = *dep_fields["Hardening Modulus"];
  delta_time_        = *dep_fields["Delta Time"];
  ice_saturation_    = *dep_fields["ACE_Ice_Saturation"];
  displacement_      = *dep_fields["Displacement"];

  // extract evaluated MDFields
  stress_                 = *eval_fields[cauchy_str];
  Fp_                     = *eval_fields[Fp_str];
  eqps_                   = *eval_fields[eqps_str];
  yield_surf_             = *eval_fields[yield_surf_str];
  j2_stress_              = *eval_fields[j2_stress_str];
  tilt_angle_             = *eval_fields[tilt_angle_str];
  yield_indicator_        = *eval_fields[yield_indicator_str];
  tensile_indicator_      = *eval_fields[tensile_indicator_str];
  strain_indicator_       = *eval_fields[strain_indicator_str];
  angle_indicator_        = *eval_fields[angle_indicator_str];
  displacement_indicator_ = *eval_fields[displacement_indicator_str];
  elastic_modulus_used_   = *eval_fields[elastic_modulus_str];
  failed_                 = *eval_fields["failure_state"];
  dead_                   = *eval_fields["cell_death"];
  failure_modes_          = *eval_fields["failure_modes"];

  if (have_temperature_ == true) {
    source_      = *eval_fields[source_str];
    temperature_ = *dep_fields["Temperature"];
  }

  // get State Variables
  Fp_old_            = (*workset.stateArrayPtr)[Fp_str + "_old"];
  eqps_old_          = (*workset.stateArrayPtr)[eqps_str + "_old"];
  failure_modes_old_ = (*workset.stateArrayPtr)["failure_modes_old"];
  // Read death status from the workset.  The ACE solver populates this
  // at the start of each step from the prior converged state, and
  // J2Erosion's operator() writes into it live when new failures occur
  // during a Newton iteration.  The scatter evaluator reads it to skip
  // dead cells in assembly.
  has_failed_old_ = false;
  if (workset.death_status_vec != Teuchos::null) {
    death_status_vec_ = workset.death_status_vec;
    has_failed_old_   = true;
  }

  current_time_ = workset.current_time;

  // Seed failed_ (diagnostic, decimal-encoded per-mode counts) and
  // dead_ (binary cell-death flag) from the per-(cell, pt) failure-mode
  // bitmask converged at the previous step (failure_modes_old). The mask
  // is an STK-backed element state, so the discretization keeps it mapped
  // to the right cell across workset rebuilds (erosion).
  //
  // Each set bit at (cell, pt) contributes a decimal magnitude (1, 10,
  // 100, 1000, 10000) to failed_(cell, 0) so the encoded value decodes
  // back to per-mode trip counts across all pts of the cell. operator()
  // adds this fill's newly tripped bits on top.
  //
  // dead_(cell, 0) is 1.0 iff every pt in the cell has at least one bit
  // set, i.e. every integration point has failed in some way. This is the
  // predicate consumed by the ACE solver to populate death_status_vec.
  auto const num_cells = workset.numCells;
  for (auto cell = 0; cell < num_cells; ++cell) {
    double seed       = 0.0;
    bool   all_failed = true;
    for (auto pt = 0; pt < num_pts_; ++pt) {
      uint8_t const m = static_cast<uint8_t>(failure_modes_old_(cell, pt));
      if (m & 0x01) seed += 1.0;
      if (m & 0x02) seed += 10.0;
      if (m & 0x04) seed += 100.0;
      if (m & 0x08) seed += 1000.0;
      if (m & 0x10) seed += 10000.0;
      if (m == 0u) all_failed = false;
    }
    failed_(cell, 0) = seed;
    dead_(cell, 0)   = all_failed ? 1.0 : 0.0;
  }
}

// J2 nonlinear system
template <typename EvalT, minitensor::Index M = 1>
class J2ErosionNLS : public minitensor::Function_Base<J2ErosionNLS<EvalT, M>, typename EvalT::ScalarT, M>
{
  using S = typename EvalT::ScalarT;

 public:
  J2ErosionNLS(RealType sat_mod, RealType sat_exp, RealType eqps_old, S const& K, S const& smag, S const& mubar, S const& Y)
      : sat_mod_(sat_mod), sat_exp_(sat_exp), eqps_old_(eqps_old), K_(K), smag_(smag), mubar_(mubar), Y_(Y)
  {
  }

  constexpr static char const* const NAME{"J2 NLS"};

  using Base = minitensor::Function_Base<J2ErosionNLS<EvalT, M>, typename EvalT::ScalarT, M>;

  // Default value.
  template <typename T, minitensor::Index N>
  T
  value(minitensor::Vector<T, N> const& x)
  {
    return Base::value(*this, x);
  }

  // Explicit gradient.
  template <typename T, minitensor::Index N>
  minitensor::Vector<T, N>
  gradient(minitensor::Vector<T, N> const& x)
  {
    // Firewalls.
    minitensor::Index const dimension = x.get_dimension();

    ALBANY_EXPECT(dimension == Base::DIMENSION);

    // Variables that potentially have Albany::Traits sensitivity
    // information need to be handled by the peel functor so that
    // proper conversions take place.
    T const K     = peel<EvalT, T, N>()(K_);
    T const smag  = peel<EvalT, T, N>()(smag_);
    T const mubar = peel<EvalT, T, N>()(mubar_);
    T const Y     = peel<EvalT, T, N>()(Y_);

    // This is the actual computation of the gradient.
    minitensor::Vector<T, N> r(dimension);

    T const& X     = x(0);
    T const  alpha = eqps_old_ + SQ23 * X;
    T const  H     = K * alpha + sat_mod_ * (1.0 - std::exp(-sat_exp_ * alpha));
    T const  R     = smag - (2.0 * mubar * X + SQ23 * (Y + H));

    r(0) = R;

    return r;
  }

  // Default AD hessian.
  template <typename T, minitensor::Index N>
  minitensor::Tensor<T, N>
  hessian(minitensor::Vector<T, N> const& x)
  {
    return Base::hessian(*this, x);
  }

  // Constants.
  RealType const sat_mod_{0.0};
  RealType const sat_exp_{0.0};
  RealType const eqps_old_{0.0};

  // Inputs
  S const& K_;
  S const& smag_;
  S const& mubar_;
  S const& Y_;
};

namespace {
template <typename T>
T
E_fit_max(T x, RealType y)
{
  // UPDATE 3-5-25: changing fit to be the one derived from using purely experimental values, 
  // instead of one using *simulated recreations* of the experiments
  // x = ice saturation;   y = porosity 
  // EM_fit = -24.6901 + -167.6662*x + -25.9496*y + 819.0673*x*y 

  // -Overall R2: 0.5869 
  // -Expt pt R2: -32.3787 
  // -Bound pt R2: 0.7339 
  // -->Max val = 600.7614 MPa 

  return (-24.6901 + -167.6662*x + -25.9496*y + 819.0673*x*y) / (600.7614); 
}

template <typename T>
T
Y_fit_max(T const x, RealType const y)
{
  // UPDATE 3-5-25: changing fit to be the one derived from using purely experimental values, 
  // instead of one using *simulated recreations* of the experiments

  // // x = ice saturation;   y = porosity 
  // Y_fit = -0.0419 + -0.2972*x + -0.0418*y + 4.7013*x*y 

  // -Overall R2: 0.9175 
  // -Expt pt R2: -0.0225 
  // -Bound pt R2: 0.9688 
  // -->Max val = 4.3204 MPa 
  return (-0.0419*y  + -0.2972*x + -0.0418*y + 4.7013*x*y) / (4.3204);
}

template <typename T>
T
K_fit_min(T const x, RealType const y)
{
  // Wed 09/21/2022 w/ BCs
  // x = ice saturation;   y = porosity
  return (2.535e-01 + 2.575e-01 * y + 1.990 * x - 7.985 * y * x) / (-5.484);
}

template <typename T>
std::tuple<T, T, T>
unit_fit(T ice_saturation, RealType porosity)
{
  auto const critical_porosity       = 0.01;  // can't be zero
  auto const critical_ice_saturation = 0.01;
  auto const x                       = ice_saturation;
  auto const y                       = porosity;
  auto const xc                      = critical_porosity;
  auto const yc                      = critical_ice_saturation;
  T          E{1.0};
  T          Y{1.0};
  T          K{1.0};
  // Elyce: commenting out linear decrease to zero
  Y = Y_fit_max(x, y);
  E = E_fit_max(x, y);
  K = K_fit_min(x, y);

  // if (xc < x && yc < y) {
  //   Y = Y_fit_max(x, y);
  //   E = E_fit_max(x, y);
  //   K = K_fit_min(x, y);
  // } else if (x <= xc) {
  //   Y = Y_fit_max(xc, y) * x / xc;
  //   E = E_fit_max(xc, y) * x / xc;
  //   K = K_fit_min(xc, y) * x / xc;
  // } else if (y <= yc) {
  //   Y = Y_fit_max(x, yc) * y / yc;
  //   E = E_fit_max(x, yc) * y / yc;
  //   K = K_fit_min(x, yc) * y / yc;
  // } else if (x <= xc && y <= yc) {
  //   Y = Y_fit_max(xc, yc) * x * y / xc / yc;
  //   E = E_fit_max(xc, yc) * x * y / xc / yc;
  //   K = K_fit_min(xc, yc) * x * y / xc / yc;
  // }
  return std::make_tuple(E, Y, K);
}

template <typename T>
T
safe_quotient(T num, T den)
{
  if (den == 0.0) return den;
  return num / den;
}

}  // anonymous namespace

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
J2ErosionKernel<EvalT, Traits>::operator()(int cell, int pt) const
{
  constexpr minitensor::Index MAX_DIM{3};
  using Tensor = minitensor::Tensor<ScalarT, MAX_DIM>;
  using Vector = minitensor::Vector<ScalarT, MAX_DIM>;

  // Element death: if this element was previously marked as failed,
  // set stress to zero and preserve old internal states.  The scatter
  // evaluator will skip this element entirely, so the stress value
  // here is only for output purposes.
  if (has_failed_old_ && (*death_status_vec_)[cell] > 0.0) {
    for (int i(0); i < num_dims_; ++i) {
      for (int j(0); j < num_dims_; ++j) {
        stress_(cell, pt, i, j) = 0.0;
      }
    }
    eqps_(cell, pt) = eqps_old_(cell, pt);
    for (int i{0}; i < num_dims_; ++i) {
      for (int j{0}; j < num_dims_; ++j) {
        Fp_(cell, pt, i, j) = ScalarT(Fp_old_(cell, pt, i, j));
      }
    }
    yield_surf_(cell, pt)             = 0.0;
    j2_stress_(cell, pt)              = 0.0;
    tilt_angle_(cell, pt)             = 0.0;
    yield_indicator_(cell, pt)        = 0.0;
    tensile_indicator_(cell, pt)      = 0.0;
    strain_indicator_(cell, pt)       = 0.0;
    angle_indicator_(cell, pt)        = 0.0;
    displacement_indicator_(cell, pt) = 0.0;
    elastic_modulus_used_(cell, pt)        = 0.0;
    // Carry the failure-mode bitmask forward unchanged: a dead cell never
    // trips new bits, but the state must still be written every fill.
    failure_modes_(cell, pt) = failure_modes_old_(cell, pt);
    return;
  }

  Tensor       F(num_dims_);
  Tensor const I(minitensor::eye<ScalarT, MAX_DIM>(num_dims_));
  Tensor       sigma(num_dims_);
  Vector       displacement(num_dims_);

  auto const coords       = this->model_.getCoordVecField();
  auto const height       = Sacado::Value<ScalarT>::eval(coords(cell, pt, 2));
  auto const current_time = current_time_;
  auto const sea_level    = sea_level_.size() > 0 ? interpolateVectors(time_, sea_level_, current_time) : -999.0;

  ScalarT const ice_saturation = ice_saturation_(cell, pt);

  auto const peat     = peat_from_file_.size() > 0 ? interpolateVectors(z_above_mean_sea_level_, peat_from_file_, height) : 0.0;
  auto const porosity = porosity_from_file_.size() > 0 ? interpolateVectors(z_above_mean_sea_level_, porosity_from_file_, height) : bulk_porosity_;
  // IKT, 2/17/2024: added air for specification of snow
  // TODO: work this variable into implementation
  auto const air = air_from_file_.size() > 0 ? interpolateVectors(z_above_mean_sea_level_, air_from_file_, height) : 0.0;
  // std::cout << "IKT J2Erosion air = " << air << "\n";
  ScalarT ne{1.0};
  ScalarT ny{1.0};
  ScalarT nk{1.0};

  std::tie(ne, ny, nk) = unit_fit(ice_saturation, porosity);

  ScalarT E = elastic_modulus_(cell, pt) * ne;
  ScalarT Y = yield_strength_(cell, pt) * ny;
  ScalarT K = hardening_modulus_(cell, pt) * nk;

  E = std::max(E, residual_elastic_modulus_);
  Y = std::max(Y, soil_yield_strength_);

  Y = std::max(Y, 0.0);
  E = std::max(E, 0.0);

  if (ice_saturation < ice_saturation_material_fit_truncation_){
    E = residual_elastic_modulus_;
    Y = soil_yield_strength_;
  }

  auto&& delta_time = delta_time_(0);
  auto&& failed     = failed_(cell, 0);

  // TODO Phase B: derive from "*-erodible" side-set membership.
  bool const is_erodible = false;

  auto strain_limit = strain_limit_;
  if ((porosity < 0.99) && (strain_limit > 0.0)) {
    strain_limit = 1.0 + peat;
    // strain_limit = std::max(strain_limit, 1.04);
    strain_limit = std::max(
        strain_limit, strain_limit_);  // elyce 8-23-24: changed this so it actually uses the input deck value specified, otherwise that doesn't get used
  }

  // Make the elements exposed to ocean "weaker"
  auto tensile_strength = tensile_strength_;
  if ((is_erodible == true) && (height <= sea_level)) {
    Y = Y / (Y_weakening_factor_);
    E = E / (E_weakening_factor_);
    if (strain_limit > 0.0) {
      strain_limit = 1.0 + ((strain_limit - 1.0) / SL_weakening_factor_);
    }
  }

  // Save the elastic modulus used for output: 
  elastic_modulus_used_(cell, pt) = E;

  ScalarT const nu    = poissons_ratio_(cell, pt);
  ScalarT const kappa = E / (3.0 * (1.0 - 2.0 * nu));
  ScalarT const mu    = E / (2.0 * (1.0 + nu));
  ScalarT const J1    = J_(cell, pt);
  ScalarT const Jm23  = 1.0 / std::cbrt(J1 * J1);

  // fill local tensors
  F.fill(def_grad_, cell, pt, 0, 0);
  displacement.fill(displacement_, cell, pt, 0);

  // Mechanical deformation gradient
  auto Fm = Tensor(F);
  if (have_temperature_) {
    ScalarT dtemp           = temperature_(cell, pt) - ref_temperature_;
    ScalarT thermal_stretch = std::exp(expansion_coeff_ * dtemp);
    Fm /= thermal_stretch;
  }

  Tensor Fpn(num_dims_);

  for (int i{0}; i < num_dims_; ++i) {
    for (int j{0}; j < num_dims_; ++j) {
      Fpn(i, j) = ScalarT(Fp_old_(cell, pt, i, j));
    }
  }

  // compute trial state
  Tensor const  Fpinv = minitensor::inverse(Fpn);
  Tensor const  Cpinv = Fpinv * minitensor::transpose(Fpinv);
  Tensor const  be    = Jm23 * Fm * Cpinv * minitensor::transpose(Fm);
  Tensor        s     = mu * minitensor::dev(be);
  ScalarT const mubar = minitensor::trace(be) * mu / (num_dims_);

  // check yield condition
  ScalarT const smag = minitensor::norm(s);
  ScalarT const ys   = SQ23 * (Y + K * eqps_old_(cell, pt) + sat_mod_ * (1.0 - std::exp(-sat_exp_ * eqps_old_(cell, pt))));
  ScalarT const f    = smag - ys;

  // update for output
  yield_surf_(cell, pt)      = ys;
  j2_stress_(cell, pt)       = smag;
  yield_indicator_(cell, pt) = safe_quotient(smag, ys);

  // Compute tilt angle only when needed for the critical angle failure check.
  // polar_rotation is expensive (SVD-based) and not worth computing for output alone.
  auto const Fval = Sacado::Value<decltype(F)>::eval(F);
  RealType   theta{0.0};
  if (critical_angle_ > 0.0) {
    auto const Q      = minitensor::polar_rotation(Fval);
    auto       cosine = 0.5 * (minitensor::trace(Q) - 1.0);
    cosine            = cosine > 1.0 ? 1.0 : cosine;
    cosine            = cosine < -1.0 ? -1.0 : cosine;
    theta             = std::acos(cosine);
  }
  tilt_angle_(cell, pt) = theta;

  RealType constexpr yield_tolerance = 1.0e-12;
  bool const yielded                 = f > yield_tolerance;

  if (yielded == true) {
    // Use minimization equivalent to return mapping
    using ValueT = typename Sacado::ValueType<ScalarT>::type;
    using NLS    = J2ErosionNLS<EvalT>;

    constexpr minitensor::Index nls_dim{NLS::DIMENSION};

    using MIN  = minitensor::Minimizer<ValueT, nls_dim>;
    using STEP = minitensor::NewtonStep<NLS, ValueT, nls_dim>;

    MIN  minimizer;
    STEP step;
    NLS  j2nls(sat_mod_, sat_exp_, eqps_old_(cell, pt), K, smag, mubar, Y);

    minitensor::Vector<ScalarT, nls_dim> x;

    x(0) = 0.0;

    LCM::MiniSolver<MIN, STEP, NLS, EvalT, nls_dim> mini_solver(minimizer, step, j2nls, x);

    ScalarT const alpha = eqps_old_(cell, pt) + SQ23 * x(0);
    ScalarT const H     = K * alpha + sat_mod_ * (1.0 - exp(-sat_exp_ * alpha));
    ScalarT const dgam  = x(0);

    // plastic direction
    Tensor const N = (1 / smag) * s;

    // update s
    s -= 2 * mubar * dgam * N;

    // update eqps
    eqps_(cell, pt) = alpha;

    // mechanical source
    if (have_temperature_ == true && delta_time_(0) > 0.0) {
      source_(cell, pt) = (SQ23 * dgam / delta_time_(0) * (Y + H + temperature_(cell, pt))) / (density_ * heat_capacity_);
    }

    // exponential map to get Fpnew
    Tensor const A     = dgam * N;
    Tensor const expA  = minitensor::exp(A);
    Tensor const Fpnew = expA * Fpn;

    for (int i{0}; i < num_dims_; ++i) {
      for (int j{0}; j < num_dims_; ++j) {
        Fp_(cell, pt, i, j) = Fpnew(i, j);
      }
    }
  } else {
    eqps_(cell, pt) = eqps_old_(cell, pt);

    if (have_temperature_ == true) source_(cell, pt) = 0.0;

    for (int i{0}; i < num_dims_; ++i) {
      for (int j{0}; j < num_dims_; ++j) {
        Fp_(cell, pt, i, j) = Fpn(i, j);
      }
    }
  }

  // compute pressure
  ScalarT const p = 0.5 * kappa * (J_(cell, pt) - 1.0 / (J_(cell, pt)));

  // compute stress
  sigma = p * I + s / J_(cell, pt);

  for (int i(0); i < num_dims_; ++i) {
    for (int j(0); j < num_dims_; ++j) {
      stress_(cell, pt, i, j) = sigma(i, j);
    }
  }

  // Per-(cell, pt) OR-accumulation: add each mode's decimal magnitude to
  // `failed` exactly the first time that mode trips at this integration
  // point. `mask` starts from the bitmask converged at the previous step
  // (failure_modes_old) and is written back to failure_modes_ below, so a
  // criterion that keeps tripping at the same point contributes its
  // magnitude once and only once across the run.
  //
  // When disable_erosion_ is true the cell cannot die regardless of how
  // criteria trip, so skip both the bitmask update and the failed
  // accumulator. Indicators above are still computed for diagnostics.
  uint8_t mask = static_cast<uint8_t>(failure_modes_old_(cell, pt));
  auto    trip = [&](bool fired, uint8_t bit, double magnitude) {
    if (fired == false) return;
    if (disable_erosion_) return;
    if ((mask & bit) != 0u) return;
    mask = static_cast<uint8_t>(mask | bit);
    failed += magnitude;
  };

  // Hack for tensile strength
  if (tensile_strength > 0) {
    auto          sig = Sacado::Value<decltype(sigma)>::eval(sigma);
    decltype(sig) V(num_dims_);
    decltype(sig) S(num_dims_);
    std::tie(V, S)               = minitensor::eig_sym(sig);
    auto const Smax              = std::max(S(0, 0), std::max(S(1, 1), S(2, 2)));
    bool const tension_failure   = Smax >= tensile_strength;
    tensile_indicator_(cell, pt) = safe_quotient(Smax, tensile_strength);
    trip(tension_failure, 0x01, 1.0);
  }

  // Hack for strain limit
  if (strain_limit > 0.0) {
    decltype(Fval) Cval           = minitensor::transpose(Fval) * Fval ; //  was Fval * Fval^T before, but should be Fval^T * Fval, as per convo with Alejandro ~3/5/25
    auto const     Jval           = minitensor::det(Fval);
    auto const     Jm23val        = 1.0 / std::cbrt(Jval * Jval);
    decltype(Fval) Cdevval        = Jm23val * Cval;
    auto const     distortion     = minitensor::norm(Cdevval) / std::sqrt(3.0);
    bool const     strain_failure = distortion >= strain_limit;
    strain_indicator_(cell, pt)   = safe_quotient(distortion, strain_limit);
    trip(strain_failure, 0x02, 10.0);
  }

  // Determine if critical stress is exceeded
  trip(yielded, 0x04, 100.0);

  // Determine if kinematic failure occurred
  auto critical_angle = critical_angle_;
  if (height <= sea_level) {
    critical_angle = 1.0 * critical_angle_;
  }
  if (critical_angle > 0.0) {
    auto const theta_abs = std::abs(theta);
    trip(theta_abs >= critical_angle, 0x08, 1000.0);
    angle_indicator_(cell, pt) = safe_quotient(theta_abs, critical_angle);
  }
  auto const disp_val               = Sacado::Value<decltype(displacement)>::eval(displacement);
  auto const displacement_norm      = minitensor::norm(disp_val);
  displacement_indicator_(cell, pt) = safe_quotient(displacement_norm, maximum_displacement_);
  bool const disp_failure =
      (maximum_displacement_ > 0.0) && (displacement_norm > maximum_displacement_);
  trip(disp_failure, 0x10, 10000.0);

  // Persist this fill's updated bitmask (old | newly tripped bits) to the
  // STK-backed state, so it follows the cell across workset rebuilds.
  failure_modes_(cell, pt) = static_cast<double>(mask);

  // Live death propagation: when the last pt of a cell is processed,
  // check whether every pt in this cell now has at least one failure bit
  // set. If so, mark the cell dead so the scatter evaluator and
  // fixOrphanNodesForElementDeath see it as dead in this same fill.
  // Without this, Newton can push partially-failed cells (now condemned
  // by the last-pt trip) into regimes where they would fail more, while
  // scatter keeps assembling their nonphysical stress contributions —
  // breaking convergence after a death cascade. Points are processed in
  // order, so at the last pt every failure_modes_(cell, p) is current.
  //
  // When disable_erosion_ is true, trips above are no-ops, so no pt
  // ever has a bit set and the predicate cannot trigger.
  if (pt == num_pts_ - 1 && has_failed_old_) {
    bool all_failed = true;
    for (auto p = 0; p < num_pts_; ++p) {
      if (static_cast<uint8_t>(Sacado::Value<ScalarT>::eval(failure_modes_(cell, p))) == 0u) {
        all_failed = false;
        break;
      }
    }
    if (all_failed && (*death_status_vec_)[cell] == 0.0) {
      (*death_status_vec_)[cell] = 1.0;
      dead_(cell, 0)             = 1.0;
    }
  }
}
}  // namespace LCM
