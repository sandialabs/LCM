// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include "ACEcommon.hpp"
#include "Albany_STKDiscretization.hpp"
#include "J2Erosion.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
J2ErosionKernel<EvalT, Traits>::J2ErosionKernel(
    ConstitutiveModel<EvalT, Traits>&    model,
    Teuchos::ParameterList*              p,
    Teuchos::RCP<Albany::Layouts> const& dl)
    : BaseKernel(model)
{
  this->setIntegrationPointLocationFlag(true);

  // Baseline constants
  sat_mod_                  = p->get<RealType>("Saturation Modulus", 0.0);
  sat_exp_                  = p->get<RealType>("Saturation Exponent", 0.0);
  bulk_porosity_            = p->get<RealType>("ACE Bulk Porosity", 0.0);
  critical_angle_           = p->get<RealType>("ACE Critical Angle", 0.0);
  Y_weakening_factor_       = p->get<RealType>("ACE Y Weakening Factor", 1.0);
  E_weakening_factor_       = p->get<RealType>("ACE E Weakening Factor", 1.0);
  SL_weakening_factor_      = p->get<RealType>("ACE SL Weakening Factor", 1.0);
  soil_yield_strength_      = p->get<RealType>("ACE Soil Yield Strength", 0.0);
  residual_elastic_modulus_ = p->get<RealType>("ACE Residual Elastic Modulus", 0.0);
  tensile_strength_         = p->get<RealType>("ACE Tensile Strength", 0.0);
  strain_limit_             = p->get<RealType>("ACE Strain Limit", 0.0);

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
  setEvaluatedField(cauchy_str, dl->qp_tensor);
  setEvaluatedField(Fp_str, dl->qp_tensor);
  setEvaluatedField(eqps_str, dl->qp_scalar);
  setEvaluatedField(yield_surf_str, dl->qp_scalar);
  setEvaluatedField(j2_stress_str, dl->qp_scalar);
  setEvaluatedField(tilt_angle_str, dl->qp_scalar);
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

  if (have_temperature_ == true) {
    addStateVariable("Temperature", dl->qp_scalar, "scalar", 0.0, true, p->get<bool>("Output Temperature", false));
    addStateVariable(source_str, dl->qp_scalar, "scalar", 0.0, false, p->get<bool>("Output Mechanical Source", false));
  }

  addStateVariable(
      "failure_state", dl->cell_scalar2, "scalar", 0.0, false, p->get<bool>("Output Failure State", false));
}

template <typename EvalT, typename Traits>
void
J2ErosionKernel<EvalT, Traits>::init(
    Workset&                 workset,
    FieldMap<ScalarT const>& dep_fields,
    FieldMap<ScalarT>&       eval_fields)
{
  std::string cauchy_str     = field_name_map_["Cauchy_Stress"];
  std::string Fp_str         = field_name_map_["Fp"];
  std::string eqps_str       = field_name_map_["eqps"];
  std::string yield_surf_str = field_name_map_["Yield_Surface"];
  std::string j2_stress_str  = field_name_map_["J2_Stress"];
  std::string tilt_angle_str = field_name_map_["Tilt_Angle"];
  std::string source_str     = field_name_map_["Mechanical_Source"];
  std::string F_str          = field_name_map_["F"];
  std::string J_str          = field_name_map_["J"];

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
  stress_     = *eval_fields[cauchy_str];
  Fp_         = *eval_fields[Fp_str];
  eqps_       = *eval_fields[eqps_str];
  yield_surf_ = *eval_fields[yield_surf_str];
  j2_stress_  = *eval_fields[j2_stress_str];
  tilt_angle_ = *eval_fields[tilt_angle_str];
  failed_     = *eval_fields["failure_state"];

  if (have_temperature_ == true) {
    source_      = *eval_fields[source_str];
    temperature_ = *dep_fields["Temperature"];
  }

  // get State Variables
  Fp_old_   = (*workset.stateArrayPtr)[Fp_str + "_old"];
  eqps_old_ = (*workset.stateArrayPtr)[eqps_str + "_old"];

  auto& disc                    = *workset.disc;
  auto& stk_disc                = dynamic_cast<Albany::STKDiscretization&>(disc);
  auto& mesh_struct             = *(stk_disc.getSTKMeshStruct());
  auto& field_cont              = *(mesh_struct.getFieldContainer());
  have_cell_boundary_indicator_ = field_cont.hasCellBoundaryIndicatorField();

  if (have_cell_boundary_indicator_ == true) {
    cell_boundary_indicator_ = workset.cell_boundary_indicator;
    ALBANY_ASSERT(cell_boundary_indicator_.is_null() == false);
  }

  current_time_ = workset.current_time;

  auto const num_cells = workset.numCells;
  for (auto cell = 0; cell < num_cells; ++cell) {
    failed_(cell, 0) = 0.0;
  }
}

// J2 nonlinear system
template <typename EvalT, minitensor::Index M = 1>
class J2ErosionNLS : public minitensor::Function_Base<J2ErosionNLS<EvalT, M>, typename EvalT::ScalarT, M>
{
  using S = typename EvalT::ScalarT;

 public:
  J2ErosionNLS(
      RealType sat_mod,
      RealType sat_exp,
      RealType eqps_old,
      S const& K,
      S const& smag,
      S const& mubar,
      S const& Y)
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
  // Wed 09/21/2022 w/ BCs
  // x = ice saturation;   y = porosity
  return (-6.4724 - 6.50845 * y - 52.9043 * x + 345.809 * y * x) / (279.9238);
}

template <typename T>
T
Y_fit_max(T const x, RealType const y)
{
  // Wed 09/21/2022 w/ BCs
  // x = ice saturation;   y = porosity
  return (-4.374e-02 - 4.337e-02 * y - 3.610e-01 * x + 4.639 * y * x) / (4.19089);
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
  if (xc < x && yc < y) {
    Y = Y_fit_max(x, y);
    E = E_fit_max(x, y);
    K = K_fit_min(x, y);
  } else if (x <= xc) {
    Y = Y_fit_max(xc, y) * x / xc;
    E = E_fit_max(xc, y) * x / xc;
    K = K_fit_min(xc, y) * x / xc;
  } else if (y <= yc) {
    Y = Y_fit_max(x, yc) * y / yc;
    E = E_fit_max(x, yc) * y / yc;
    K = K_fit_min(x, yc) * y / yc;
  } else if (x <= xc && y <= yc) {
    Y = Y_fit_max(xc, yc) * x * y / xc / yc;
    E = E_fit_max(xc, yc) * x * y / xc / yc;
    K = K_fit_min(xc, yc) * x * y / xc / yc;
  }
  return std::make_tuple(E, Y, K);
}

}  // anonymous namespace

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
J2ErosionKernel<EvalT, Traits>::operator()(int cell, int pt) const
{
  constexpr minitensor::Index MAX_DIM{3};
  using Tensor = minitensor::Tensor<ScalarT, MAX_DIM>;
  using Vector = minitensor::Vector<ScalarT, MAX_DIM>;
  Tensor       F(num_dims_);
  Tensor const I(minitensor::eye<ScalarT, MAX_DIM>(num_dims_));
  Tensor       sigma(num_dims_);
  Vector       displacement(num_dims_);

  auto const coords       = this->model_.getCoordVecField();
  auto const height       = Sacado::Value<ScalarT>::eval(coords(cell, pt, 2));
  auto const current_time = current_time_;
  auto const sea_level    = sea_level_.size() > 0 ? interpolateVectors(time_, sea_level_, current_time) : -999.0;

  ScalarT const ice_saturation = ice_saturation_(cell, pt);

  auto const peat =
      peat_from_file_.size() > 0 ? interpolateVectors(z_above_mean_sea_level_, peat_from_file_, height) : 0.0;
  auto const porosity = porosity_from_file_.size() > 0 ?
                            interpolateVectors(z_above_mean_sea_level_, porosity_from_file_, height) :
                            bulk_porosity_;
  ScalarT    ne{1.0};
  ScalarT    ny{1.0};
  ScalarT    nk{1.0};

  std::tie(ne, ny, nk) = unit_fit(ice_saturation, porosity);

  ScalarT E = elastic_modulus_(cell, pt) * ne;
  ScalarT Y = yield_strength_(cell, pt) * ny;
  ScalarT K = hardening_modulus_(cell, pt) * nk;

  E = std::max(E, residual_elastic_modulus_);
  Y = std::max(Y, soil_yield_strength_);

  Y = std::max(Y, 0.0);
  E = std::max(E, 0.0);

  auto&& delta_time = delta_time_(0);
  auto&& failed     = failed_(cell, 0);

  auto const cell_bi        = have_cell_boundary_indicator_ == true ? *(cell_boundary_indicator_[cell]) : 0.0;
  auto const is_at_boundary = cell_bi == 1.0;
  auto const is_erodible    = cell_bi == 2.0;

  auto strain_limit = strain_limit_;
  if (porosity < 0.99) {
    strain_limit = 1.0 + peat;
    strain_limit = std::max(strain_limit, 1.04);
  }

  // Make the elements exposed to ocean "weaker"
  auto tensile_strength = tensile_strength_;
  if ((is_erodible == true) && (height <= sea_level)) {
    Y            = Y / (Y_weakening_factor_);
    E            = E / (E_weakening_factor_);
    strain_limit = 1.0 + ((strain_limit - 1.0) / SL_weakening_factor_);
  }

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
  ScalarT const ys =
      SQ23 * (Y + K * eqps_old_(cell, pt) + sat_mod_ * (1.0 - std::exp(-sat_exp_ * eqps_old_(cell, pt))));
  ScalarT const f = smag - ys;

  // update for output
  yield_surf_(cell, pt) = ys;
  j2_stress_(cell, pt)  = smag;
  auto const Fval       = Sacado::Value<decltype(F)>::eval(F);
  auto const Q          = minitensor::polar_rotation(Fval);
  auto       cosine     = 0.5 * (minitensor::trace(Q) - 1.0);
  cosine                = cosine > 1.0 ? 1.0 : cosine;
  cosine                = cosine < -1.0 ? -1.0 : cosine;
  auto const theta      = std::acos(cosine);
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
      source_(cell, pt) =
          (SQ23 * dgam / delta_time_(0) * (Y + H + temperature_(cell, pt))) / (density_ * heat_capacity_);
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

  // Hack for tensile strength
  if (tensile_strength > 0) {
    auto          sig = Sacado::Value<decltype(sigma)>::eval(sigma);
    decltype(sig) V(num_dims_);
    decltype(sig) S(num_dims_);
    std::tie(V, S) = minitensor::eig_sym(sig);
    bool const tension_failure =
        S(0, 0) >= tensile_strength || S(1, 1) >= tensile_strength || S(2, 2) >= tensile_strength;
    if (tension_failure == true) {
      failed += 1.0;
    }
  }

  // Hack for strain limit
  if (strain_limit > 0.0) {
    decltype(Fval) Cval           = Fval * minitensor::transpose(Fval);
    auto const     Jval           = minitensor::det(Fval);
    auto const     Jm23val        = 1.0 / std::cbrt(Jval * Jval);
    decltype(Fval) Cdevval        = Jm23val * Cval;
    auto const     distortion     = minitensor::norm(Cdevval) / std::sqrt(3.0);
    bool const     strain_failure = distortion >= strain_limit;
    if (strain_failure == true) {
      failed += 1.0;
    }
  }

  // Determine if critical stress is exceeded
  if (yielded == true) {
    failed += 1.0;
    // std::cout << "Cell " << cell << " pt " << pt << " :: yielded \n";
  }

  // Determine if kinematic failure occurred
  auto critical_angle = critical_angle_;
  if (height <= sea_level) {
    critical_angle = 1.0 * critical_angle_;
  }
  if (critical_angle > 0.0) {
    if (std::abs(theta) >= critical_angle) {
      failed += 1.0;
      // std::cout << "Cell " << cell << " pt " << pt << " :: critical angle \n";
    }
  }
  auto const maximum_displacement = 0.35;  // [m]
  auto const displacement_norm    = minitensor::norm(displacement);
  if (displacement_norm > maximum_displacement) {
    failed += 8.0;
    // std::cout << "Cell " << cell << " pt " << pt << " :: max displacement \n";
  }
}
}  // namespace LCM
