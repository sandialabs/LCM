// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_CapImplicit_hpp)
#define LCM_CapImplicit_hpp

#include <MiniTensor.h>

#include "KernelConstitutiveModel.hpp"

namespace LCM {

///
/// Three-invariant cap plasticity model with kinematic + isotropic
/// (volumetric) hardening, integrated by a fully-implicit backward-Euler
/// return map on the 13-unknown system (6 stress + 5 deviatoric back
/// stress + cap parameter + plastic multiplier).  The local Newton solve
/// is driven by LCM::MiniSolver, which keeps the iteration on value-only
/// arithmetic and re-derives global Sacado-FAD sensitivities once via the
/// implicit-function theorem (see MiniNonlinearSolver_Def.hpp).
///
/// All helper functions (yield surface f, plastic potential g, its
/// gradient dg/dsigma, kinematic hardening h_alpha, isotropic hardening
/// h_kappa, etc.) live in the local CapImplicitNLS class in the _Def
/// file rather than as kernel members so that the std::vector + nested
/// Sacado::Fad::DFad code path the old LocalNonlinearSolver pattern
/// relied on is avoided -- that path corrupts the outer FAD info on
/// sigmaVal under gcc release optimization even when never executed.
///
template <typename EvalT, typename Traits>
struct CapImplicitKernel : public ParallelKernel<EvalT, Traits>
{
  ///
  /// Constructor
  ///
  CapImplicitKernel(ConstitutiveModel<EvalT, Traits>& model, Teuchos::ParameterList* p, Teuchos::RCP<Albany::Layouts> const& dl);

  ///
  /// No copy constructor
  ///
  CapImplicitKernel(CapImplicitKernel const&) = delete;

  ///
  /// No copy assignment
  ///
  CapImplicitKernel&
  operator=(CapImplicitKernel const&) = delete;

  using ScalarT          = typename EvalT::ScalarT;
  using ScalarField      = PHX::MDField<ScalarT>;
  using ConstScalarField = PHX::MDField<ScalarT const>;
  using BaseKernel       = ParallelKernel<EvalT, Traits>;
  using Workset          = typename BaseKernel::Workset;

  using BaseKernel::field_name_map_;
  using BaseKernel::num_dims_;
  using BaseKernel::num_pts_;

  using BaseKernel::addStateVariable;
  using BaseKernel::setDependentField;
  using BaseKernel::setEvaluatedField;

  using BaseKernel::compute_tangent_;
  using BaseKernel::nox_status_test_;

  // Dependent MDFields
  ConstScalarField def_grad_;
  ConstScalarField J_;
  ConstScalarField strain_;
  ConstScalarField elastic_modulus_;
  ConstScalarField poissons_ratio_;

  // Evaluated MDFields
  ScalarField stress_;
  ScalarField Fp_;
  ScalarField backStress_;
  ScalarField capParameter_;
  ScalarField eqps_;
  ScalarField volPlasticStrain_;

  // Old state arrays
  Albany::MDArray Fp_old_;
  Albany::MDArray stress_old_;
  Albany::MDArray strain_old_;
  Albany::MDArray backStress_old_;
  Albany::MDArray capParameter_old_;
  Albany::MDArray eqps_old_;
  Albany::MDArray volPlasticStrain_old_;

  // Finite deformation flag
  bool finite_deformation_;

  // Material parameters
  RealType A_;
  RealType D_;
  RealType C_;
  RealType theta_;
  RealType R_;
  RealType kappa0_;
  RealType W_;
  RealType D1_;
  RealType D2_;
  RealType calpha_;
  RealType psi_;
  RealType N_;
  RealType L_;
  RealType phi_;
  RealType Q_;

  void
  init(Workset& workset, FieldMap<ScalarT const>& dep_fields, FieldMap<ScalarT>& eval_fields);

  KOKKOS_INLINE_FUNCTION
  void
  operator()(int cell, int pt) const;
};

template <typename EvalT, typename Traits>
class CapImplicit : public LCM::KernelConstitutiveModel<EvalT, Traits, CapImplicitKernel<EvalT, Traits>>
{
 public:
  CapImplicit(Teuchos::ParameterList* p, const Teuchos::RCP<Albany::Layouts>& dl);
};

}  // namespace LCM

#endif  // LCM_CapImplicit_hpp
