// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_CapExplicit_hpp)
#define LCM_CapExplicit_hpp

#include <MiniTensor.h>

#include "KernelConstitutiveModel.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
struct CapExplicitKernel : public ParallelKernel<EvalT, Traits>
{
  ///
  /// Constructor
  ///
  CapExplicitKernel(ConstitutiveModel<EvalT, Traits>& model, Teuchos::ParameterList* p, Teuchos::RCP<Albany::Layouts> const& dl);

  ///
  /// No copy constructor
  ///
  CapExplicitKernel(CapExplicitKernel const&) = delete;

  ///
  /// No copy assignment
  ///
  CapExplicitKernel&
  operator=(CapExplicitKernel const&) = delete;

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

 private:
  ScalarT
  compute_f(
      minitensor::Tensor<ScalarT>& sigma,
      minitensor::Tensor<ScalarT>& alpha,
      ScalarT& kappa) const;

  minitensor::Tensor<ScalarT>
  compute_dfdsigma(
      minitensor::Tensor<ScalarT>& sigma,
      minitensor::Tensor<ScalarT>& alpha,
      ScalarT& kappa) const;

  minitensor::Tensor<ScalarT>
  compute_dgdsigma(
      minitensor::Tensor<ScalarT>& sigma,
      minitensor::Tensor<ScalarT>& alpha,
      ScalarT& kappa) const;

  ScalarT
  compute_dfdkappa(
      minitensor::Tensor<ScalarT>& sigma,
      minitensor::Tensor<ScalarT>& alpha,
      ScalarT& kappa) const;

  ScalarT
  compute_Galpha(ScalarT& J2_alpha) const;

  minitensor::Tensor<ScalarT>
  compute_halpha(
      minitensor::Tensor<ScalarT>& dgdsigma,
      ScalarT& J2_alpha) const;

  ScalarT
  compute_dedkappa(ScalarT& kappa) const;
};

template <typename EvalT, typename Traits>
class CapExplicit : public LCM::KernelConstitutiveModel<EvalT, Traits, CapExplicitKernel<EvalT, Traits>>
{
 public:
  CapExplicit(Teuchos::ParameterList* p, const Teuchos::RCP<Albany::Layouts>& dl);
};

}  // namespace LCM

#endif  // LCM_CapExplicit_hpp
