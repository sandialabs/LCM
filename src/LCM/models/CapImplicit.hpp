// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_CapImplicit_hpp)
#define LCM_CapImplicit_hpp

#include <MiniTensor.h>

#include "KernelConstitutiveModel.hpp"

namespace LCM {

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

  typedef typename Sacado::mpl::apply<FadType, ScalarT>::type  DFadType;
  typedef typename Sacado::mpl::apply<FadType, DFadType>::type D2FadType;

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
  ScalarField tangent_;

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
  // yield function
  template <typename T>
  T
  compute_f(minitensor::Tensor<T>& sigma, minitensor::Tensor<T>& alpha, T& kappa) const;

  // plastic potential
  template <typename T>
  T
  compute_g(minitensor::Tensor<T>& sigma, minitensor::Tensor<T>& alpha, T& kappa) const;

  // unknown variable value list
  std::vector<ScalarT>
  initialize(
      minitensor::Tensor<ScalarT>& sigmaVal,
      minitensor::Tensor<ScalarT>& alphaVal,
      ScalarT&                     kappaVal,
      ScalarT&                     dgammaVal) const;

  // local iteration jacobian
  void
  compute_ResidJacobian(
      std::vector<ScalarT> const&         XXVal,
      std::vector<ScalarT>&               R,
      std::vector<ScalarT>&               dRdX,
      const minitensor::Tensor<ScalarT>&  sigmaVal,
      const minitensor::Tensor<ScalarT>&  alphaVal,
      ScalarT const&                      kappaVal,
      minitensor::Tensor4<ScalarT> const& Celastic,
      bool                                kappa_flag) const;

  // derivatives via AD
  minitensor::Tensor<ScalarT>
  compute_dfdsigma(std::vector<ScalarT> const& XX) const;

  ScalarT
  compute_dfdkappa(std::vector<ScalarT> const& XX) const;

  minitensor::Tensor<ScalarT>
  compute_dgdsigma(std::vector<ScalarT> const& XX) const;

  minitensor::Tensor<DFadType>
  compute_dgdsigma(std::vector<DFadType> const& XX) const;

  // hardening functions
  template <typename T>
  T
  compute_Galpha(T J2_alpha) const;

  template <typename T>
  minitensor::Tensor<T>
  compute_halpha(minitensor::Tensor<T> const& dgdsigma, T const J2_alpha) const;

  template <typename T>
  T
  compute_dedkappa(T const kappa) const;

  template <typename T>
  T
  compute_hkappa(T const I1_dgdsigma, T const dedkappa) const;

  // elasto-plastic tangent modulus
  minitensor::Tensor4<ScalarT>
  compute_Cep(
      minitensor::Tensor4<ScalarT>& Celastic,
      minitensor::Tensor<ScalarT>&  sigma,
      minitensor::Tensor<ScalarT>&  alpha,
      ScalarT&                      kappa,
      ScalarT&                      dgamma) const;
};

template <typename EvalT, typename Traits>
class CapImplicit : public LCM::KernelConstitutiveModel<EvalT, Traits, CapImplicitKernel<EvalT, Traits>>
{
 public:
  CapImplicit(Teuchos::ParameterList* p, const Teuchos::RCP<Albany::Layouts>& dl);
};

}  // namespace LCM

#endif  // LCM_CapImplicit_hpp
