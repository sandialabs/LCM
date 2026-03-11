// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(CapImplicitFD_hpp)
#define CapImplicitFD_hpp

#include <MiniTensor.h>

#include "Albany_Layouts.hpp"
#include "ConstitutiveModel.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
/// \brief CapPlasticity implicit finite deformation stress response
///
/// This evaluator computes stress based on a cap plasticity model
/// extended to the finite deformation regime using Technique A
/// (Logarithmic Strain-based Plasticity) with implicit integration.
///

template <typename EvalT, typename Traits>
class CapImplicitFD : public LCM::ConstitutiveModel<EvalT, Traits>
{
 public:
  using Base        = LCM::ConstitutiveModel<EvalT, Traits>;
  using DepFieldMap = typename Base::DepFieldMap;
  using FieldMap    = typename Base::FieldMap;

  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;
  typedef typename Sacado::mpl::apply<FadType, ScalarT>::type DFadType;
  typedef typename Sacado::mpl::apply<FadType, DFadType>::type D2FadType;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;
  using ConstitutiveModel<EvalT, Traits>::compute_tangent_;

  ///
  /// Constructor
  ///
  CapImplicitFD(Teuchos::ParameterList* p, const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual ~CapImplicitFD() {};

  ///
  /// Implementation of physics
  ///
  virtual void
  computeState(typename Traits::EvalData workset, DepFieldMap dep_fields, FieldMap eval_fields);

  virtual void
  computeStateParallel(typename Traits::EvalData workset, DepFieldMap dep_fields, FieldMap eval_fields)
  {
    ALBANY_ABORT("Not implemented.");
  }

  ///
  /// No copy constructor
  ///
  CapImplicitFD(const CapImplicitFD&) = delete;

  ///
  /// No copy assignment
  ///
  CapImplicitFD&
  operator=(const CapImplicitFD&) = delete;

 private:
  ///
  /// functions for integrating cap model stress
  ///
  template <typename T>
  T
  compute_f(minitensor::Tensor<T>& sigma, minitensor::Tensor<T>& alpha, T& kappa);

  template <typename T>
  T
  compute_g(minitensor::Tensor<T>& sigma, minitensor::Tensor<T>& alpha, T& kappa);

  minitensor::Tensor<ScalarT>
  compute_dfdsigma(std::vector<ScalarT> const& XX);

  ScalarT
  compute_dfdkappa(std::vector<ScalarT> const& XX);

  minitensor::Tensor<ScalarT>
  compute_dgdsigma(std::vector<ScalarT> const& XX);

  minitensor::Tensor<DFadType>
  compute_dgdsigma(std::vector<DFadType> const& XX);

  template <typename T>
  T
  compute_Galpha(T J2_alpha);

  template <typename T>
  minitensor::Tensor<T>
  compute_halpha(minitensor::Tensor<T> const& dgdsigma, T const J2_alpha);

  template <typename T>
  T
  compute_dedkappa(T const kappa);

  template <typename T>
  T
  compute_hkappa(T const I1_dgdsigma, T const dedkappa);

  std::vector<ScalarT>
  initialize(minitensor::Tensor<ScalarT>& sigmaVal, minitensor::Tensor<ScalarT>& alphaVal, ScalarT& kappaVal, ScalarT& dgammaVal);

  void
  compute_ResidJacobian(
      std::vector<ScalarT> const&         XXVal,
      std::vector<ScalarT>&               R,
      std::vector<ScalarT>&               dRdX,
      const minitensor::Tensor<ScalarT>&  sigmaVal,
      const minitensor::Tensor<ScalarT>&  alphaVal,
      ScalarT const&                      kappaVal,
      minitensor::Tensor4<ScalarT> const& Celastic,
      bool                                kappa_flag);

  minitensor::Tensor4<ScalarT>
  compute_Cep(
      minitensor::Tensor4<ScalarT>& Celastic,
      minitensor::Tensor<ScalarT>&  sigma,
      minitensor::Tensor<ScalarT>&  alpha,
      ScalarT&                      kappa,
      ScalarT&                      dgamma);

  ///
  /// constant material parameters in Cap plasticity model
  ///
  RealType A;
  RealType B;
  RealType C;
  RealType theta;
  RealType R;
  RealType kappa0;
  RealType W;
  RealType D1;
  RealType D2;
  RealType calpha;
  RealType psi;
  RealType N;
  RealType L;
  RealType phi;
  RealType Q;
};
}  // namespace LCM

#endif
