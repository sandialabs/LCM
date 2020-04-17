// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_ThermoMechanical_Coefficients_hpp)
#define LCM_ThermoMechanical_Coefficients_hpp

#include "Albany_Layouts.hpp"
#include "Albany_Types.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
/// \brief
///
/// This evaluator computes quantities needed for heat conduction problems.
///
template <typename EvalT, typename Traits>
class ThermoMechanicalCoefficients : public PHX::EvaluatorWithBaseImpl<Traits>,
                                     public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ///
  /// Constructor
  ///
  ThermoMechanicalCoefficients(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Phalanx method to allocate space
  ///
  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  ///
  /// Implementation of physics
  ///
  void
  evaluateFields(typename Traits::EvalData d);

 private:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  ///
  /// Input: temperature
  ///
  PHX::MDField<ScalarT const, Cell, QuadPoint> temperature_;

  ///
  /// Input: thermal conductivity
  ///
  PHX::MDField<ScalarT const, Cell, QuadPoint> thermal_cond_;

  ///
  /// Input: time step
  ///
  PHX::MDField<ScalarT const, Dummy> delta_time_;

  ///
  /// Optional deformation gradient
  ///
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim, Dim> def_grad_;

  ///
  /// Output: thermal transient coefficient
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> thermal_transient_coeff_;

  ///
  /// Output: thermal Diffusivity
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> thermal_diffusivity_;

  ///
  /// Output: temperature dot
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> temperature_dot_;

  ///
  /// Number of integration points
  ///
  int num_pts_;

  ///
  /// Number of spatial dimesions
  ///
  int num_dims_;

  ///
  /// Thermal Constants
  ///
  RealType heat_capacity_, density_, transient_coeff_;

  ///
  /// Scalar name
  ///
  std::string temperature_name_;

  std::string SolutionType_;

  ///
  /// Mechanics flag
  ///
  bool have_mech_;
};
}  // namespace LCM

#endif
