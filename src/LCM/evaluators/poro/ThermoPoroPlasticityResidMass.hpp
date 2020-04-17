// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef THERMOPOROPLASTICITYRESIDMASS_HPP
#define THERMOPOROPLASTICITYRESIDMASS_HPP

#include "Albany_Types.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
/** \brief
 *   Balance of energy residual for large deformation
 *   thermoporomechanics problem.

*/

template <typename EvalT, typename Traits>
class ThermoPoroPlasticityResidMass
    : public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ThermoPoroPlasticityResidMass(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint>      wBF;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                porePressure;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                densityPoreFluid;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                Temp;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                RefTemp;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                stabParameter;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                ThermalCond;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                kcPermeability;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                porosity;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                biotCoefficient;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                biotModulus;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                young_modulus_;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                poissons_ratio_;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wGradBF;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>           TGrad;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>           TempGrad;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                alphaMixture;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                alphaPoreFluid;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                alphaSkeleton;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                Source;
  Teuchos::Array<double>                                      convectionVels;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                rhoCp;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                Absorption;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim, Dim>      strain;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim, Dim>      defgrad;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                J;

  // stabilization term
  PHX::MDField<const MeshScalarT, Cell, Vertex, Dim> coordVec;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>     cubature;
  Teuchos::RCP<shards::CellTopology>                 cellType;
  PHX::MDField<const MeshScalarT, Cell, QuadPoint>   weights;

  // Time
  PHX::MDField<ScalarT const, Dummy> deltaTime;

  // Data from previous time step
  std::string strainName, porePressureName, porosityName, JName, TempName;

  bool         haveSource;
  bool         haveConvection;
  bool         haveAbsorption;
  bool         enableTransient;
  bool         haverhoCp;
  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  // Temporary Views
  Kokkos::DynRankView<ScalarT, PHX::Device> flux;
  Kokkos::DynRankView<ScalarT, PHX::Device> fgravity;
  Kokkos::DynRankView<ScalarT, PHX::Device> fluxdt;
  Kokkos::DynRankView<ScalarT, PHX::Device> pterm;
  Kokkos::DynRankView<ScalarT, PHX::Device> Tterm;
  Kokkos::DynRankView<ScalarT, PHX::Device> aterm;
  Kokkos::DynRankView<ScalarT, PHX::Device> tpterm;

  // Work space FCs
  Kokkos::DynRankView<ScalarT, PHX::Device> F_inv;
  Kokkos::DynRankView<ScalarT, PHX::Device> F_invT;
  Kokkos::DynRankView<ScalarT, PHX::Device> C;
  Kokkos::DynRankView<ScalarT, PHX::Device> Cinv;
  Kokkos::DynRankView<ScalarT, PHX::Device> JF_invT;
  Kokkos::DynRankView<ScalarT, PHX::Device> KJF_invT;
  Kokkos::DynRankView<ScalarT, PHX::Device> Kref;

  ScalarT porePbar, Tempbar, vol;
  ScalarT trialPbar;
  ScalarT shearModulus, bulkModulus;
  ScalarT safeFactor;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> TResidual;
};
}  // namespace LCM

#endif
