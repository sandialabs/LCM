// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ACETHERMALPARAMETERS_HPP
#define ACETHERMALPARAMETERS_HPP

#include "Albany_MaterialDatabase.hpp"
#include "Albany_Types.hpp"
#include "Albany_config.h"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
/**
 * \brief Evaluates thermal parameters (e.g. conductivity, inertia) for ACE
 * stand-alone thermal problem.
 */

template <typename EvalT, typename Traits>
class ACEThermalParameters : public PHX::EvaluatorWithBaseImpl<Traits>, public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ACEThermalParameters(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

  void
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

  void
  createElementBlockParameterMaps();

  ScalarT
  queryElementBlockParameterMap(std::string const eb_name, const std::map<std::string, RealType> map);

  std::vector<RealType>
  queryElementBlockParameterMap(std::string const eb_name, const std::map<std::string, std::vector<RealType>> map);

 private:
  //! Validate the name strings under "ACE Thermal Parameters" section in input
  //! file
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidThermalCondParameters() const;

  unsigned int                                          num_qps_{0}, num_dims_{0}, num_nodes_{0}, workset_size_{0};
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> coord_vec_;
  PHX::MDField<ScalarT, Cell, QuadPoint>                thermal_conductivity_;
  PHX::MDField<ScalarT, Cell, QuadPoint>                thermal_inertia_;
  PHX::MDField<ScalarT, Cell, QuadPoint>                bluff_salinity_;
  PHX::MDField<ScalarT, Cell, QuadPoint>                ice_saturation_;
  PHX::MDField<ScalarT, Cell, QuadPoint>                density_;
  PHX::MDField<ScalarT, Cell, QuadPoint>                heat_capacity_;
  PHX::MDField<ScalarT, Cell, QuadPoint>                water_saturation_;
  PHX::MDField<ScalarT, Cell, QuadPoint>                porosity_;
  PHX::MDField<ScalarT, Cell, QuadPoint>                temperature_;
  PHX::MDField<ScalarT, Cell, Node, Dim>                thermal_cond_grad_at_nodes_;
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim>           thermal_cond_grad_at_qps_;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wgradbf_;
  //! Basis Functions
  PHX::MDField<const RealType, Cell, Node, QuadPoint> bf_;

  //! Material database - holds thermal conductivity and inertia, among other
  //! quantities
  Teuchos::RCP<Albany::MaterialDatabase> material_db_;

  //! Array containing the names of the element blocks present in the materials
  //! file
  Teuchos::ArrayRCP<std::string> eb_names_;

  //! Block-dependent saturation hardening constants read in from materials.yaml
  //! file
  std::map<std::string, RealType> const_thermal_conduct_map_;
  std::map<std::string, RealType> const_thermal_inertia_map_;
  std::map<std::string, RealType> ice_density_map_;
  std::map<std::string, RealType> water_density_map_;
  std::map<std::string, RealType> soil_density_map_;
  std::map<std::string, RealType> ice_thermal_cond_map_;
  std::map<std::string, RealType> water_thermal_cond_map_;
  std::map<std::string, RealType> soil_thermal_cond_map_;
  std::map<std::string, RealType> ice_heat_capacity_map_;
  std::map<std::string, RealType> water_heat_capacity_map_;
  std::map<std::string, RealType> soil_heat_capacity_map_;
  std::map<std::string, RealType> salinity_base_map_;
  std::map<std::string, RealType> salt_enhanced_D_map_;
  std::map<std::string, RealType> latent_heat_map_;
  std::map<std::string, RealType> porosity_bulk_map_;
  std::map<std::string, RealType> element_size_map_;  // IKT FIXME? this is not block dept currently unless
                                                      // have different mesh resolution in different blocks,
                                                      // so can be RealType

  //! Block-dependent params with depth read in from materials.yaml file
  std::map<std::string, std::vector<RealType>> time_map_;  // IKT FIXME?: this is not block dept so can be std::vector
  std::map<std::string, std::vector<RealType>> z_above_mean_sea_level_map_;
  std::map<std::string, std::vector<RealType>> sea_level_map_;
  std::map<std::string, std::vector<RealType>> salinity_map_;
  std::map<std::string, std::vector<RealType>> ocean_salinity_map_;  // IKT FIXME?: this is not block dept so can be a
                                                                     // std::vector
  std::map<std::string, std::vector<RealType>> porosity_from_file_map_;
  std::map<std::string, std::vector<RealType>> sand_from_file_map_;
  std::map<std::string, std::vector<RealType>> clay_from_file_map_;
  std::map<std::string, std::vector<RealType>> silt_from_file_map_;
  std::map<std::string, std::vector<RealType>> peat_from_file_map_;

  //! Variables keeping track of whether cells are on erodible boundary
  bool                       have_cell_boundary_indicator_{false};
  Teuchos::ArrayRCP<double*> cell_boundary_indicator_;
};
}  // namespace LCM

#endif
