// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_DISCRETIZATIONFACTORY_HPP
#define ALBANY_DISCRETIZATIONFACTORY_HPP

#include <vector>

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractFieldContainer.hpp"
#include "Albany_AbstractMeshStruct.hpp"
#include "Albany_NullSpaceUtils.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {

/*!
 * \brief A factory class to instantiate AbstractDiscretization objects
 */
class DiscretizationFactory
{
 public:
  //! Default constructor
  DiscretizationFactory(
      const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
      const Teuchos::RCP<Teuchos_Comm const>&     commT);

  //! Destructor
  ~DiscretizationFactory() {}

  static Teuchos::RCP<Albany::AbstractMeshStruct>
  createMeshStruct(
      Teuchos::RCP<Teuchos::ParameterList> disc_params,
      Teuchos::RCP<Teuchos::ParameterList> adapt_params,
      Teuchos::RCP<Teuchos_Comm const>     comm);

  Teuchos::RCP<Albany::AbstractMeshStruct>
  getMeshStruct()
  {
    return meshStruct;
  }

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>
  createMeshSpecs();

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>
  createMeshSpecs(Teuchos::RCP<Albany::AbstractMeshStruct> mesh);

  Teuchos::RCP<Albany::AbstractDiscretization>
  createDiscretization(
      unsigned int                                              num_equations,
      const Teuchos::RCP<Albany::StateInfoStruct>&              sis,
      const AbstractFieldContainer::FieldContainerRequirements& req,
      const Teuchos::RCP<Albany::RigidBodyModes>&               rigidBodyModes =
          Teuchos::null);

  Teuchos::RCP<Albany::AbstractDiscretization>
  createDiscretization(
      unsigned int                                   num_equations,
      std::map<int, std::vector<std::string>> const& sideSetEquations,
      const Teuchos::RCP<Albany::StateInfoStruct>&   sis,
      std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> const&
                                                                side_set_sis,
      const AbstractFieldContainer::FieldContainerRequirements& req,
      std::map<
          std::string,
          AbstractFieldContainer::FieldContainerRequirements> const&
                                                  side_set_req,
      const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes =
          Teuchos::null);

  void
  setupInternalMeshStruct(
      unsigned int                                              neq,
      const Teuchos::RCP<Albany::StateInfoStruct>&              sis,
      const AbstractFieldContainer::FieldContainerRequirements& req);

  void
  setupInternalMeshStruct(
      unsigned int                                 neq,
      const Teuchos::RCP<Albany::StateInfoStruct>& sis,
      std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> const&
                                                                side_set_sis,
      const AbstractFieldContainer::FieldContainerRequirements& req,
      std::map<
          std::string,
          AbstractFieldContainer::FieldContainerRequirements> const&
          side_set_req);

  Teuchos::RCP<Albany::AbstractDiscretization>
  createDiscretizationFromInternalMeshStruct(
      const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes);

  Teuchos::RCP<Albany::AbstractDiscretization>
  createDiscretizationFromInternalMeshStruct(
      std::map<int, std::vector<std::string>> const& sideSetEquations,
      const Teuchos::RCP<Albany::RigidBodyModes>&    rigidBodyModes);

  /* This function overwrite previous discretization parameter list */
  void
  setDiscretizationParameters(Teuchos::RCP<Teuchos::ParameterList> disc_params);

 private:
  //! Private to prohibit copying
  DiscretizationFactory(const DiscretizationFactory&);

  //! Private to prohibit copying
  DiscretizationFactory&
  operator=(const DiscretizationFactory&);

  std::map<int, std::vector<std::string>> const empty_side_set_equations;
  std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> const
      empty_side_set_sis;
  const std::
      map<std::string, AbstractFieldContainer::FieldContainerRequirements>
          empty_side_set_req;

 protected:
  //! Parameter list specifying what element to create
  Teuchos::RCP<Teuchos::ParameterList> discParams;

  //! Parameter list specifying adaptation parameters (null if problem isn't
  //! adaptive)
  Teuchos::RCP<Teuchos::ParameterList> adaptParams;

  //! Parameter list specifying solver parameters
  Teuchos::RCP<Teuchos::ParameterList> piroParams;

  //! Parameter list specifying parameters for Catalyst
  Teuchos::RCP<Teuchos::ParameterList> catalystParams;

  Teuchos::RCP<Teuchos_Comm const> commT;

  Teuchos::RCP<Albany::AbstractMeshStruct> meshStruct;
};

}  // namespace Albany

#endif  // ALBANY_DISCRETIZATIONFACTORY_HPP
