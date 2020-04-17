// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_GENERIC_STK_MESH_STRUCT_HPP
#define ALBANY_GENERIC_STK_MESH_STRUCT_HPP

#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Teuchos_ParameterList.hpp"

// Refinement
#if defined(ALBANY_STK_PERCEPT)
#include <stk_adapt/UniformRefinerPattern.hpp>
#include <stk_percept/PerceptMesh.hpp>
#endif

namespace Albany {

// Forward declaration(s)
class CombineAndScatterManager;

class GenericSTKMeshStruct : public AbstractSTKMeshStruct
{
 public:
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>&
  getMeshSpecs();
  const Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>&
  getMeshSpecs() const;

#if defined(ALBANY_STK_PERCEPT)
  Teuchos::RCP<stk::percept::PerceptMesh>
  getPerceptMesh()
  {
    return eMesh;
  }
  Teuchos::RCP<stk::adapt::UniformRefinerPatternBase>
  getRefinerPattern()
  {
    return refinerPattern;
  }
#endif

  //! Re-load balance adapted mesh
  void
  rebalanceAdaptedMeshT(
      const Teuchos::RCP<Teuchos::ParameterList>&   params,
      Teuchos::RCP<Teuchos::Comm<int> const> const& comm);

  bool
  useCompositeTet()
  {
    return compositeTet;
  }

  // This routine builds two maps: side3D_id->cell2D_id, and
  // side3D_node_lid->cell2D_node_lid. These maps are used because the side id
  // may differ from the cell id and the nodes order in a 2D cell may not be the
  // same as in the corresponding 3D side. The second map works as follows:
  // map[3DsideGID][3Dside_local_node] = 2Dcell_local_node
  void
  buildCellSideNodeNumerationMap(
      std::string const&              sideSetName,
      std::map<GO, GO>&               sideMap,
      std::map<GO, std::vector<int>>& sideNodeMap);

 protected:
  GenericSTKMeshStruct(
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<Teuchos::ParameterList>& adaptParams,
      int const                                   numDim = -1);

  virtual ~GenericSTKMeshStruct() = default;

  void
  SetupFieldData(
      const Teuchos::RCP<Teuchos_Comm const>&                   commT,
      int const                                                 neq_,
      const AbstractFieldContainer::FieldContainerRequirements& req,
      const Teuchos::RCP<Albany::StateInfoStruct>&              sis,
      int const                                                 worksetSize_);

  bool
  buildUniformRefiner();

  bool
  buildLocalRefiner();

  void
  printParts(stk::mesh::MetaData* metaData);

  void
  cullSubsetParts(
      std::vector<std::string>&                ssNames,
      std::map<std::string, stk::mesh::Part*>& partVec);

  //! Utility function that uses some integer arithmetic to choose a good
  //! worksetSize
  int
  computeWorksetSize(int const worksetSizeMax, int const ebSizeMax) const;

  //! Re-load balance mesh
  void
  rebalanceInitialMeshT(Teuchos::RCP<Teuchos::Comm<int> const> const& comm);

  //! Sets all mesh parts as IO parts (will be written to file)
  void
  setAllPartsIO();

  //! Determine if a percept mesh object is needed
  bool buildEMesh;
  bool
  buildPerceptEMesh();

  //! Perform initial uniform refinement of the mesh
  void
  uniformRefineMesh(const Teuchos::RCP<Teuchos_Comm const>& commT);

  //! Creates a node set from a side set
  void
  addNodeSetsFromSideSets();

  //! Checks the integrity of the nodesets created from sidesets
  void
  checkNodeSetsFromSideSetsIntegrity();

  //! Creates empty mesh structs if required (and not already present)
  void
  initializeSideSetMeshSpecs(const Teuchos::RCP<Teuchos_Comm const>& commT);

  //! Creates empty mesh structs if required (and not already present)
  void
  initializeSideSetMeshStructs(const Teuchos::RCP<Teuchos_Comm const>& commT);

  //! Completes the creation of the side set mesh structs (if of type
  //! SideSetSTKMeshStruct)
  void
  finalizeSideSetMeshStructs(
      const Teuchos::RCP<Teuchos_Comm const>& commT,
      std::map<
          std::string,
          AbstractFieldContainer::FieldContainerRequirements> const&
          side_set_req,
      std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> const&
          side_set_sis,
      int worksetSize);

  //! Loads from file input required fields not found in the mesh
  void
  loadRequiredInputFields(
      const AbstractFieldContainer::FieldContainerRequirements& req,
      const Teuchos::RCP<Teuchos_Comm const>&                   commT);

  // Routines to load, fill, or compute a field
  void
  loadField(
      std::string const&                        field_name,
      Teuchos::ParameterList const&             params,
      Teuchos::RCP<Thyra_MultiVector>&          field_mv,
      const CombineAndScatterManager&           cas_manager,
      const Teuchos::RCP<Teuchos_Comm const>&   commT,
      bool                                      node,
      bool                                      scalar,
      bool                                      layered,
      const Teuchos::RCP<Teuchos::FancyOStream> out);
  void
  fillField(
      std::string const&                           field_name,
      Teuchos::ParameterList const&                params,
      Teuchos::RCP<Thyra_MultiVector>&             field_mv,
      Teuchos::RCP<Thyra_VectorSpace const> const& entities_vs,
      bool                                         node,
      bool                                         scalar,
      bool                                         layered,
      const Teuchos::RCP<Teuchos::FancyOStream>    out);
  void
  computeField(
      std::string const&                           field_name,
      Teuchos::ParameterList const&                params,
      Teuchos::RCP<Thyra_MultiVector>&             field_mv,
      Teuchos::RCP<Thyra_VectorSpace const> const& entities_vs,
      std::vector<stk::mesh::Entity> const&        entities,
      bool                                         node,
      bool                                         scalar,
      bool                                         layered,
      const Teuchos::RCP<Teuchos::FancyOStream>    out);

  // Routines to read a field from file
  void
  readScalarFileSerial(
      std::string const&                           fname,
      Teuchos::RCP<Thyra_MultiVector>&             contentVec,
      Teuchos::RCP<Thyra_VectorSpace const> const& vs,
      const Teuchos::RCP<Teuchos_Comm const>&      comm) const;

  void
  readVectorFileSerial(
      std::string const&                           fname,
      Teuchos::RCP<Thyra_MultiVector>&             contentVec,
      Teuchos::RCP<Thyra_VectorSpace const> const& vs,
      const Teuchos::RCP<Teuchos_Comm const>&      comm) const;

  void
  readLayeredScalarFileSerial(
      std::string const&                           fname,
      Teuchos::RCP<Thyra_MultiVector>&             contentVec,
      Teuchos::RCP<Thyra_VectorSpace const> const& vs,
      std::vector<double>&                         normalizedLayersCoords,
      const Teuchos::RCP<Teuchos_Comm const>&      comm) const;

  void
  readLayeredVectorFileSerial(
      std::string const&                           fname,
      Teuchos::RCP<Thyra_MultiVector>&             contentVec,
      Teuchos::RCP<Thyra_VectorSpace const> const& vs,
      std::vector<double>&                         normalizedLayersCoords,
      const Teuchos::RCP<Teuchos_Comm const>&      comm) const;

  void
  checkFieldIsInMesh(std::string const& fname, std::string const& ftype) const;

  //! Perform initial adaptation input checking
  void
  checkInput(std::string option, std::string value, std::string allowed_values);

  //! Rebuild the mesh with elem->face->segment->node connectivity for
  //! adaptation
  void
  computeAddlConnectivity();

  void
  setDefaultCoordinates3d();

  Teuchos::RCP<Teuchos::ParameterList>
  getValidGenericSTKParameters(
      std::string listname = "Discretization Param Names") const;

  Teuchos::RCP<Teuchos::ParameterList> params;

  //! The adaptation parameter list (null if the problem isn't adaptive)
  Teuchos::RCP<Teuchos::ParameterList> adaptParams;

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> meshSpecs;

#if defined(ALBANY_STK_PERCEPT)
  Teuchos::RCP<stk::percept::PerceptMesh>             eMesh;
  Teuchos::RCP<stk::adapt::UniformRefinerPatternBase> refinerPattern;
#endif

  bool uniformRefinementInitialized;

  bool requiresAutomaticAura;

  bool compositeTet;

  std::vector<std::string> m_nodesets_from_sidesets;
};

}  // namespace Albany

#endif  // ALBANY_GENERIC_STK_MESH_STRUCT_HPP
