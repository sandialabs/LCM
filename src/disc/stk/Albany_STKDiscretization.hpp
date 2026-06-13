// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_STK_DISCRETIZATION_HPP
#define ALBANY_STK_DISCRETIZATION_HPP

#include <utility>
#include <vector>

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_NullSpaceUtils.hpp"
#include "utility/Albany_ThyraCrsMatrixFactory.hpp"
#include "utility/Albany_ThyraUtils.hpp"

// Start of STK stuff
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldTraits.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/parallel/Parallel.hpp>

namespace Albany {

typedef shards::Array<GO, shards::NaturalOrder> GIDArray;

class GlobalLocalIndexer;

struct DOFsStruct
{
  Teuchos::RCP<Thyra_VectorSpace const> node_vs;
  Teuchos::RCP<Thyra_VectorSpace const> overlap_node_vs;
  Teuchos::RCP<Thyra_VectorSpace const> vs;
  Teuchos::RCP<Thyra_VectorSpace const> overlap_vs;
  NodalDOFManager                       dofManager;
  NodalDOFManager                       overlap_dofManager;
  std::vector<std::vector<LO>>          wsElNodeEqID_rawVec;
  std::vector<IDArray>                  wsElNodeEqID;
  std::vector<std::vector<GO>>          wsElNodeID_rawVec;
  std::vector<GIDArray>                 wsElNodeID;

  Teuchos::RCP<const GlobalLocalIndexer> node_vs_indexer;
  Teuchos::RCP<const GlobalLocalIndexer> overlap_node_vs_indexer;
  Teuchos::RCP<const GlobalLocalIndexer> vs_indexer;
  Teuchos::RCP<const GlobalLocalIndexer> overlap_vs_indexer;
};

struct NodalDOFsStructContainer
{
  typedef std::map<std::pair<std::string, int>, DOFsStruct> MapOfDOFsStructs;

  MapOfDOFsStructs                                        mapOfDOFsStructs;
  std::map<std::string, MapOfDOFsStructs::const_iterator> fieldToMap;

  const DOFsStruct&
  getDOFsStruct(std::string const& field_name) const
  {
    return fieldToMap.find(field_name)->second->second;
  };  // TODO handole errors

  // IKT: added the following function, which may be useful for debugging.
  void
  printFieldToMap() const
  {
    typedef std::map<std::string, MapOfDOFsStructs::const_iterator>::const_iterator MapIterator;
    Teuchos::RCP<Teuchos::FancyOStream>                                             out = Teuchos::VerboseObjectBase::getDefaultOStream();
    for (MapIterator iter = fieldToMap.begin(); iter != fieldToMap.end(); iter++) {
      std::string key = iter->first;
      *out << "IKT Key: " << key << "\n";
      auto vs = getDOFsStruct(key).vs;
      *out << "IKT Vector Space \n: ";
      describe(vs, *out, Teuchos::VERB_EXTREME);
    }
  }

  void
  addEmptyDOFsStruct(std::string const& field_name, std::string const& meshPart, int numComps)
  {
    if (numComps != 1) mapOfDOFsStructs.insert(make_pair(make_pair(meshPart, 1), DOFsStruct()));

    fieldToMap[field_name] = mapOfDOFsStructs.insert(make_pair(make_pair(meshPart, numComps), DOFsStruct())).first;
  }
};

class STKDiscretization : public AbstractDiscretization
{
 public:
  //! Constructor
  STKDiscretization(
      const Teuchos::RCP<Teuchos::ParameterList>&    discParams,
      Teuchos::RCP<AbstractSTKMeshStruct>&           stkMeshStruct,
      const Teuchos::RCP<Teuchos_Comm const>&        comm,
      const Teuchos::RCP<RigidBodyModes>&            rigidBodyModes   = Teuchos::null,
      std::map<int, std::vector<std::string>> const& sideSetEquations = std::map<int, std::vector<std::string>>());

  //! Destructor
  virtual ~STKDiscretization();

  void
  printConnectivity() const;

  //! Get node vector space (owned and overlapped)
  Teuchos::RCP<Thyra_VectorSpace const>
  getNodeVectorSpace() const
  {
    return m_node_vs;
  }
  Teuchos::RCP<Thyra_VectorSpace const>
  getOverlapNodeVectorSpace() const
  {
    return m_overlap_node_vs;
  }

  //! Get solution DOF vector space (owned and overlapped).
  Teuchos::RCP<Thyra_VectorSpace const>
  getVectorSpace() const
  {
    return m_vs;
  }
  Teuchos::RCP<Thyra_VectorSpace const>
  getOverlapVectorSpace() const
  {
    return m_overlap_vs;
  }

  //! Get Field node vector space (owned and overlapped)
  Teuchos::RCP<Thyra_VectorSpace const>
  getNodeVectorSpace(std::string const& field_name) const;
  Teuchos::RCP<Thyra_VectorSpace const>
  getOverlapNodeVectorSpace(std::string const& field_name) const;

  //! Get Field vector space (owned and overlapped)
  Teuchos::RCP<Thyra_VectorSpace const>
  getVectorSpace(std::string const& field_name) const;
  Teuchos::RCP<Thyra_VectorSpace const>
  getOverlapVectorSpace(std::string const& field_name) const;

  //! Create a Jacobian operator (owned and overlapped)
  Teuchos::RCP<Thyra_LinearOp>
  createJacobianOp() const
  {
    return m_jac_factory->createOp();
  }
  Teuchos::RCP<Thyra_LinearOp>
  createOverlapJacobianOp() const
  {
    return m_overlap_jac_factory->createOp();
  }

  void
  setConstrainedDOFs(std::set<GO> const& constrained_dof_gids, std::map<GO, double> const& constrained_dof_values = {}) override;

  void
  setConstrainedDOFValues(
      std::map<GO, double> const& u_values,
      std::map<GO, double> const& v_values = {},
      std::map<GO, double> const& a_values = {}) override;

  //! Get Node set lists (typedef in Albany_AbstractDiscretization.hpp)
  NodeSetList const&
  getNodeSets() const
  {
    return nodeSets;
  }
  NodeSetGIDsList const&
  getNodeSetGIDs() const
  {
    return nodeSetGIDs;
  }
  NodeSetCoordList const&
  getNodeSetCoords() const
  {
    return nodeSetCoords;
  }
  NodeSetGIDsList const&
  getNodeSetOverlapGIDs() const
  {
    return nodeSetOverlapGIDs;
  }
  NodeSetCoordList const&
  getNodeSetOverlapCoords() const
  {
    return nodeSetOverlapCoords;
  }
  NodeGID2LIDMap const&
  getNodeGID2LIDMap() const
  {
    return node_GID_2_LID_map;
  }

  NodeSetList&
  getNodeSets()
  {
    return nodeSets;
  }
  NodeSetGIDsList&
  getNodeSetGIDs()
  {
    return nodeSetGIDs;
  }
  NodeSetCoordList&
  getNodeSetCoords()
  {
    return nodeSetCoords;
  }
  NodeGID2LIDMap&
  getNodeGID2LIDMap()
  {
    return node_GID_2_LID_map;
  }

  //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
  SideSetList const&
  getSideSets(int const workset) const
  {
    return sideSets[workset];
  }

  SideSetList&
  getSideSets(int const workset)
  {
    return sideSets[workset];
  }

  //! Get connectivity map from elementGID to workset
  WsLIDList&
  getElemGIDws()
  {
    return elemGIDws;
  }
  WsLIDList const&
  getElemGIDws() const
  {
    return elemGIDws;
  }

  //! Get map from ws, elem, node [, eq] -> [Node|DOF] GID
  Conn const&
  getWsElNodeEqID() const
  {
    return wsElNodeEqID;
  }

  WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>>::type const&
  getWsElNodeID() const
  {
    return wsElNodeID;
  }

  //! Get IDArray for (Ws, Local Node, nComps) -> (local) NodeLID, works for
  //! both scalar and vector fields
  std::vector<IDArray> const&
  getElNodeEqID(std::string const& field_name) const
  {
    return nodalDOFsStructContainer.getDOFsStruct(field_name).wsElNodeEqID;
  }

  const NodalDOFManager&
  getDOFManager(std::string const& field_name) const
  {
    return nodalDOFsStructContainer.getDOFsStruct(field_name).dofManager;
  }

  const NodalDOFManager&
  getOverlapDOFManager(std::string const& field_name) const
  {
    return nodalDOFsStructContainer.getDOFsStruct(field_name).overlap_dofManager;
  }

  //! Retrieve coodinate vector (num_used_nodes * 3)
  const Teuchos::ArrayRCP<double>&
  getCoordinates() const;
  void
  setCoordinates(const Teuchos::ArrayRCP<double const>& c);
  void
  setReferenceConfigurationManager(const Teuchos::RCP<AAdapt::rc::Manager>& rcm);

#if defined(ALBANY_CONTACT)
  //! Get the contact manager
  Teuchos::RCP<const ContactManager>
  getContactManager() const
  {
    return contactManager;
  }
#endif

  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*>>>::type&
  getCoords() const
  {
    return coords;
  }
  const WorksetArray<Teuchos::ArrayRCP<double>>::type&
  getSphereVolume() const
  {
    return sphereVolume;
  }
  const WorksetArray<Teuchos::ArrayRCP<double*>>::type&
  getLatticeOrientation() const
  {
    return latticeOrientation;
  }

  const WorksetArray<Teuchos::ArrayRCP<std::uint8_t>>::type&
  getCellIsErodible() const
  {
    return cell_is_erodible;
  }

  void
  printElemGIDws() const;

  std::map<std::pair<int, int>, GO>
  getElemWsLIDGIDMap() const;

  void
  printWsElNodeID() const;

  //! Print the coordinates for debugging
  void
  printCoords() const;

  //! Set stateArrays
  void
  setStateArrays(StateArrays& sa)
  {
    stateArrays = sa;
  }

  //! Get stateArrays
  StateArrays&
  getStateArrays()
  {
    return stateArrays;
  }

  //! Get nodal parameters state info struct
  const StateInfoStruct&
  getNodalParameterSIS() const
  {
    return stkMeshStruct->getFieldContainer()->getNodalParameterSIS();
  }

  //! Retrieve Vector (length num worksets) of element block names
  const WorksetArray<std::string>::type&
  getWsEBNames() const
  {
    return wsEBNames;
  }
  //! Retrieve Vector (length num worksets) of physics set index
  const WorksetArray<int>::type&
  getWsPhysIndex() const
  {
    return wsPhysIndex;
  }

  // Retrieve mesh struct
  Teuchos::RCP<AbstractSTKMeshStruct>
  getSTKMeshStruct() const
  {
    return stkMeshStruct;
  }
  Teuchos::RCP<AbstractMeshStruct>
  getMeshStruct() const
  {
    return stkMeshStruct;
  }

  const SideSetDiscretizationsType&
  getSideSetDiscretizations() const
  {
    return sideSetDiscretizations;
  }

  std::map<std::string, std::map<GO, GO>> const&
  getSideToSideSetCellMap() const
  {
    return sideToSideSetCellMap;
  }

  std::map<std::string, std::map<GO, std::vector<int>>> const&
  getSideNodeNumerationMap() const
  {
    return sideNodeNumerationMap;
  }

  //! Flag if solution has a restart values -- used in Init Cond
  bool
  hasRestartSolution() const
  {
    return stkMeshStruct->hasRestartSolution();
  }

  //! If restarting, convenience function to return restart data time
  double
  restartDataTime() const
  {
    return stkMeshStruct->restartDataTime();
  }

  //! After mesh modification, need to update the element connectivity and nodal
  //! coordinates
  void
  updateMesh();

  //! Phase 0: workset-only subset of updateMesh(), for testing the
  //! mid-run rebuild path before any element-death semantics are layered on.
  void
  rebuildWorksets() override;

  //! Element death: find live owned cells that have calved off the
  //! structure -- cells no longer connected, through live-cell node
  //! adjacency, to any of the named anchor node sets (the kinematic
  //! ground, e.g. the bluff's back wall and base). A detached block free
  //! falls under gravity and, at a coarse coupling step, teleports an
  //! absurd distance in one step before the kinematic criteria fire; this
  //! removes it the moment it disconnects instead. The flood-fill is pure
  //! node-sharing connectivity, so it is dimension- and topology-agnostic
  //! (2D/3D, hex/tet). cell_death is set to 1 on each detached cell here
  //! and the cells are returned so the caller adds them to the kill set.
  //! pending_dead are this step's criterion-killed cells, treated as dead
  //! for the connectivity (they are about to be removed). Parallel: under
  //! NO_AUTO_AURA cross-rank connectivity is only through shared nodes, so
  //! reachability is exchanged via the shared-node GID union each round,
  //! iterating to global convergence.
  stk::mesh::EntityVector
  findDetachedCells(
      std::vector<std::string> const& anchor_node_sets,
      stk::mesh::EntityVector const&  pending_dead);

  //! Function that transforms an STK mesh of a unit cube (for LandIce problems)
  void
  transformMesh();

  //! Close current exodus file in stk_io and create a new one for an adapted
  //! mesh and new results
  void
  reNameExodusOutput(std::string& filename);

  //! Get number of spatial dimensions
  int
  getNumDim() const
  {
    return stkMeshStruct->numDim;
  }

  //! Get number of total DOFs per node
  int
  getNumEq() const
  {
    return neq;
  }

  //! Locate nodal dofs in non-overlapping vectors using local indexing
  int
  getOwnedDOF(int const inode, int const eq) const;

  //! Locate nodal dofs in overlapping vectors using local indexing
  int
  getOverlapDOF(int const inode, int const eq) const;

  //! Get global id of the stk entity
  GO
  gid(const stk::mesh::Entity entity) const;

  //! Locate nodal dofs using global indexing
  GO
  getGlobalDOF(const GO inode, int const eq) const;

  Teuchos::RCP<LayeredMeshNumbering<LO>>
  getLayeredMeshNumbering() const
  {
    return stkMeshStruct->layered_mesh_numbering;
  }

  const stk::mesh::MetaData&
  getSTKMetaData() const
  {
    return metaData;
  }
  const stk::mesh::BulkData&
  getSTKBulkData() const
  {
    return bulkData;
  }

  // --- Get/set solution/residual/field vectors to/from mesh --- //

  Teuchos::RCP<Thyra_Vector>
  getSolutionField(bool const overlapped = false) const;
  Teuchos::RCP<Thyra_MultiVector>
  getSolutionMV(bool const overlapped = false) const;

  void
  setResidualField(Thyra_Vector const& residual);

  void
  getField(Thyra_Vector& field_vector, std::string const& field_name) const;
  void
  setField(Thyra_Vector const& field_vector, std::string const& field_name, bool const overlapped = false);

  Teuchos::RCP<Thyra_MultiVector>
  getCoordMV()
  {
    return coordMV;
  }

  // --- Methods to write solution in the output file --- //

  void
  writeSolution(Thyra_Vector const& solution, double const time, bool const overlapped = false);
  void
  writeSolution(Thyra_Vector const& solution, Thyra_Vector const& solution_dot, double const time, bool const overlapped = false);
  void
  writeSolution(
      Thyra_Vector const& solution,
      Thyra_Vector const& solution_dot,
      Thyra_Vector const& solution_dotdot,
      double const        time,
      bool const          overlapped = false);
  void
  writeSolutionMV(const Thyra_MultiVector& solution, double const time, bool const overlapped = false);

  //! Write the solution to the mesh database.
  void
  writeSolutionToMeshDatabase(Thyra_Vector const& solution, double const /* time */, bool const overlapped = false);
  void
  writeSolutionToMeshDatabase(Thyra_Vector const& solution, Thyra_Vector const& solution_dot, double const /* time */, bool const overlapped = false);
  void
  writeSolutionToMeshDatabase(
      Thyra_Vector const& solution,
      Thyra_Vector const& solution_dot,
      Thyra_Vector const& solution_dotdot,
      double const /* time */,
      bool const overlapped = false);
  void
  writeSolutionMVToMeshDatabase(const Thyra_MultiVector& solution, double const /* time */, bool const overlapped = false);

  //! Write the solution to file. Must call writeSolution first.
  void
  writeSolutionToFile(Thyra_Vector const& solution, double const time, bool const overlapped = false);
  void
  writeSolutionMVToFile(const Thyra_MultiVector& solution, double const time, bool const overlapped = false);

  void
  outputExodusSolutionInitialTime(const bool output_initial_soln_to_exo_file_)
  {
    output_initial_soln_to_exo_file = output_initial_soln_to_exo_file_;
  };

  void
  setOutputInterval(int oi)
  {
    outputInterval = oi;
  }

  int
  getOutputInterval()
  {
    return outputInterval;
  }

  //! used when NetCDF output on a latitude-longitude grid is requested.
  // Each struct contains a latitude/longitude index and it's parametric
  // coordinates in an element.
  struct interp
  {
    std::pair<double, double>     parametric_coords;
    std::pair<unsigned, unsigned> latitude_longitude;
  };

  Teuchos::RCP<Teuchos_Comm const>
  getComm() const
  {
    return comm;
  }

  //! Find the local side id number within parent element
  unsigned
  determine_local_side_id(const stk::mesh::Entity elem, stk::mesh::Entity side);

 protected:
  void
  getSolutionField(Thyra_Vector& result, bool overlapped) const;
  void
  getSolutionMV(Thyra_MultiVector& result, bool overlapped) const;

  void
  setSolutionField(Thyra_Vector const& soln, bool const overlapped);
  void
  setSolutionField(Thyra_Vector const& soln, Thyra_Vector const& soln_dot, bool const overlapped);
  void
  setSolutionField(Thyra_Vector const& soln, Thyra_Vector const& soln_dot, Thyra_Vector const& soln_dotdot, bool const overlapped);
  void
  setSolutionFieldMV(const Thyra_MultiVector& solnT, bool const overlapped);

  double
  monotonicTimeLabel(double const time);

  void
  computeNodalVectorSpaces(bool overlapped);

  //! Process STK mesh for CRS Graphs
  virtual void
  computeGraphs();
  //! Process STK mesh for Owned nodal quantitites
  void
  computeOwnedNodesAndUnknowns();
  //! Process coords for ML
  void
  setupMLCoords();
  //! Process STK mesh for Overlap nodal quantitites
  void
  computeOverlapNodesAndUnknowns();
  //! Process STK mesh for Workset/Bucket Info
  void
  computeWorksetInfo();
  //! Process STK mesh for NodeSets
  void
  computeNodeSets();
  //! Process STK mesh for SideSets
  void
  computeSideSets();
  //! Call stk_io for creating exodus output file
  void
  setupExodusOutput();

  //! Convert the stk mesh on this processor to a nodal graph using SEACAS
  void
  meshToGraph();

  void
  writeCoordsToMatrixMarket() const;

  void
  buildSideSetProjectors();

  double previous_time_label;

  int
  nonzeroesPerRow(int const neq) const;

  // ==================== Members =================== //

  Teuchos::RCP<Teuchos::FancyOStream> out;

  //! Stk Mesh Objects
  stk::mesh::MetaData& metaData;
  stk::mesh::BulkData& bulkData;

  //! Teuchos communicator
  Teuchos::RCP<Teuchos_Comm const> comm;

  //! DOF GIDs eliminated from the owned space via Dirichlet BC elimination,
  //! and their prescribed u / u̇ / ü values (for solution output reconstruction
  //! via the expand-and-inject path in setSolutionField). The u map is seeded
  //! at setConstrainedDOFs time and refreshed each call by Application::
  //! injectConstrainedDOFValues; v and a are populated only when refreshed.
  std::set<GO>         constrained_dof_gids_;
  std::map<GO, double> constrained_dof_values_;
  std::map<GO, double> constrained_dof_dot_values_;
  std::map<GO, double> constrained_dof_dotdot_values_;

  //! Unknown map and node map
  Teuchos::RCP<Thyra_VectorSpace const> m_vs;
  Teuchos::RCP<Thyra_VectorSpace const> m_node_vs;

  //! Overlapped unknown map and node map
  Teuchos::RCP<Thyra_VectorSpace const> m_overlap_vs;
  Teuchos::RCP<Thyra_VectorSpace const> m_overlap_node_vs;

  //! Jacobian matrix graph proxy (owned and overlap)
  Teuchos::RCP<ThyraCrsMatrixFactory> m_jac_factory;
  Teuchos::RCP<ThyraCrsMatrixFactory> m_overlap_jac_factory;

  NodalDOFsStructContainer nodalDOFsStructContainer;

  //! Processor ID
  unsigned int myPID;

  //! Number of equations (and unknowns) per node
  const unsigned int neq;

  //! Equations that are defined only on some side sets of the mesh
  std::map<int, std::vector<std::string>> sideSetEquations;

  //! Number of elements on this processor
  unsigned int numMyElements;

  //! node sets stored as std::map(string ID, int vector of GIDs)
  NodeSetList      nodeSets;
  NodeSetGIDsList  nodeSetGIDs;
  NodeSetCoordList nodeSetCoords;
  //! Overlap-scope nodeset GIDs and coords (includes ghosted nodes). Used by
  //! DBC DOF elimination so every rank knows which of its overlap DOFs are
  //! constrained, even if the constraint's owner is a different rank.
  NodeSetGIDsList  nodeSetOverlapGIDs;
  NodeSetCoordList nodeSetOverlapCoords;
  NodeGID2LIDMap   node_GID_2_LID_map;

  //! side sets stored as std::map(string ID, SideArray classes) per workset
  //! (std::vector across worksets)
  std::vector<SideSetList> sideSets;

  //! Connectivity array [workset, element, local-node, Eq] => LID
  Conn wsElNodeEqID;

  //! Connectivity array [workset, element, local-node] => GID
  WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>>::type wsElNodeID;

  mutable Teuchos::ArrayRCP<double>                                 coordinates;
  Teuchos::RCP<Thyra_MultiVector>                                   coordMV;
  WorksetArray<std::string>::type                                   wsEBNames;
  WorksetArray<int>::type                                           wsPhysIndex;
  WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*>>>::type coords;
  WorksetArray<Teuchos::ArrayRCP<double>>::type                     sphereVolume;
  WorksetArray<Teuchos::ArrayRCP<double*>>::type                    latticeOrientation;
  WorksetArray<Teuchos::ArrayRCP<std::uint8_t>>::type               cell_is_erodible;

#if defined(ALBANY_CONTACT)
  Teuchos::RCP<ContactManager> contactManager;
#endif

  //! Connectivity map from elementGID to workset and LID in workset
  WsLIDList elemGIDws;

  // States: vector of length worksets of a map from field name to shards array
  StateArrays                                   stateArrays;
  std::vector<std::vector<std::vector<double>>> nodesOnElemStateVec;

  //! list of all owned nodes, saved for setting solution
  std::vector<stk::mesh::Entity> ownednodes;
  std::vector<stk::mesh::Entity> cells;

  //! list of all overlap nodes, saved for getting coordinates for mesh motion
  std::vector<stk::mesh::Entity> overlapnodes;

  //! Number of elements on this processor
  int numOwnedNodes;
  int numOverlapNodes;
  GO  maxGlobalNodeGID;

  // Needed to pass coordinates to ML.
  Teuchos::RCP<RigidBodyModes> rigidBodyModes;

  int              netCDFp;
  size_t           netCDFOutputRequest;
  std::vector<int> varSolns;

  WorksetArray<Teuchos::ArrayRCP<std::vector<interp>>>::type interpolateData;

  // Storage used in periodic BCs to un-roll coordinates. Pointers saved for
  // destructor.
  std::vector<double*> toDelete;

  Teuchos::RCP<AbstractSTKMeshStruct> stkMeshStruct;

  Teuchos::RCP<Teuchos::ParameterList> discParams;

  // Sideset discretizations
  std::map<std::string, Teuchos::RCP<AbstractDiscretization>> sideSetDiscretizations;
  std::map<std::string, Teuchos::RCP<STKDiscretization>>      sideSetDiscretizationsSTK;
  std::map<std::string, std::map<GO, GO>>                     sideToSideSetCellMap;
  std::map<std::string, std::map<GO, std::vector<int>>>       sideNodeNumerationMap;
  std::map<std::string, Teuchos::RCP<Thyra_LinearOp>>         projectors;
  std::map<std::string, Teuchos::RCP<Thyra_LinearOp>>         ov_projectors;

  // Used in Exodus writing capability
  Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data;

  int outputInterval;

  size_t outputFileIdx;
  bool   interleavedOrdering;

  // Boolean for disabling output of initial solution to Exodus file
  bool output_initial_soln_to_exo_file{true};

 private:
  Teuchos::RCP<ThyraCrsMatrixFactory> nodalMatrixFactory;

  template <typename T, typename ContainerType>
  bool
  in_list(const T& value, const ContainerType& list)
  {
    for (const T& item : list) {
      if (item == value) {
        return true;
      }
    }
    return false;
  }

  void
  printVertexConnectivity();

  void
  computeGraphsUpToFillComplete();

  void
  fillCompleteGraphs();

  //! Flag every owned cell that touches a "*-erodible" side-set.
  //! Rebuilt on every computeWorksetInfo / rebuildWorksets so the
  //! predicate tracks the receding bluff after element death.
  void
  computeWorksetInfoErodibleCells();
};

}  // namespace Albany

#endif  // ALBANY_STK_DISCRETIZATION_HPP
