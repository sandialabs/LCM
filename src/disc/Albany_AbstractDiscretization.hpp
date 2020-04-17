// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_ABSTRACT_DISCRETIZATION_HPP
#define ALBANY_ABSTRACT_DISCRETIZATION_HPP

#include "Albany_AbstractMeshStruct.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include "Albany_NodalDOFManager.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Albany_config.h"
#include "Shards_Array.hpp"
#include "Shards_CellTopologyData.h"

namespace Albany {

#if defined(ALBANY_CONTACT)
class ContactManager;
#endif

class AbstractDiscretization
{
 public:
  typedef std::map<std::string, Teuchos::RCP<AbstractDiscretization>>
      SideSetDiscretizationsType;

  //! Constructor
  AbstractDiscretization() = default;

  //! Prohibit copying
  AbstractDiscretization(const AbstractDiscretization&) = delete;

  //! Private to prohibit copying
  AbstractDiscretization&
  operator=(const AbstractDiscretization&) = default;

  //! Destructor
  virtual ~AbstractDiscretization() = default;

  //! Get node vector space (owned and overlapped)
  virtual Teuchos::RCP<Thyra_VectorSpace const>
  getNodeVectorSpace() const = 0;
  virtual Teuchos::RCP<Thyra_VectorSpace const>
  getOverlapNodeVectorSpace() const = 0;

  //! Get solution DOF vector space (owned and overlapped).
  virtual Teuchos::RCP<Thyra_VectorSpace const>
  getVectorSpace() const = 0;
  virtual Teuchos::RCP<Thyra_VectorSpace const>
  getOverlapVectorSpace() const = 0;

  //! Get Field node vector space (owned and overlapped)
  virtual Teuchos::RCP<Thyra_VectorSpace const>
  getNodeVectorSpace(std::string const& field_name) const = 0;
  virtual Teuchos::RCP<Thyra_VectorSpace const>
  getOverlapNodeVectorSpace(std::string const& field_name) const = 0;

  //! Get Field vector space (owned and overlapped)
  virtual Teuchos::RCP<Thyra_VectorSpace const>
  getVectorSpace(std::string const& field_name) const = 0;
  virtual Teuchos::RCP<Thyra_VectorSpace const>
  getOverlapVectorSpace(std::string const& field_name) const = 0;

  //! Create a Jacobian operator (owned and overlapped)
  virtual Teuchos::RCP<Thyra_LinearOp>
  createJacobianOp() const = 0;
  virtual Teuchos::RCP<Thyra_LinearOp>
  createOverlapJacobianOp() const = 0;

  //! Get Node set lists
  virtual NodeSetList const&
  getNodeSets() const = 0;
  virtual NodeSetGIDsList const&
  getNodeSetGIDs() const = 0;
  virtual NodeSetCoordList const&
  getNodeSetCoords() const = 0;
  virtual const NodeGID2LIDMap&
  getNodeGID2LIDMap() const = 0;

  //! Get Side set lists
  virtual const SideSetList&
  getSideSets(int const ws) const = 0;

  //! Get map from (Ws, El, Local Node, Eq) -> unkLID
  virtual const Conn&
  getWsElNodeEqID() const = 0;

  //! Get map from (Ws, El, Local Node) -> unkGID
  virtual const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>>::type&
  getWsElNodeID() const = 0;

  //! Get IDArray for (Ws, Local Node, nComps) -> (local) NodeLID, works for
  //! both scalar and vector fields
  virtual std::vector<IDArray> const&
  getElNodeEqID(std::string const& field_name) const = 0;

  //! Get Dof Manager of field field_name
  virtual const NodalDOFManager&
  getDOFManager(std::string const& field_name) const = 0;

  //! Get Dof Manager of field field_name
  virtual const NodalDOFManager&
  getOverlapDOFManager(std::string const& field_name) const = 0;

  //! Retrieve coodinate ptr_field (ws, el, node)
  virtual const WorksetArray<
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*>>>::type&
  getCoords() const = 0;

  //! Get coordinates (overlap map).
  virtual const Teuchos::ArrayRCP<double>&
  getCoordinates() const = 0;
  //! Set coordinates (overlap map) for mesh adaptation.
  virtual void
  setCoordinates(const Teuchos::ArrayRCP<double const>& c) = 0;

  //! The reference configuration manager handles updating the reference
  //! configuration. This is only relevant, and also only optional, in the
  //! case of mesh adaptation.
  virtual void
  setReferenceConfigurationManager(
      const Teuchos::RCP<AAdapt::rc::Manager>& rcm) = 0;

#if defined(ALBANY_CONTACT)
  //! Get the contact manager
  virtual Teuchos::RCP<const ContactManager>
  getContactManager() const = 0;
#endif

  virtual const WorksetArray<Teuchos::ArrayRCP<double>>::type&
  getSphereVolume() const = 0;

  virtual const WorksetArray<Teuchos::ArrayRCP<double*>>::type&
  getLatticeOrientation() const = 0;

  virtual WorksetArray<Teuchos::ArrayRCP<double*>>::type const&
  getCellBoundaryIndicator() const = 0;

  virtual WorksetArray<Teuchos::ArrayRCP<double*>>::type const&
  getFaceBoundaryIndicator() const = 0;

  virtual WorksetArray<Teuchos::ArrayRCP<double*>>::type const&
  getEdgeBoundaryIndicator() const = 0;

  virtual std::map<GO, double*> const&
  getNodeBoundaryIndicator() const = 0;

  virtual bool
  hasCellBoundaryIndicator() const = 0;

  virtual bool
  hasFaceBoundaryIndicator() const = 0;

  virtual bool
  hasEdgeBoundaryIndicator() const = 0;

  virtual bool
  hasNodeBoundaryIndicator() const = 0;

  virtual void
  printElemGIDws() const
  {
  }

  virtual std::map<std::pair<int, int>, GO>
  getElemWsLIDGIDMap() const
  {
    return std::map<std::pair<int, int>, GO>();
  }

  virtual void
  printWsElNodeID() const
  {
  }

  //! Print the coords for mesh debugging
  virtual void
  printCoords() const = 0;

  //! Get sideSet discretizations map
  virtual const SideSetDiscretizationsType&
  getSideSetDiscretizations() const = 0;

  //! Get the map side_id->side_set_elem_id
  virtual std::map<std::string, std::map<GO, GO>> const&
  getSideToSideSetCellMap() const = 0;

  //! Get the map side_node_id->side_set_cell_node_id
  virtual std::map<std::string, std::map<GO, std::vector<int>>> const&
  getSideNodeNumerationMap() const = 0;

  //! Get MeshStruct
  virtual Teuchos::RCP<AbstractMeshStruct>
  getMeshStruct() const = 0;

  //! Set stateArrays
  virtual void
  setStateArrays(StateArrays& sa) = 0;

  //! Get stateArrays
  virtual StateArrays&
  getStateArrays() = 0;

  //! Get nodal parameters state info struct
  virtual const StateInfoStruct&
  getNodalParameterSIS() const = 0;

  //! Retrieve Vector (length num worksets) of element block names
  virtual const WorksetArray<std::string>::type&
  getWsEBNames() const = 0;

  //! Retrieve Vector (length num worksets) of Physics Index
  virtual const WorksetArray<int>::type&
  getWsPhysIndex() const = 0;

  //! Retrieve connectivity map from elementGID to workset
  virtual WsLIDList&
  getElemGIDws() = 0;
  virtual const WsLIDList&
  getElemGIDws() const = 0;

  //! Flag if solution has a restart values -- used in Init Cond
  virtual bool
  hasRestartSolution() const = 0;

  //! File time of restart solution
  virtual double
  restartDataTime() const = 0;

  //! Get number of spatial dimensions
  virtual int
  getNumDim() const = 0;

  //! Get number of total DOFs per node
  virtual int
  getNumEq() const = 0;

  //! Get Numbering for layered mesh (mesh structred in one direction)
  virtual Teuchos::RCP<LayeredMeshNumbering<LO>>
  getLayeredMeshNumbering() const = 0;

  // --- Get/set solution/residual/field vectors to/from mesh --- //

  virtual Teuchos::RCP<Thyra_Vector>
  getSolutionField(bool overlapped = false) const = 0;
  virtual Teuchos::RCP<Thyra_MultiVector>
  getSolutionMV(bool overlapped = false) const = 0;
  virtual void
  setResidualField(Thyra_Vector const& residual) = 0;
  virtual void
  getField(Thyra_Vector& field_vector, std::string const& field_name) const = 0;
  virtual void
  setField(
      Thyra_Vector const& field_vector,
      std::string const&  field_name,
      bool                overlapped) = 0;

  // --- Methods to write solution in the output file --- //

  //! Write the solution to the output file. Calls next two together.
  virtual void
  writeSolution(
      Thyra_Vector const& solution,
      double const        time,
      bool const          overlapped = false) = 0;
  virtual void
  writeSolution(
      Thyra_Vector const& solution,
      Thyra_Vector const& solution_dot,
      double const        time,
      bool const          overlapped = false) = 0;
  virtual void
  writeSolution(
      Thyra_Vector const& solution,
      Thyra_Vector const& solution_dot,
      Thyra_Vector const& solution_dotdot,
      double const        time,
      bool const          overlapped = false) = 0;
  virtual void
  writeSolutionMV(
      const Thyra_MultiVector& solution,
      double const             time,
      bool const               overlapped = false) = 0;
  //! Write the solution to the mesh database.
  virtual void
  writeSolutionToMeshDatabase(
      Thyra_Vector const& solution,
      double const        time,
      bool const          overlapped = false) = 0;
  virtual void
  writeSolutionToMeshDatabase(
      Thyra_Vector const& solution,
      Thyra_Vector const& solution_dot,
      double const        time,
      bool const          overlapped = false) = 0;
  virtual void
  writeSolutionToMeshDatabase(
      Thyra_Vector const& solution,
      Thyra_Vector const& solution_dot,
      Thyra_Vector const& solution_dotdot,
      double const        time,
      bool const          overlapped = false) = 0;
  virtual void
  writeSolutionMVToMeshDatabase(
      const Thyra_MultiVector& solution,
      double const             time,
      bool const               overlapped = false) = 0;

  //! Write the solution to file. Must call writeSolution first.
  virtual void
  writeSolutionToFile(
      Thyra_Vector const& solution,
      double const        time,
      bool const          overlapped = false) = 0;
  virtual void
  writeSolutionMVToFile(
      const Thyra_MultiVector& solution,
      double const             time,
      bool const               overlapped = false) = 0;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_DISCRETIZATION_HPP
