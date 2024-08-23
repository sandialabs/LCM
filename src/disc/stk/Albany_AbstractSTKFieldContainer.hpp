// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_ABSTRACT_STK_FIELD_CONTAINER_HPP
#define ALBANY_ABSTRACT_STK_FIELD_CONTAINER_HPP

#include "Albany_config.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

// This include is added in Tpetra branch to get all the necessary
// Tpetra includes (e.g., Tpetra_Vector.hpp, Tpetra_Map.hpp, etc.)
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldTraits.hpp>

#include "Albany_AbstractFieldContainer.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_NodalDOFManager.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_Utils.hpp"

namespace Albany {

/*!
 * \brief Abstract interface for an STK field container
 *
 */
class AbstractSTKFieldContainer : public AbstractFieldContainer
{
 public:
  typedef stk::mesh::Field<double> SphereVolumeFieldType;

  // STK field (scalar/vector/tensor , Node/Cell)
  using STKFieldType = stk::mesh::Field<double>;
  // STK int field
  using STKIntState  = stk::mesh::Field<int>;

  using ValueState = std::vector<const std::string*>;
  using STKState   = std::vector<STKFieldType*>;

  using MeshScalarState          = std::map<std::string, double>;
  using MeshVectorState          = std::map<std::string, std::vector<double>>;
  using MeshScalarIntegerState   = std::map<std::string, int>;
  using MeshScalarInteger64State = std::map<std::string, GO>;
  using MeshVectorIntegerState   = std::map<std::string, std::vector<int>>;
  //! Destructor
  virtual ~AbstractSTKFieldContainer(){};

  virtual void
  addStateStructs(const Teuchos::RCP<Albany::StateInfoStruct>& sis) = 0;

  // Coordinates field ALWAYS in 3D
  const STKFieldType*
  getCoordinatesField3d() const
  {
    return coordinates_field3d;
  }
  STKFieldType*
  getCoordinatesField3d()
  {
    return coordinates_field3d;
  }

  const STKFieldType*
  getCoordinatesField() const
  {
    return coordinates_field;
  }
  STKFieldType*
  getCoordinatesField()
  {
    return coordinates_field;
  }
  STKIntState*
  getProcRankField()
  {
    return proc_rank_field;
  }
  STKIntState*
  getRefineField()
  {
    return refine_field;
  }
  ScalarFieldType*
  getFailureState(stk::topology::rank_t rank)
  {
    return failure_state[rank];
  }
  stk::mesh::Field<double>*
  getCellBoundaryIndicator()
  {
    ALBANY_ASSERT(cell_boundary_indicator != nullptr);
    return cell_boundary_indicator;
  }
  stk::mesh::Field<double>*
  getFaceBoundaryIndicator()
  {
    ALBANY_ASSERT(face_boundary_indicator != nullptr);
    return face_boundary_indicator;
  }
  stk::mesh::Field<double>*
  getEdgeBoundaryIndicator()
  {
    ALBANY_ASSERT(edge_boundary_indicator != nullptr);
    return edge_boundary_indicator;
  }
  stk::mesh::Field<double>*
  getNodeBoundaryIndicator()
  {
    ALBANY_ASSERT(node_boundary_indicator != nullptr);
    return node_boundary_indicator;
  }
  SphereVolumeFieldType*
  getSphereVolumeField()
  {
    return sphereVolume_field;
  }
  stk::mesh::Field<double>*
  getLatticeOrientationField()
  {
    return latticeOrientation_field;
  }

  ValueState&
  getScalarValueStates()
  {
    return scalarValue_states;
  }
  MeshScalarState&
  getMeshScalarStates()
  {
    return mesh_scalar_states;
  }
  MeshVectorState&
  getMeshVectorStates()
  {
    return mesh_vector_states;
  }
  MeshScalarIntegerState&
  getMeshScalarIntegerStates()
  {
    return mesh_scalar_integer_states;
  }
  MeshVectorIntegerState&
  getMeshVectorIntegerStates()
  {
    return mesh_vector_integer_states;
  }
  STKState&
  getCellScalarStates()
  {
    return cell_scalar_states;
  }
  STKState&
  getCellVectorStates()
  {
    return cell_vector_states;
  }
  STKState&
  getCellTensorStates()
  {
    return cell_tensor_states;
  }
  STKState&
  getQPScalarStates()
  {
    return qpscalar_states;
  }
  STKState&
  getQPVectorStates()
  {
    return qpvector_states;
  }
  STKState&
  getQPTensorStates()
  {
    return qptensor_states;
  }
  const StateInfoStruct&
  getNodalSIS() const
  {
    return nodal_sis;
  }
  const StateInfoStruct&
  getNodalParameterSIS() const
  {
    return nodal_parameter_sis;
  }

  virtual bool
  hasResidualField() const = 0;
  virtual bool
  hasSphereVolumeField() const = 0;
  virtual bool
  hasLatticeOrientationField() const = 0;

  virtual bool
  hasCellBoundaryIndicatorField() const = 0;
  virtual bool
  hasFaceBoundaryIndicatorField() const = 0;
  virtual bool
  hasEdgeBoundaryIndicatorField() const = 0;
  virtual bool
  hasNodeBoundaryIndicatorField() const = 0;

  std::map<std::string, double>&
  getTime()
  {
    return time;
  }

  virtual void
  fillSolnVector(Thyra_Vector& soln, stk::mesh::Selector& sel, Teuchos::RCP<Thyra_VectorSpace const> const& node_vs) = 0;
  virtual void
  fillVector(
      Thyra_Vector&                                field_vector,
      std::string const&                           field_name,
      stk::mesh::Selector&                         field_selection,
      Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
      const NodalDOFManager&                       nodalDofManager) = 0;
  virtual void
  fillSolnMultiVector(Thyra_MultiVector& soln, stk::mesh::Selector& sel, Teuchos::RCP<Thyra_VectorSpace const> const& node_vs) = 0;
  virtual void
  saveVector(
      Thyra_Vector const&                          field_vector,
      std::string const&                           field_name,
      stk::mesh::Selector&                         field_selection,
      Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
      const NodalDOFManager&                       nodalDofManager) = 0;
  virtual void
  saveSolnVector(Thyra_Vector const& soln, stk::mesh::Selector& sel, Teuchos::RCP<Thyra_VectorSpace const> const& node_vs) = 0;
  virtual void
  saveSolnVector(Thyra_Vector const& soln, Thyra_Vector const& soln_dot, stk::mesh::Selector& sel, Teuchos::RCP<Thyra_VectorSpace const> const& node_vs) = 0;
  virtual void
  saveSolnVector(
      Thyra_Vector const&                          soln,
      Thyra_Vector const&                          soln_dot,
      Thyra_Vector const&                          soln_dotdot,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs) = 0;
  virtual void
  saveResVector(Thyra_Vector const& res, stk::mesh::Selector& sel, Teuchos::RCP<Thyra_VectorSpace const> const& node_vs) = 0;
  virtual void
  saveSolnMultiVector(const Thyra_MultiVector& soln, stk::mesh::Selector& sel, Teuchos::RCP<Thyra_VectorSpace const> const& node_vs) = 0;

  virtual void
  transferSolutionToCoords() = 0;

 protected:
  // Note: for 3d meshes, coordinates_field3d==coordinates_field (they point to
  // the same field).
  //       Otherwise, coordinates_field3d stores coordinates in 3d (useful for
  //       non-flat 2d meshes)
  STKFieldType*          coordinates_field3d;
  STKFieldType*          coordinates_field;
  STKIntState*       proc_rank_field;
  STKIntState*       refine_field;
  ScalarFieldType*          failure_state[stk::topology::ELEMENT_RANK + 1];
  stk::mesh::Field<double>* cell_boundary_indicator;
  stk::mesh::Field<double>* face_boundary_indicator;
  stk::mesh::Field<double>* edge_boundary_indicator;
  stk::mesh::Field<double>* node_boundary_indicator;

  // Required for Peridynamics in LCM
  SphereVolumeFieldType* sphereVolume_field;

  // Required for certain LCM material models
  stk::mesh::Field<double>* latticeOrientation_field;

  ValueState       scalarValue_states;
  MeshScalarState        mesh_scalar_states;
  MeshVectorState        mesh_vector_states;
  MeshScalarIntegerState mesh_scalar_integer_states;
  MeshVectorIntegerState mesh_vector_integer_states;
  STKState            cell_scalar_states;
  STKState            cell_vector_states;
  STKState            cell_tensor_states;
  STKState          qpscalar_states;
  STKState          qpvector_states;
  STKState          qptensor_states;

  StateInfoStruct nodal_sis;
  StateInfoStruct nodal_parameter_sis;

  std::map<std::string, double> time;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_STK_FIELD_CONTAINER_HPP
