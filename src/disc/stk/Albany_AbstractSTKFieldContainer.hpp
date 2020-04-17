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
  // Tensor per Node/Cell  - (Node, Dim, Dim) or (Cell,Dim,Dim)
  typedef stk::mesh::Field<double, stk::mesh::Cartesian, stk::mesh::Cartesian>
      TensorFieldType;
  // Vector per Node/Cell  - (Node, Dim) or (Cell,Dim)
  typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;
  // Scalar per Node/Cell  - (Node) or (Cell)
  typedef stk::mesh::Field<double> ScalarFieldType;
  // One int scalar per Node/Cell  - (Node) or (Cell)
  typedef stk::mesh::Field<int> IntScalarFieldType;
  // int vector per Node/Cell  - (Node,Dim/VecDim) or (Cell,Dim/VecDim)
  typedef stk::mesh::Field<int, stk::mesh::Cartesian> IntVectorFieldType;

  typedef stk::mesh::Cartesian QPTag;  // need to invent shards::ArrayDimTag
  // Tensor per QP   - (Cell, QP, Dim, Dim)
  typedef stk::mesh::
      Field<double, QPTag, stk::mesh::Cartesian, stk::mesh::Cartesian>
          QPTensorFieldType;
  // Vector per QP   - (Cell, QP, Dim)
  typedef stk::mesh::Field<double, QPTag, stk::mesh::Cartesian>
      QPVectorFieldType;
  // One scalar per QP   - (Cell, QP)
  typedef stk::mesh::Field<double, QPTag> QPScalarFieldType;
  typedef stk::mesh::Field<double, stk::mesh::Cartesian3d>
      SphereVolumeFieldType;

  typedef std::vector<std::string const*> ScalarValueState;
  typedef std::vector<QPScalarFieldType*> QPScalarState;
  typedef std::vector<QPVectorFieldType*> QPVectorState;
  typedef std::vector<QPTensorFieldType*> QPTensorState;

  typedef std::vector<ScalarFieldType*> ScalarState;
  typedef std::vector<VectorFieldType*> VectorState;
  typedef std::vector<TensorFieldType*> TensorState;

  typedef std::map<std::string, double>              MeshScalarState;
  typedef std::map<std::string, std::vector<double>> MeshVectorState;

  typedef std::map<std::string, int>              MeshScalarIntegerState;
  typedef std::map<std::string, std::vector<int>> MeshVectorIntegerState;
  //! Destructor
  virtual ~AbstractSTKFieldContainer(){};

  virtual void
  addStateStructs(const Teuchos::RCP<Albany::StateInfoStruct>& sis) = 0;

  // Coordinates field ALWAYS in 3D
  const VectorFieldType*
  getCoordinatesField3d() const
  {
    return coordinates_field3d;
  }
  VectorFieldType*
  getCoordinatesField3d()
  {
    return coordinates_field3d;
  }

  const VectorFieldType*
  getCoordinatesField() const
  {
    return coordinates_field;
  }
  VectorFieldType*
  getCoordinatesField()
  {
    return coordinates_field;
  }
  IntScalarFieldType*
  getProcRankField()
  {
    return proc_rank_field;
  }
  IntScalarFieldType*
  getRefineField()
  {
    return refine_field;
  }
  IntScalarFieldType*
  getFailureState(stk::topology::rank_t rank)
  {
    return failure_state[rank];
  }
  stk::mesh::FieldBase*
  getCellBoundaryIndicator()
  {
    ALBANY_ASSERT(cell_boundary_indicator != nullptr);
    return cell_boundary_indicator;
  }
  stk::mesh::FieldBase*
  getFaceBoundaryIndicator()
  {
    ALBANY_ASSERT(face_boundary_indicator != nullptr);
    return face_boundary_indicator;
  }
  stk::mesh::FieldBase*
  getEdgeBoundaryIndicator()
  {
    ALBANY_ASSERT(edge_boundary_indicator != nullptr);
    return edge_boundary_indicator;
  }
  stk::mesh::FieldBase*
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
  stk::mesh::FieldBase*
  getLatticeOrientationField()
  {
    return latticeOrientation_field;
  }

  ScalarValueState&
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
  ScalarState&
  getCellScalarStates()
  {
    return cell_scalar_states;
  }
  VectorState&
  getCellVectorStates()
  {
    return cell_vector_states;
  }
  TensorState&
  getCellTensorStates()
  {
    return cell_tensor_states;
  }
  QPScalarState&
  getQPScalarStates()
  {
    return qpscalar_states;
  }
  QPVectorState&
  getQPVectorStates()
  {
    return qpvector_states;
  }
  QPTensorState&
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
  fillSolnVector(
      Thyra_Vector&                                soln,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs) = 0;
  virtual void
  fillVector(
      Thyra_Vector&                                field_vector,
      std::string const&                           field_name,
      stk::mesh::Selector&                         field_selection,
      Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
      const NodalDOFManager&                       nodalDofManager) = 0;
  virtual void
  fillSolnMultiVector(
      Thyra_MultiVector&                           soln,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs) = 0;
  virtual void
  saveVector(
      Thyra_Vector const&                          field_vector,
      std::string const&                           field_name,
      stk::mesh::Selector&                         field_selection,
      Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
      const NodalDOFManager&                       nodalDofManager) = 0;
  virtual void
  saveSolnVector(
      Thyra_Vector const&                          soln,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs) = 0;
  virtual void
  saveSolnVector(
      Thyra_Vector const&                          soln,
      Thyra_Vector const&                          soln_dot,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs) = 0;
  virtual void
  saveSolnVector(
      Thyra_Vector const&                          soln,
      Thyra_Vector const&                          soln_dot,
      Thyra_Vector const&                          soln_dotdot,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs) = 0;
  virtual void
  saveResVector(
      Thyra_Vector const&                          res,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs) = 0;
  virtual void
  saveSolnMultiVector(
      const Thyra_MultiVector&                     soln,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs) = 0;

  virtual void
  transferSolutionToCoords() = 0;

 protected:
  // Note: for 3d meshes, coordinates_field3d==coordinates_field (they point to
  // the same field).
  //       Otherwise, coordinates_field3d stores coordinates in 3d (useful for
  //       non-flat 2d meshes)
  VectorFieldType*      coordinates_field3d;
  VectorFieldType*      coordinates_field;
  IntScalarFieldType*   proc_rank_field;
  IntScalarFieldType*   refine_field;
  IntScalarFieldType*   failure_state[stk::topology::ELEMENT_RANK + 1];
  stk::mesh::FieldBase* cell_boundary_indicator;
  stk::mesh::FieldBase* face_boundary_indicator;
  stk::mesh::FieldBase* edge_boundary_indicator;
  stk::mesh::FieldBase* node_boundary_indicator;

  // Required for Peridynamics in LCM
  SphereVolumeFieldType* sphereVolume_field;

  // Required for certain LCM material models
  stk::mesh::FieldBase* latticeOrientation_field;

  ScalarValueState       scalarValue_states;
  MeshScalarState        mesh_scalar_states;
  MeshVectorState        mesh_vector_states;
  MeshScalarIntegerState mesh_scalar_integer_states;
  MeshVectorIntegerState mesh_vector_integer_states;
  ScalarState            cell_scalar_states;
  VectorState            cell_vector_states;
  TensorState            cell_tensor_states;
  QPScalarState          qpscalar_states;
  QPVectorState          qpvector_states;
  QPTensorState          qptensor_states;

  StateInfoStruct nodal_sis;
  StateInfoStruct nodal_parameter_sis;

  std::map<std::string, double> time;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_STK_FIELD_CONTAINER_HPP
