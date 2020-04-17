// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_ORDINARY_STK_FIELD_CONTAINER_HPP
#define ALBANY_ORDINARY_STK_FIELD_CONTAINER_HPP

#include "Albany_GenericSTKFieldContainer.hpp"

namespace Albany {

template <bool Interleaved>
class OrdinarySTKFieldContainer : public GenericSTKFieldContainer<Interleaved>
{
 public:
  OrdinarySTKFieldContainer(
      const Teuchos::RCP<Teuchos::ParameterList>&               params_,
      const Teuchos::RCP<stk::mesh::MetaData>&                  metaData_,
      const Teuchos::RCP<stk::mesh::BulkData>&                  bulkData_,
      int const                                                 neq_,
      const AbstractFieldContainer::FieldContainerRequirements& req,
      int const                                                 numDim_,
      const Teuchos::RCP<Albany::StateInfoStruct>&              sis);

  ~OrdinarySTKFieldContainer() = default;

  bool
  hasResidualField() const
  {
    return (residual_field != NULL);
  }
  bool
  hasSphereVolumeField() const
  {
    return buildSphereVolume;
  }
  bool
  hasLatticeOrientationField() const
  {
    return false;
  }

  bool
  hasCellBoundaryIndicatorField() const
  {
    return build_cell_boundary_indicator;
  }
  bool
  hasFaceBoundaryIndicatorField() const
  {
    return build_face_boundary_indicator;
  }
  bool
  hasEdgeBoundaryIndicatorField() const
  {
    return build_edge_boundary_indicator;
  }
  bool
  hasNodeBoundaryIndicatorField() const
  {
    return build_node_boundary_indicator;
  }

  Teuchos::Array<AbstractSTKFieldContainer::VectorFieldType*>
  getSolutionFieldArray()
  {
    return solution_field;
  }

  AbstractSTKFieldContainer::VectorFieldType*
  getSolutionField()
  {
    return solution_field[0];
  };

  AbstractSTKFieldContainer::VectorFieldType*
  getResidualField()
  {
    return residual_field;
  };
#if defined(ALBANY_DTK)
  Teuchos::Array<AbstractSTKFieldContainer::VectorFieldType*>
  getSolutionFieldDTKArray()
  {
    return solution_field_dtk;
  };

  AbstractSTKFieldContainer::VectorFieldType*
  getSolutionFieldDTK()
  {
    return solution_field_dtk[0];
  };
#endif

  void
  fillSolnVector(
      Thyra_Vector&                                soln,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs);
  void
  fillVector(
      Thyra_Vector&                                field_vector,
      std::string const&                           field_name,
      stk::mesh::Selector&                         field_selection,
      Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
      const NodalDOFManager&                       nodalDofManager);
  void
  fillSolnMultiVector(
      Thyra_MultiVector&                           soln,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs);
  void
  saveVector(
      Thyra_Vector const&                          field_vector,
      std::string const&                           field_name,
      stk::mesh::Selector&                         field_selection,
      Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
      const NodalDOFManager&                       nodalDofManager);
  void
  saveSolnVector(
      Thyra_Vector const&                          soln,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs);
  void
  saveSolnVector(
      Thyra_Vector const&                          soln,
      Thyra_Vector const&                          soln_dot,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs);
  void
  saveSolnVector(
      Thyra_Vector const&                          soln,
      Thyra_Vector const&                          soln_dot,
      Thyra_Vector const&                          soln_dotdot,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs);
  void
  saveResVector(
      Thyra_Vector const&                          res,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs);
  void
  saveSolnMultiVector(
      const Thyra_MultiVector&                     soln,
      stk::mesh::Selector&                         sel,
      Teuchos::RCP<Thyra_VectorSpace const> const& node_vs);

  void
  transferSolutionToCoords();

 private:
  void
  fillVectorImpl(
      Thyra_Vector&                                field_vector,
      std::string const&                           field_name,
      stk::mesh::Selector&                         field_selection,
      Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
      const NodalDOFManager&                       nodalDofManager);
  void
  saveVectorImpl(
      Thyra_Vector const&                          field_vector,
      std::string const&                           field_name,
      stk::mesh::Selector&                         field_selection,
      Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
      const NodalDOFManager&                       nodalDofManager);

  void
  initializeSTKAdaptation();

  bool buildSphereVolume;

  bool build_cell_boundary_indicator{false};
  bool build_face_boundary_indicator{false};
  bool build_edge_boundary_indicator{false};
  bool build_node_boundary_indicator{false};

  Teuchos::Array<AbstractSTKFieldContainer::VectorFieldType*> solution_field;
  Teuchos::Array<AbstractSTKFieldContainer::VectorFieldType*>
                                              solution_field_dtk;
  AbstractSTKFieldContainer::VectorFieldType* residual_field;
};

}  // namespace Albany

#endif  // ALBANY_ORDINARY_STK_FIELD_CONTAINER_HPP
