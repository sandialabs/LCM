// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <iostream>

#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_Macros.hpp"
#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "Albany_STKFieldContainerHelper.hpp"
#include "Albany_ThyraUtils.hpp"

// Start of STK stuff
#include <stk_io/IossBridge.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>

namespace Albany {

static char const* sol_tag_name[3] = {"Exodus Solution Name", "Exodus SolutionDot Name", "Exodus SolutionDotDot Name"};

static char const* sol_id_name[3] = {"solution", "solution_dot", "solution_dotdot"};

static char const* sol_dtk_id_name[3] = {"solution dtk", "solution_dot dtk", "solution_dotdot dtk"};

static char const* res_tag_name[1] = {
    "Exodus Residual Name",
};

static char const* sol_dtk_tag_name[3] = {"Exodus Solution DTK Name", "Exodus SolutionDot DTK Name", "Exodus SolutionDotDot DTK Name"};

static char const* res_id_name[1] = {
    "residual",
};

namespace {

// Tag a multi-component nodal field with its IOSS output type so the
// Exodus writer emits it as a vector (components named _x/_y/_z) rather
// than as N scalar variables (_1/_2/_3). The vector tagging is what lets
// ParaView's IOSS reader aggregate the components and apply the
// "Apply Displacements" filter. Lost in the STK simple-fields migration
// (commit 63379cb3f3) because the typed Field<double, Cartesian> used
// to carry the hint implicitly.
inline void
applyVectorOutputType(stk::mesh::FieldBase& field, int numComponents)
{
  if (numComponents == 2) {
    stk::io::set_field_output_type(field, stk::io::FieldOutputType::VECTOR_2D);
  } else if (numComponents == 3) {
    stk::io::set_field_output_type(field, stk::io::FieldOutputType::VECTOR_3D);
  }
}

}  // namespace

template <bool Interleaved>
OrdinarySTKFieldContainer<Interleaved>::OrdinarySTKFieldContainer(
    const Teuchos::RCP<Teuchos::ParameterList>&               params_,
    const Teuchos::RCP<stk::mesh::MetaData>&                  metaData_,
    const Teuchos::RCP<stk::mesh::BulkData>&                  bulkData_,
    int const                                                 neq_,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    int const                                                 numDim_,
    const Teuchos::RCP<StateInfoStruct>&                      sis)
    : GenericSTKFieldContainer<Interleaved>(params_, metaData_, bulkData_, neq_, numDim_), buildSphereVolume(false)
{
  typedef typename AbstractSTKFieldContainer::VectorFieldType       VFT;
  typedef typename AbstractSTKFieldContainer::ScalarFieldType       SFT;
  typedef typename AbstractSTKFieldContainer::SphereVolumeFieldType SVFT;

  int num_time_deriv = params_->get<int>("Number Of Time Derivatives");

  int output_dtk_field = params_->get<bool>("Output DTK Field to Exodus", false);

  // Start STK stuff
  this->coordinates_field = &metaData_->declare_field<double>(stk::topology::NODE_RANK, "coordinates");
  stk::mesh::put_field_on_mesh(*this->coordinates_field, metaData_->universal_part(), numDim_, nullptr);
  stk::io::set_field_role(*this->coordinates_field, Ioss::Field::MESH);
  applyVectorOutputType(*this->coordinates_field, numDim_);
  if (numDim_ == 3) {
    this->coordinates_field3d = this->coordinates_field;
  } else {
    this->coordinates_field3d = &metaData_->declare_field<double>(stk::topology::NODE_RANK, "coordinates3d");
    stk::mesh::put_field_on_mesh(*this->coordinates_field3d, metaData_->universal_part(), 3, nullptr);
    applyVectorOutputType(*this->coordinates_field3d, 3);
    if (params_->get<bool>("Export 3d coordinates field", false)) {
      stk::io::set_field_role(*this->coordinates_field3d, Ioss::Field::TRANSIENT);
    }
  }

  solution_field.resize(num_time_deriv + 1);
  solution_field_dtk.resize(num_time_deriv + 1);

  for (int num_vecs = 0; num_vecs <= num_time_deriv; num_vecs++) {
    solution_field[num_vecs] =
        &metaData_->declare_field<double>(stk::topology::NODE_RANK, params_->get<std::string>(sol_tag_name[num_vecs], sol_id_name[num_vecs]));
    stk::mesh::put_field_on_mesh(*solution_field[num_vecs], metaData_->universal_part(), neq_, nullptr);
    applyVectorOutputType(*solution_field[num_vecs], neq_);

    if (output_dtk_field == true) {
      solution_field_dtk[num_vecs] =
          &metaData_->declare_field<double>(stk::topology::NODE_RANK, params_->get<std::string>(sol_dtk_tag_name[num_vecs], sol_dtk_id_name[num_vecs]));
      stk::mesh::put_field_on_mesh(*solution_field_dtk[num_vecs], metaData_->universal_part(), neq_, nullptr);
      applyVectorOutputType(*solution_field_dtk[num_vecs], neq_);
    }

    stk::io::set_field_role(*solution_field[num_vecs], Ioss::Field::TRANSIENT);
    if (output_dtk_field == true) stk::io::set_field_role(*solution_field_dtk[num_vecs], Ioss::Field::TRANSIENT);
  }

  residual_field = &metaData_->declare_field<double>(stk::topology::NODE_RANK, params_->get<std::string>(res_tag_name[0], res_id_name[0]));
  stk::mesh::put_field_on_mesh(*residual_field, metaData_->universal_part(), neq_, nullptr);
  applyVectorOutputType(*residual_field, neq_);
  stk::io::set_field_role(*residual_field, Ioss::Field::TRANSIENT);

  // sphere volume is a mesh attribute read from a genesis mesh file containing
  // sphere element (used for peridynamics)
  this->sphereVolume_field = metaData_->template get_field<double>(stk::topology::ELEMENT_RANK, "volume");
  if (this->sphereVolume_field != 0) {
    buildSphereVolume = true;
    stk::io::set_field_role(*this->sphereVolume_field, Ioss::Field::ATTRIBUTE);
  }
  // If the problem requests that the initial guess at the solution equals the
  // input node coordinates, set that here
  /*
    if(std::find(req.begin(), req.end(), "Initial Guess Coords") != req.end()){
       this->copySTKField(this->coordinates_field, solution_field);
    }
  */

  this->addStateStructs(sis);

  initializeSTKAdaptation();
}

template <bool Interleaved>
void
OrdinarySTKFieldContainer<Interleaved>::initializeSTKAdaptation()
{
  using ISFT = AbstractSTKFieldContainer::IntScalarFieldType;
  using SFT  = AbstractSTKFieldContainer::ScalarFieldType;

  this->proc_rank_field = &this->metaData->template declare_field<int>(stk::topology::ELEMENT_RANK, "proc_rank");

  this->refine_field = &this->metaData->template declare_field<int>(stk::topology::ELEMENT_RANK, "refine_field");

  // Processor rank field, a scalar
  stk::mesh::put_field_on_mesh(*this->proc_rank_field, this->metaData->universal_part(), nullptr);

  stk::mesh::put_field_on_mesh(*this->refine_field, this->metaData->universal_part(), nullptr);

  // Failure state used for mesh adaptation
  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank <= stk::topology::ELEMENT_RANK; ++rank) {
    this->failure_state[rank] = &this->metaData->template declare_field<double>(rank, "failure_state");

    stk::mesh::put_field_on_mesh(*this->failure_state[rank], this->metaData->universal_part(), nullptr);
  }

  stk::io::set_field_role(*this->proc_rank_field, Ioss::Field::MESH);
  stk::io::set_field_role(*this->refine_field, Ioss::Field::MESH);
}

template <bool Interleaved>
void
OrdinarySTKFieldContainer<Interleaved>::fillVector(
    Thyra_Vector&                                field_vector,
    std::string const&                           field_name,
    stk::mesh::Selector&                         field_selection,
    Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
    const NodalDOFManager&                       nodalDofManager)
{
  fillVectorImpl(field_vector, field_name, field_selection, field_node_vs, nodalDofManager);
}

template <bool Interleaved>
void
OrdinarySTKFieldContainer<Interleaved>::fillSolnVector(Thyra_Vector& solution, stk::mesh::Selector& sel, Teuchos::RCP<Thyra_VectorSpace const> const& node_vs)
{
  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, Interleaved);

  fillVectorImpl(solution, solution_field[0]->name(), sel, node_vs, nodalDofManager);
}

template <bool Interleaved>
void
OrdinarySTKFieldContainer<Interleaved>::fillSolnMultiVector(
    Thyra_MultiVector&                           solution,
    stk::mesh::Selector&                         sel,
    Teuchos::RCP<Thyra_VectorSpace const> const& node_vs)
{
  typedef typename AbstractSTKFieldContainer::VectorFieldType VFT;

  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, Interleaved);

  for (int icomp = 0; icomp < solution.domain()->dim(); ++icomp) {
    fillVectorImpl(*solution.col(icomp), solution_field[icomp]->name(), sel, node_vs, nodalDofManager);
  }
}

template <bool Interleaved>
void
OrdinarySTKFieldContainer<Interleaved>::saveVector(
    Thyra_Vector const&                          field_vector,
    std::string const&                           field_name,
    stk::mesh::Selector&                         field_selection,
    Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
    const NodalDOFManager&                       nodalDofManager)
{
  saveVectorImpl(field_vector, field_name, field_selection, field_node_vs, nodalDofManager);
}

template <bool Interleaved>
void
OrdinarySTKFieldContainer<Interleaved>::saveSolnVector(
    Thyra_Vector const&                          solution,
    stk::mesh::Selector&                         sel,
    Teuchos::RCP<Thyra_VectorSpace const> const& node_vs)
{
  // IKT, FIXME? throw exception if num_time_deriv == 0 and we are calling this
  // function?

  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, Interleaved);

  saveVectorImpl(solution, solution_field[0]->name(), sel, node_vs, nodalDofManager);
}

template <bool Interleaved>
void
OrdinarySTKFieldContainer<Interleaved>::saveSolnVector(
    Thyra_Vector const&                          solution,
    Thyra_Vector const&                          solution_dot,
    stk::mesh::Selector&                         sel,
    Teuchos::RCP<Thyra_VectorSpace const> const& node_vs)
{
  // IKT, FIXME? throw exception if num_time_deriv == 0 and we are calling this
  // function?

  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, Interleaved);

  saveVectorImpl(solution, solution_field[0]->name(), sel, node_vs, nodalDofManager);
  saveVectorImpl(solution_dot, solution_field[1]->name(), sel, node_vs, nodalDofManager);
}

template <bool Interleaved>
void
OrdinarySTKFieldContainer<Interleaved>::saveSolnVector(
    Thyra_Vector const&                          solution,
    Thyra_Vector const&                          solution_dot,
    Thyra_Vector const&                          solution_dotdot,
    stk::mesh::Selector&                         sel,
    Teuchos::RCP<Thyra_VectorSpace const> const& node_vs)
{
  // IKT, FIXME? throw exception if num_time_deriv < 2 and we are calling this
  // function?

  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, Interleaved);

  saveVectorImpl(solution, solution_field[0]->name(), sel, node_vs, nodalDofManager);
  saveVectorImpl(solution_dot, solution_field[1]->name(), sel, node_vs, nodalDofManager);
  saveVectorImpl(solution_dotdot, solution_field[2]->name(), sel, node_vs, nodalDofManager);
}

template <bool Interleaved>
void
OrdinarySTKFieldContainer<Interleaved>::saveSolnMultiVector(
    const Thyra_MultiVector&                     solution,
    stk::mesh::Selector&                         sel,
    Teuchos::RCP<Thyra_VectorSpace const> const& node_vs)
{
  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, Interleaved);

  for (int icomp = 0; icomp < solution.domain()->dim(); ++icomp) {
    saveVectorImpl(*solution.col(icomp), solution_field[icomp]->name(), sel, node_vs, nodalDofManager);
  }
}

template <bool Interleaved>
void
OrdinarySTKFieldContainer<Interleaved>::saveResVector(Thyra_Vector const& res, stk::mesh::Selector& sel, Teuchos::RCP<Thyra_VectorSpace const> const& node_vs)
{
  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, Interleaved);

  saveVectorImpl(res, residual_field->name(), sel, node_vs, nodalDofManager);
}

template <bool Interleaved>
void
OrdinarySTKFieldContainer<Interleaved>::transferSolutionToCoords()
{
  STKFieldContainerHelper::copySTKField(*solution_field[0], *this->coordinates_field);
}

template <bool Interleaved>
void
OrdinarySTKFieldContainer<Interleaved>::fillVectorImpl(
    Thyra_Vector&                                field_vector,
    std::string const&                           field_name,
    stk::mesh::Selector&                         field_selection,
    Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
    const NodalDOFManager&                       nodalDofManager)
{
  auto* field = this->metaData->template get_field<double>(stk::topology::NODE_RANK, field_name);
  ALBANY_EXPECT(field != nullptr, "Error! Something went wrong while retrieving a field.\n");

  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  stk::mesh::BucketVector const& all_elements = this->bulkData->get_buckets(stk::topology::NODE_RANK, field_selection);

  auto field_node_vs_indexer = createGlobalLocalIndexer(field_node_vs);
  for (auto it = all_elements.begin(); it != all_elements.end(); ++it) {
    const stk::mesh::Bucket& bucket = **it;
    STKFieldContainerHelper::fillVector(field_vector, *field, field_node_vs_indexer, bucket, nodalDofManager, 0);
  }
}

template <bool Interleaved>
void
OrdinarySTKFieldContainer<Interleaved>::saveVectorImpl(
    Thyra_Vector const&                          field_vector,
    std::string const&                           field_name,
    stk::mesh::Selector&                         field_selection,
    Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
    const NodalDOFManager&                       nodalDofManager)
{
  auto* field = this->metaData->template get_field<double>(stk::topology::NODE_RANK, field_name);
  ALBANY_EXPECT(field != nullptr, "Error! Something went wrong while retrieving a field.\n");

  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  stk::mesh::BucketVector const& all_elements = this->bulkData->get_buckets(stk::topology::NODE_RANK, field_selection);

  auto field_node_vs_indexer = createGlobalLocalIndexer(field_node_vs);
  for (auto it = all_elements.begin(); it != all_elements.end(); ++it) {
    const stk::mesh::Bucket& bucket = **it;
    STKFieldContainerHelper::saveVector(field_vector, *field, field_node_vs_indexer, bucket, nodalDofManager, 0);
  }
}

}  // namespace Albany
