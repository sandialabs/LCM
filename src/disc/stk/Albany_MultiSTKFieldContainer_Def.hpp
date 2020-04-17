// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <iostream>
#include <string>

#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_Macros.hpp"
#include "Albany_MultiSTKFieldContainer.hpp"
#include "Albany_STKFieldContainerHelper.hpp"
#include "Albany_ThyraUtils.hpp"

// Start of STK stuff
#include <stk_io/IossBridge.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>

#include "Teuchos_VerboseObject.hpp"

namespace Albany {

static char const* sol_tag_name[3] = {"Exodus Solution Name",
                                      "Exodus SolutionDot Name",
                                      "Exodus SolutionDotDot Name"};

static char const* sol_id_name[3] = {"solution",
                                     "solution_dot",
                                     "solution_dotdot"};

static char const* res_tag_name = {
    "Exodus Residual Name",
};

static char const* res_id_name = {
    "residual",
};

template <bool Interleaved>
MultiSTKFieldContainer<Interleaved>::MultiSTKFieldContainer(
    const Teuchos::RCP<Teuchos::ParameterList>& params_,
    const Teuchos::RCP<stk::mesh::MetaData>&    metaData_,
    const Teuchos::RCP<stk::mesh::BulkData>&    bulkData_,
    int const                                   neq_,
    const AbstractFieldContainer::FieldContainerRequirements&
        req,  // TODO: remove this altogether?
              // AM: No, used in LCM for crystal plasticity and ACE
    int const                                          numDim_,
    const Teuchos::RCP<StateInfoStruct>&               sis,
    const Teuchos::Array<Teuchos::Array<std::string>>& solution_vector,
    const Teuchos::Array<std::string>&                 residual_vector)
    : GenericSTKFieldContainer<Interleaved>(
          params_,
          metaData_,
          bulkData_,
          neq_,
          numDim_),
      haveResidual(false),
      buildSphereVolume(false),
      buildLatticeOrientation(false)
{
  typedef typename AbstractSTKFieldContainer::VectorFieldType       VFT;
  typedef typename AbstractSTKFieldContainer::ScalarFieldType       SFT;
  typedef typename AbstractSTKFieldContainer::SphereVolumeFieldType SVFT;

  sol_vector_name.resize(solution_vector.size());
  sol_index.resize(solution_vector.size());

  // Check the input

  auto const num_derivs = solution_vector[0].size();
  for (auto i = 1; i < solution_vector.size(); ++i) {
    ALBANY_ASSERT(
        solution_vector[i].size() == num_derivs,
        "\n*** ERROR ***\n"
        "Number of derivatives for each variable is different.\n"
        "Check definition of solution vector and its derivatives.\n");
  }

  for (int vec_num = 0; vec_num < solution_vector.size(); vec_num++) {
    if (solution_vector[vec_num].size() ==
        0) {  // Do the default solution vector

      std::string name = params_->get<std::string>(
          sol_tag_name[vec_num], sol_id_name[vec_num]);
      VFT* solution =
          &metaData_->declare_field<VFT>(stk::topology::NODE_RANK, name);
      stk::mesh::put_field_on_mesh(
          *solution, metaData_->universal_part(), neq_, nullptr);
      stk::io::set_field_role(*solution, Ioss::Field::TRANSIENT);

      sol_vector_name[vec_num].push_back(name);
      sol_index[vec_num].push_back(this->neq);
    } else if (solution_vector[vec_num].size() == 1) {  // User is just renaming
                                                        // the entire solution
                                                        // vector

      VFT* solution = &metaData_->declare_field<VFT>(
          stk::topology::NODE_RANK, solution_vector[vec_num][0]);
      stk::mesh::put_field_on_mesh(
          *solution, metaData_->universal_part(), neq_, nullptr);
      stk::io::set_field_role(*solution, Ioss::Field::TRANSIENT);

      sol_vector_name[vec_num].push_back(solution_vector[vec_num][0]);
      sol_index[vec_num].push_back(neq_);

    } else {  // user is breaking up the solution into multiple fields

      // make sure the number of entries is even

      ALBANY_PANIC(
          (solution_vector[vec_num].size() % 2),
          "Error in input file: specification of solution vector layout is "
          "incorrect."
              << std::endl);

      int len, accum = 0;

      for (int i = 0; i < solution_vector[vec_num].size(); i += 2) {
        if (solution_vector[vec_num][i + 1] == "V") {
          len = numDim_;  // vector
          accum += len;
          VFT* solution = &metaData_->declare_field<VFT>(
              stk::topology::NODE_RANK, solution_vector[vec_num][i]);
          stk::mesh::put_field_on_mesh(
              *solution, metaData_->universal_part(), len, nullptr);
          stk::io::set_field_role(*solution, Ioss::Field::TRANSIENT);
          sol_vector_name[vec_num].push_back(solution_vector[vec_num][i]);
          sol_index[vec_num].push_back(len);

        } else if (solution_vector[vec_num][i + 1] == "S") {
          len = 1;  // scalar
          accum += len;
          SFT* solution = &metaData_->declare_field<SFT>(
              stk::topology::NODE_RANK, solution_vector[vec_num][i]);
          stk::mesh::put_field_on_mesh(
              *solution, metaData_->universal_part(), nullptr);
          stk::io::set_field_role(*solution, Ioss::Field::TRANSIENT);
          sol_vector_name[vec_num].push_back(solution_vector[vec_num][i]);
          sol_index[vec_num].push_back(len);

        } else {
          ALBANY_ABORT(
              "Error in input file: specification of solution vector layout is "
              "incorrect."
              << std::endl);
        }
      }
      ALBANY_PANIC(
          accum != neq_,
          "Error in input file: specification of solution vector layout is "
          "incorrect."
              << std::endl);
    }
  }

  // do the residual next

  if (residual_vector.size() == 0) {  // Do the default residual vector

    std::string name = params_->get<std::string>(res_tag_name, res_id_name);
    VFT*        residual =
        &metaData_->declare_field<VFT>(stk::topology::NODE_RANK, name);
    stk::mesh::put_field_on_mesh(
        *residual, metaData_->universal_part(), neq_, nullptr);
    stk::io::set_field_role(*residual, Ioss::Field::TRANSIENT);

    res_vector_name.push_back(name);
    res_index.push_back(neq_);

  } else if (residual_vector.size() == 1) {  // User is just renaming the entire
                                             // residual vector

    VFT* residual = &metaData_->declare_field<VFT>(
        stk::topology::NODE_RANK, residual_vector[0]);
    stk::mesh::put_field_on_mesh(
        *residual, metaData_->universal_part(), neq_, nullptr);
    stk::io::set_field_role(*residual, Ioss::Field::TRANSIENT);

    res_vector_name.push_back(residual_vector[0]);
    res_index.push_back(neq_);

  } else {  // user is breaking up the residual into multiple fields

    // make sure the number of entries is even

    ALBANY_PANIC(
        (residual_vector.size() % 2),
        "Error in input file: specification of residual vector layout is "
        "incorrect."
            << std::endl);

    int len, accum = 0;

    for (int i = 0; i < residual_vector.size(); i += 2) {
      if (residual_vector[i + 1] == "V") {
        len = numDim_;  // vector
        accum += len;
        VFT* residual = &metaData_->declare_field<VFT>(
            stk::topology::NODE_RANK, residual_vector[i]);
        stk::mesh::put_field_on_mesh(
            *residual, metaData_->universal_part(), len, nullptr);
        stk::io::set_field_role(*residual, Ioss::Field::TRANSIENT);
        res_vector_name.push_back(residual_vector[i]);
        res_index.push_back(len);

      } else if (residual_vector[i + 1] == "S") {
        len = 1;  // scalar
        accum += len;
        SFT* residual = &metaData_->declare_field<SFT>(
            stk::topology::NODE_RANK, residual_vector[i]);
        stk::mesh::put_field_on_mesh(
            *residual, metaData_->universal_part(), nullptr);
        stk::io::set_field_role(*residual, Ioss::Field::TRANSIENT);
        res_vector_name.push_back(residual_vector[i]);
        res_index.push_back(len);

      } else {
        ALBANY_ABORT(
            "Error in input file: specification of residual vector layout is "
            "incorrect."
            << std::endl);
      }
    }
    ALBANY_PANIC(
        accum != neq_,
        "Error in input file: specification of residual vector layout is "
        "incorrect."
            << std::endl);
  }

  haveResidual = true;

  // Do the coordinates
  this->coordinates_field =
      &metaData_->declare_field<VFT>(stk::topology::NODE_RANK, "coordinates");
  stk::mesh::put_field_on_mesh(
      *this->coordinates_field, metaData_->universal_part(), numDim_, nullptr);
  stk::io::set_field_role(*this->coordinates_field, Ioss::Field::MESH);

  if (numDim_ == 3) {
    this->coordinates_field3d = this->coordinates_field;
  } else {
    this->coordinates_field3d = &metaData_->declare_field<VFT>(
        stk::topology::NODE_RANK, "coordinates3d");
    stk::mesh::put_field_on_mesh(
        *this->coordinates_field3d, metaData_->universal_part(), 3, nullptr);
    if (params_->get<bool>("Export 3d coordinates field", false)) {
      stk::io::set_field_role(
          *this->coordinates_field3d, Ioss::Field::TRANSIENT);
    }
  }

  // sphere volume is a mesh attribute read from a genesis mesh file containing
  // sphere element (used for peridynamics)
  this->sphereVolume_field = metaData_->template get_field<SVFT>(
      stk::topology::ELEMENT_RANK, "volume");
  if (this->sphereVolume_field != 0) {
    buildSphereVolume = true;
    stk::io::set_field_role(*this->sphereVolume_field, Ioss::Field::ATTRIBUTE);
  }
  // lattice orientation is mesh attributes read from a genesis mesh file use
  // with certain solid mechanics material models
  bool hasLatticeOrientationFieldContainerRequirement =
      (std::find(req.begin(), req.end(), "Lattice_Orientation") != req.end());
  if (hasLatticeOrientationFieldContainerRequirement) {
    // STK says that attributes are of type Field<double,anonymous>[ name:
    // "extra_attribute_3" , #states: 1 ]
    this->latticeOrientation_field =
        metaData_->template get_field<stk::mesh::FieldBase>(
            stk::topology::ELEMENT_RANK, "extra_attribute_9");
    if (this->latticeOrientation_field != 0) {
      buildLatticeOrientation = true;
      stk::io::set_field_role(
          *this->latticeOrientation_field, Ioss::Field::ATTRIBUTE);
    }
  }

  this->addStateStructs(sis);

  initializeSTKAdaptation();

  bool const has_cell_boundary_indicator =
      (std::find(req.begin(), req.end(), "cell_boundary_indicator") !=
       req.end());
  bool const has_face_boundary_indicator =
      (std::find(req.begin(), req.end(), "face_boundary_indicator") !=
       req.end());
  bool const has_edge_boundary_indicator =
      (std::find(req.begin(), req.end(), "edge_boundary_indicator") !=
       req.end());
  bool const has_node_boundary_indicator =
      (std::find(req.begin(), req.end(), "node_boundary_indicator") !=
       req.end());
  if (has_cell_boundary_indicator) {
    this->cell_boundary_indicator =
        metaData_->template get_field<stk::mesh::FieldBase>(
            stk::topology::ELEMENT_RANK, "cell_boundary_indicator");
    if (this->cell_boundary_indicator != nullptr) {
      build_cell_boundary_indicator = true;
      stk::io::set_field_role(
          *this->cell_boundary_indicator, Ioss::Field::INFORMATION);
    }
  }
  if (has_face_boundary_indicator) {
    this->face_boundary_indicator =
        metaData_->template get_field<stk::mesh::FieldBase>(
            stk::topology::FACE_RANK, "face_boundary_indicator");
    if (this->face_boundary_indicator != nullptr) {
      build_face_boundary_indicator = true;
      stk::io::set_field_role(
          *this->face_boundary_indicator, Ioss::Field::INFORMATION);
    }
  }
  if (has_edge_boundary_indicator) {
    this->edge_boundary_indicator =
        metaData_->template get_field<stk::mesh::FieldBase>(
            stk::topology::EDGE_RANK, "edge_boundary_indicator");
    if (this->edge_boundary_indicator != nullptr) {
      build_edge_boundary_indicator = true;
      stk::io::set_field_role(
          *this->edge_boundary_indicator, Ioss::Field::INFORMATION);
    }
  }
  if (has_node_boundary_indicator) {
    this->node_boundary_indicator =
        metaData_->template get_field<stk::mesh::FieldBase>(
            stk::topology::NODE_RANK, "node_boundary_indicator");
    if (this->node_boundary_indicator != nullptr) {
      build_node_boundary_indicator = true;
      stk::io::set_field_role(
          *this->node_boundary_indicator, Ioss::Field::INFORMATION);
    }
  }
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::initializeSTKAdaptation()
{
  using ISFT = AbstractSTKFieldContainer::IntScalarFieldType;
  using SFT  = AbstractSTKFieldContainer::ScalarFieldType;

  this->proc_rank_field = &this->metaData->template declare_field<ISFT>(
      stk::topology::ELEMENT_RANK, "proc_rank");

  this->refine_field = &this->metaData->template declare_field<ISFT>(
      stk::topology::ELEMENT_RANK, "refine_field");

  // Processor rank field, a scalar
  stk::mesh::put_field_on_mesh(
      *this->proc_rank_field, this->metaData->universal_part(), nullptr);

  stk::mesh::put_field_on_mesh(
      *this->refine_field, this->metaData->universal_part(), nullptr);

  // Failure state used for mesh adaptation
  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK;
       rank <= stk::topology::ELEMENT_RANK;
       ++rank) {
    this->failure_state[rank] =
        &this->metaData->template declare_field<ISFT>(rank, "failure_state");
    stk::mesh::put_field_on_mesh(
        *this->failure_state[rank], this->metaData->universal_part(), nullptr);
  }

  // Cell boundary indicator
  this->cell_boundary_indicator = &this->metaData->template declare_field<SFT>(
      stk::topology::ELEMENT_RANK, "cell_boundary_indicator");
  stk::mesh::put_field_on_mesh(
      *this->cell_boundary_indicator,
      this->metaData->universal_part(),
      nullptr);

  // Face boundary indicator
  this->face_boundary_indicator = &this->metaData->template declare_field<SFT>(
      stk::topology::FACE_RANK, "face_boundary_indicator");
  stk::mesh::put_field_on_mesh(
      *this->face_boundary_indicator,
      this->metaData->universal_part(),
      nullptr);

  // Edge boundary indicator
  this->edge_boundary_indicator = &this->metaData->template declare_field<SFT>(
      stk::topology::EDGE_RANK, "edge_boundary_indicator");
  stk::mesh::put_field_on_mesh(
      *this->edge_boundary_indicator,
      this->metaData->universal_part(),
      nullptr);

  // Node boundary indicator
  this->node_boundary_indicator = &this->metaData->template declare_field<SFT>(
      stk::topology::NODE_RANK, "node_boundary_indicator");
  stk::mesh::put_field_on_mesh(
      *this->node_boundary_indicator,
      this->metaData->universal_part(),
      nullptr);

  stk::io::set_field_role(*this->proc_rank_field, Ioss::Field::MESH);
  stk::io::set_field_role(*this->refine_field, Ioss::Field::MESH);
  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK;
       rank <= stk::topology::ELEMENT_RANK;
       ++rank) {
    stk::io::set_field_role(*this->failure_state[rank], Ioss::Field::MESH);
  }
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::fillVector(
    Thyra_Vector&                                field_vector,
    std::string const&                           field_name,
    stk::mesh::Selector&                         field_selection,
    Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
    const NodalDOFManager&                       nodalDofManager)
{
  fillVectorImpl(
      field_vector,
      field_name,
      field_selection,
      field_node_vs,
      nodalDofManager,
      0);
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::fillSolnVector(
    Thyra_Vector&                                solution,
    stk::mesh::Selector&                         sel,
    Teuchos::RCP<Thyra_VectorSpace const> const& node_vs)
{
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, Interleaved);

  int offset = 0;
  for (int k = 0; k < sol_index[0].size(); k++) {
    fillVectorImpl(
        solution, sol_vector_name[0][k], sel, node_vs, nodalDofManager, offset);
    offset += sol_index[0][k];
  }
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::fillSolnMultiVector(
    Thyra_MultiVector&                           solution,
    stk::mesh::Selector&                         sel,
    Teuchos::RCP<Thyra_VectorSpace const> const& node_vs)
{
  using VFT = typename AbstractSTKFieldContainer::VectorFieldType;
  using SFT = typename AbstractSTKFieldContainer::ScalarFieldType;

  // Build a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, Interleaved);

  for (int icomp = 0; icomp < solution.domain()->dim(); ++icomp) {
    int offset = 0;

    for (int k = 0; k < sol_index[icomp].size(); k++) {
      fillVectorImpl(
          *solution.col(icomp),
          sol_vector_name[icomp][k],
          sel,
          node_vs,
          nodalDofManager,
          offset);
      offset += sol_index[icomp][k];
    }
  }
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::saveVector(
    Thyra_Vector const&                          field_vector,
    std::string const&                           field_name,
    stk::mesh::Selector&                         field_selection,
    Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
    const NodalDOFManager&                       nodalDofManager)
{
  saveVectorImpl(
      field_vector,
      field_name,
      field_selection,
      field_node_vs,
      nodalDofManager,
      0);
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::saveSolnVector(
    Thyra_Vector const&                          solution,
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

  int offset = 0;
  for (int k = 0; k < sol_index[0].size(); ++k) {
    // Recycle saveVectorImpl method
    saveVectorImpl(
        solution, sol_vector_name[0][k], sel, node_vs, nodalDofManager, offset);
    offset += sol_index[0][k];
  }
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::saveSolnVector(
    Thyra_Vector const& solution,
    Thyra_Vector const& /* solution_dot */,
    stk::mesh::Selector&                         sel,
    Teuchos::RCP<Thyra_VectorSpace const> const& node_vs)
{
  // TODO: why can't we save also solution_dot?
  Teuchos::RCP<Teuchos::FancyOStream> out =
      Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "IKT WARNING: calling MultiSTKFieldContainer::saveSolnVectorT with "
          "soln_dotT, but "
       << "this function has not been extended to write soln_dotT properly to "
          "the Exodus file.  Exodus "
       << "file will contain only soln, not soln_dot.\n";

  saveSolnVector(solution, sel, node_vs);
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::saveSolnVector(
    Thyra_Vector const& solution,
    Thyra_Vector const& /* solution_dot */,
    Thyra_Vector const& /* solution_dotdot */,
    stk::mesh::Selector&                         sel,
    Teuchos::RCP<Thyra_VectorSpace const> const& node_vs)
{
  // TODO: why can't we save also solution_dot?
  Teuchos::RCP<Teuchos::FancyOStream> out =
      Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "IKT WARNING: calling MultiSTKFieldContainer::saveSolnVectorT with "
          "soln_dotT and "
       << "soln_dotdotT, but this function has not been extended to write "
          "soln_dotT "
       << "and soln_dotdotT properly to the Exodus file.  Exodus "
       << "file will contain only soln, not soln_dot and soln_dotdot.\n";

  saveSolnVector(solution, sel, node_vs);
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::saveSolnMultiVector(
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
    int offset = 0;
    for (int k = 0; k < sol_index[icomp].size(); k++) {
      saveVectorImpl(
          *solution.col(icomp),
          sol_vector_name[icomp][k],
          sel,
          node_vs,
          nodalDofManager,
          offset);
      offset += sol_index[icomp][k];
    }
  }
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::saveResVector(
    Thyra_Vector const&                          res,
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

  int offset = 0;
  for (int k = 0; k < res_index.size(); k++) {
    saveVectorImpl(
        res, res_vector_name[k], sel, node_vs, nodalDofManager, offset);
    offset += res_index[k];
  }
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::transferSolutionToCoords()
{
  bool const MultiSTKFieldContainer_transferSolutionToCoords_not_implemented =
      true;
  ALBANY_PANIC(MultiSTKFieldContainer_transferSolutionToCoords_not_implemented);
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::fillVectorImpl(
    Thyra_Vector&                                field_vector,
    std::string const&                           field_name,
    stk::mesh::Selector&                         field_selection,
    Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
    const NodalDOFManager&                       nodalDofManager,
    int const                                    offset)
{
  using VFT = typename AbstractSTKFieldContainer::VectorFieldType;
  using SFT = typename AbstractSTKFieldContainer::ScalarFieldType;

  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  const stk::mesh::BucketVector& all_elements =
      this->bulkData->get_buckets(stk::topology::NODE_RANK, field_selection);

  auto* raw_field =
      this->metaData->get_field(stk::topology::NODE_RANK, field_name);
  ALBANY_EXPECT(
      raw_field != nullptr,
      "Error! Something went wrong while retrieving a field.\n");
  int const rank = raw_field->field_array_rank();

  auto field_node_vs_indexer = createGlobalLocalIndexer(field_node_vs);
  if (rank == 0) {
    using Helper     = STKFieldContainerHelper<SFT>;
    const SFT* field = this->metaData->template get_field<SFT>(
        stk::topology::NODE_RANK, field_name);
    for (auto it = all_elements.begin(); it != all_elements.end(); ++it) {
      const stk::mesh::Bucket& bucket = **it;
      Helper::fillVector(
          field_vector,
          *field,
          field_node_vs_indexer,
          bucket,
          nodalDofManager,
          offset);
    }
  } else if (rank == 1) {
    using Helper     = STKFieldContainerHelper<VFT>;
    const VFT* field = this->metaData->template get_field<VFT>(
        stk::topology::NODE_RANK, field_name);
    for (auto it = all_elements.begin(); it != all_elements.end(); ++it) {
      const stk::mesh::Bucket& bucket = **it;
      Helper::fillVector(
          field_vector,
          *field,
          field_node_vs_indexer,
          bucket,
          nodalDofManager,
          offset);
    }
  } else {
    ALBANY_ABORT("Error! Only scalar and vector fields supported so far.\n");
  }
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::saveVectorImpl(
    Thyra_Vector const&                          field_vector,
    std::string const&                           field_name,
    stk::mesh::Selector&                         field_selection,
    Teuchos::RCP<Thyra_VectorSpace const> const& field_node_vs,
    const NodalDOFManager&                       nodalDofManager,
    int const                                    offset)
{
  using VFT = typename AbstractSTKFieldContainer::VectorFieldType;
  using SFT = typename AbstractSTKFieldContainer::ScalarFieldType;

  auto* raw_field =
      this->metaData->get_field(stk::topology::NODE_RANK, field_name);
  ALBANY_EXPECT(
      raw_field != nullptr,
      "Error! Something went wrong while retrieving a field.\n");
  int const rank = raw_field->field_array_rank();

  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  stk::mesh::BucketVector const& all_elements =
      this->bulkData->get_buckets(stk::topology::NODE_RANK, field_selection);

  auto field_node_vs_indexer = createGlobalLocalIndexer(field_node_vs);
  if (rank == 0) {
    using Helper = STKFieldContainerHelper<SFT>;
    SFT* field   = this->metaData->template get_field<SFT>(
        stk::topology::NODE_RANK, field_name);
    for (auto it = all_elements.begin(); it != all_elements.end(); ++it) {
      const stk::mesh::Bucket& bucket = **it;
      Helper::saveVector(
          field_vector,
          *field,
          field_node_vs_indexer,
          bucket,
          nodalDofManager,
          offset);
    }
  } else if (rank == 1) {
    using Helper = STKFieldContainerHelper<VFT>;
    VFT* field   = this->metaData->template get_field<VFT>(
        stk::topology::NODE_RANK, field_name);
    for (auto it = all_elements.begin(); it != all_elements.end(); ++it) {
      const stk::mesh::Bucket& bucket = **it;
      Helper::saveVector(
          field_vector,
          *field,
          field_node_vs_indexer,
          bucket,
          nodalDofManager,
          offset);
    }
  } else {
    ALBANY_ABORT("Error! Only scalar and vector fields supported so far.\n");
  }
}

}  // namespace Albany
