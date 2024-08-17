// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_Application.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_Macros.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Neumann.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Topology.hpp"

// uncomment the following line if you want debug output to be printed to screen
// #define ACE_WAVE_PRESS_DEBUG_OUTPUT
// #define ACE_WAVE_PRESS_EXTREME_DEBUG_OUTPUT

// The following vars are for the ACE Wave Pressure NBC
static std::vector<int>    ace_press_index;
static std::vector<double> previous_times;
static std::vector<bool>   is_initial_time;

namespace PHAL {

//*****
template <typename EvalT, typename Traits>
NeumannBase<EvalT, Traits>::NeumannBase(Teuchos::ParameterList& p)
    : app_(p.get<Teuchos::RCP<Albany::Application>>("Application", Teuchos::null)),
      dl(p.get<Teuchos::RCP<Albany::Layouts>>("Layouts Struct")),
      meshSpecs(p.get<Teuchos::RCP<Albany::MeshSpecsStruct>>("Mesh Specs Struct")),
      offset(p.get<Teuchos::Array<int>>("Equation Offset")),
      sideSetID(p.get<std::string>("Side Set ID")),
      coordVec(p.get<std::string>("Coordinate Vector Name"), dl->vertices_vector)
{
  // the input.xml string "NBC on SS sidelist_12 for DOF T set dudn" (or
  // something like it)
  name = p.get<std::string>("Neumann Input String");

  // The input.xml argument for the above string
  inputValues = p.get<Teuchos::Array<double>>("Neumann Input Value");

  // The input.xml argument for the above string
  inputConditions = p.get<std::string>("Neumann Input Conditions");

  // The DOF offsets are contained in the Equation Offset array. The length of
  // this array are the number of DOFs we will set each call
  numDOFsSet = offset.size();

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib>>("Parameter Library");

  // If we are doing a Neumann internal boundary with a "scaled jump",
  // build a scale lookup table from the materialDB file (this must exist)

  int position;

  // User has specified conditions on sideset normal
  if (inputConditions == "scaled jump") {
    bc_type   = INTJUMP;
    const_val = inputValues[0];
    this->registerSacadoParameter(name, paramLib);

    // Build a vector to hold the scaling from the material DB
    matScaling.resize(meshSpecs->ebNameToIndex.size(), 1.0);

    //! Material database - holds the scaling we need
    Teuchos::RCP<Albany::MaterialDatabase> materialDB = p.get<Teuchos::RCP<Albany::MaterialDatabase>>("MaterialDB");

    // iterator over all ebnames in the mesh
    std::map<std::string, int>::const_iterator it;
    for (it = meshSpecs->ebNameToIndex.begin(); it != meshSpecs->ebNameToIndex.end(); it++) {
      ALBANY_PANIC(
          !materialDB->isElementBlockParam(it->first, "Flux Scale"),
          "Cannot locate the value of \"Flux Scale\" for element block " << it->first << " in the material database");

      matScaling[it->second] = materialDB->getElementBlockParam<double>(it->first, "Flux Scale");
    }
  } else if (inputConditions == "robin" || inputConditions == "radiate") {
    bc_type = (inputConditions == "radiate" ? STEFAN_BOLTZMANN : ROBIN);

    robin_vals[0] = inputValues[0];  // dof_value
    robin_vals[1] = inputValues[1];  // coeff multiplying difference 'dof^4 - dof_value^4'
                                     // (or 'dof-dof_value' for robin)

    for (int i = 0; i < 2; i++) {
      std::stringstream ss;
      ss << name << "[" << i << "]";
      this->registerSacadoParameter(ss.str(), paramLib);
    }

    vectorDOF = p.get<bool>("Vector Field");

    dof = decltype(dof)(p.get<std::string>("DOF Name"), p.get<Teuchos::RCP<PHX::DataLayout>>("DOF Data Layout"));
    this->addDependentField(dof);
  } else if (inputConditions == "closed_form") {
    bc_type = CLOSED_FORM;
  }

  // else parse the input to determine what type of BC to calculate

  // is there a "(" in the string?
  else if ((position = inputConditions.find_first_of("(")) != std::string::npos) {
    if (inputConditions.find("t_x", position + 1)) {
      // User has specified conditions in base coords
      bc_type = TRACTION;
    } else {
      // User has specified conditions in base coords
      bc_type = COORD;
    }

    dudx.resize(meshSpecs->numDim);
    for (int i = 0; i < dudx.size(); i++) dudx[i] = inputValues[i];

    for (int i = 0; i < dudx.size(); i++) {
      std::stringstream ss;
      ss << name << "[" << i << "]";
      this->registerSacadoParameter(ss.str(), paramLib);
    }
  } else if (inputConditions == "P") {  // Pressure boundary condition for
                                        // Elasticity

    // User has specified a pressure condition
    bc_type   = PRESS;
    const_val = inputValues[0];
    this->registerSacadoParameter(name, paramLib);

  } else if (inputConditions == "wave_pressure") {  // ACE Wave Pressure boundary condition

    // User has specified a pressure condition
    bc_type = ACEPRESS;
    this->registerSacadoParameter(name, paramLib);

  } else if (inputConditions == "wave_pressure_hydrostatic") {  // ACE Wave Pressure boundary condition

    // User has specified a pressure condition
    bc_type = ACEPRESS_HYDROSTATIC;
    this->registerSacadoParameter(name, paramLib);

  } else {
    // User has specified conditions on sideset normal
    bc_type   = NORMAL;
    const_val = inputValues[0];
    this->registerSacadoParameter(name, paramLib);
  }

  this->addDependentField(coordVec);

  PHX::Tag<ScalarT> fieldTag(name, dl->dummy);

  this->addEvaluatedField(fieldTag);

  // Build element and side integration support

  const CellTopologyData* const elem_top = &meshSpecs->ctd;

  intrepidBasis = Albany::getIntrepid2Basis(*elem_top);

  cellType = Teuchos::rcp(new shards::CellTopology(elem_top));

  Intrepid2::DefaultCubatureFactory cubFactory;
  cubatureCell = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs->cubatureDegree);

  int cubatureDegree = (p.get<int>("Cubature Degree") > 0) ? p.get<int>("Cubature Degree") : meshSpecs->cubatureDegree;

  numSidesOnElem = elem_top->side_count;
  sideType.resize(numSidesOnElem);
  cubatureSide.resize(numSidesOnElem);
  side_type.resize(numSidesOnElem);

  // Build containers that depend on side topology
  char const* sideTypeName;

  maxSideDim = maxNumQpSide = 0;
  for (int i = 0; i < numSidesOnElem; ++i) {
    sideType[i]     = Teuchos::rcp(new shards::CellTopology(elem_top->side[i].topology));
    cubatureSide[i] = cubFactory.create<PHX::Device, RealType, RealType>(*sideType[i], cubatureDegree);
    maxSideDim      = std::max(maxSideDim, (int)sideType[i]->getDimension());
    maxNumQpSide    = std::max(maxNumQpSide, (int)cubatureSide[i]->getNumPoints());
    sideTypeName    = sideType[i]->getName();
    if (strncasecmp(sideTypeName, "Line", 4) == 0)
      side_type[i] = LINE;
    else if (strncasecmp(sideTypeName, "Tri", 3) == 0)
      side_type[i] = TRI;
    else if (strncasecmp(sideTypeName, "Quad", 4) == 0)
      side_type[i] = QUAD;
    else
      ALBANY_ABORT("PHAL_Neumann: side type : " << sideTypeName << " is not supported." << std::endl);
  }

  numNodes = intrepidBasis->getCardinality();

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->qp_tensor->dimensions(dim);
  numCells = dim[0];
  numQPs   = dim[1];
  cellDims = dim[2];

  this->setName(name + PHX::print<EvalT>());
}

//*****
template <typename EvalT, typename Traits>
void
NeumannBase<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec, fm);
  if (inputConditions == "robin" || inputConditions == "radiate") {
    this->utils.setFieldData(dof, fm);
    dofSide_buffer = Kokkos::createDynRankView(dof.get_view(), "dofSide", numCells * maxNumQpSide * numDOFsSet);
  }
  // Note, we do not need to add dependent field to fm here for output - that is
  // done by Neumann Aggregator

  // Allocate Temporary Views
  physPointsCell_buffer = Kokkos::createDynRankView(coordVec.get_view(), "physPointsCell", numCells * numNodes * cellDims);
  temporary_buffer      = Kokkos::createDynRankView(coordVec.get_view(), "temporary_buffer", numCells * maxNumQpSide * cellDims * cellDims);

  cubPointsSide_buffer       = Kokkos::DynRankView<RealType, PHX::Device>("cubPointsSide", maxNumQpSide * maxSideDim);
  refPointsSide_buffer       = Kokkos::DynRankView<RealType, PHX::Device>("refPointsSide", maxNumQpSide * cellDims);
  cubWeightsSide_buffer      = Kokkos::DynRankView<RealType, PHX::Device>("cubWeightsSide", maxNumQpSide);
  basis_refPointsSide_buffer = Kokkos::DynRankView<RealType, PHX::Device>("basis_refPointsSide", numNodes * maxNumQpSide);

  physPointsSide_buffer            = Kokkos::createDynRankView(coordVec.get_view(), "physPointsSide", numCells * maxNumQpSide * cellDims);
  jacobianSide_buffer              = Kokkos::createDynRankView(coordVec.get_view(), "jacobianSide", numCells * maxNumQpSide * cellDims * cellDims);
  jacobianSide_det_buffer          = Kokkos::createDynRankView(coordVec.get_view(), "jacobianSide", numCells * maxNumQpSide);
  weighted_measure_buffer          = Kokkos::createDynRankView(coordVec.get_view(), "weighted_measure", numCells * maxNumQpSide);
  trans_basis_refPointsSide_buffer = Kokkos::createDynRankView(coordVec.get_view(), "trans_basis_refPointsSide", numCells * numNodes * maxNumQpSide);
  weighted_trans_basis_refPointsSide_buffer =
      Kokkos::createDynRankView(coordVec.get_view(), "weighted_trans_basis_refPointsSide", numCells * numNodes * maxNumQpSide);
  side_normals_buffer   = Kokkos::createDynRankView(coordVec.get_view(), "side_normals", numCells * maxNumQpSide * cellDims);
  normal_lengths_buffer = Kokkos::createDynRankView(coordVec.get_view(), "normal_lengths", numCells * maxNumQpSide);

  if (inputConditions == "robin" || inputConditions == "radiate") {
    dofCell_buffer = Kokkos::createDynRankView(dof.get_view(), "dofCell", numCells, numNodes, numDOFsSet);
  }

  d.fill_field_dependencies(this->dependentFields(), this->evaluatedFields());
}

template <typename EvalT, typename Traits>
void
NeumannBase<EvalT, Traits>::evaluateNeumannContribution(typename Traits::EvalData workset)
{
  const int numWorksets = workset.num_worksets;
  const int worksetNum  = workset.workset_num;
  if (ace_press_index.size() != numWorksets) {
    ace_press_index.resize(numWorksets, 0.0);
  }
  if (previous_times.size() != numWorksets) {
    previous_times.resize(numWorksets, 0.0);
  }
  if (is_initial_time.size() != numWorksets) {
    is_initial_time.resize(numWorksets, true);
  }
  auto rcp_disc = workset.disc;
  auto stk_disc = dynamic_cast<Albany::STKDiscretization*>(rcp_disc.get());
  commT         = stk_disc->getComm();

  // setJacobian only needs to be RealType since the data type is only
  //  used internally for Basis Fns on reference elements, which are
  //  not functions of coordinates. This save 18min of compile time!!!

  // GAH: Note that this loosely follows from
  // $TRILINOS_DIR/packages/intrepid/test/Discretization/Basis/HGRAD_QUAD_C1_FEM/test_02.cpp

  if (workset.sideSets == Teuchos::null || this->sideSetID.length() == 0) {
    ALBANY_ABORT("Side sets defined in input file but not properly specified on the mesh" << std::endl);
  }

  // neumann data type is always ScalarT, but the deriv dimension
  // actually needed depends on BC type. For many it just needs
  // deriv dimensions from MeshScalarT (cloned from coordVec).
  // "data" is same as neumann -- always ScalarT but not always
  // with full deriv dimension of a ScalarT variable.

  // std::cout << "NN0 " << std::endl;
  switch (bc_type) {
    case INTJUMP:
      neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device>>(coordVec.get_view(), "DDN", numCells, numNodes, numDOFsSet);
      break;
    case ROBIN:
      neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device>>(dof.get_view(), "DDN", numCells, numNodes, numDOFsSet);
      break;
    case STEFAN_BOLTZMANN:
      neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device>>(dof.get_view(), "DDN", numCells, numNodes, numDOFsSet);
      break;
    case NORMAL:
      neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device>>(coordVec.get_view(), "DDN", numCells, numNodes, numDOFsSet);
      break;
    case PRESS:
      neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device>>(coordVec.get_view(), "DDN", numCells, numNodes, numDOFsSet);
      break;
    case TRACTION:
      neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device>>(coordVec.get_view(), "DDN", numCells, numNodes, numDOFsSet);
      break;
    case CLOSED_FORM:
      neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device>>(dof.get_view(), "DDN", numCells, numNodes, numDOFsSet);
      break;
    default:
      neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device>>(coordVec.get_view(), "DDN", numCells, numNodes, numDOFsSet);
      break;
  }

  data_buffer = Kokkos::createDynRankView(neumann, "data", numCells * maxNumQpSide * numDOFsSet);

  // Needed?
  Kokkos::deep_copy(neumann, 0.0);

  auto const ss_id   = this->sideSetID;
  auto&      ss_list = *(workset.sideSets);
  auto       it      = ss_list.find(ss_id);

  if (it == ss_list.end()) return;

  auto& side_set = it->second;

  // If this is an ACE erodible side set, discard any information in it and
  // rebuild it by querying the topology structure for erodible faces.
  // This is because there is mesh adaptation, material may be removed, and
  // the boundary changes, and so the NBC needs to propagate with the moving
  // boundary.
  auto const is_erodible = ss_id.find("erodible") != std::string::npos;
#if 0
  {
    if (is_erodible == true) {
      side_set.clear();
      auto               topo_rcp            = workset.topology;
      auto&              bulk_data           = topo_rcp->get_bulk_data();
      auto               erodible_cells      = topo_rcp->getErodibleCells();
      auto               erodible_cell_gids  = topo_rcp->getEntityGIDs(erodible_cells);
      auto&              stk_disc            = topo_rcp->get_stk_discretization();
      auto               elem_gid_to_wslid   = stk_disc.getElemGIDws();
      auto               stk_mesh_struct_rcp = stk_disc.getSTKMeshStruct();
      auto               stk_mesh_specs_rcp  = stk_mesh_struct_rcp->getMeshSpecs()[0];
      auto               ws_eb_names         = stk_disc.getWsEBNames();
      auto const         elem_rank           = stk::topology::ELEM_RANK;
      auto const         face_rank           = stk::topology::FACE_RANK;
      Albany::SideStruct entry;
      for (auto cell : erodible_cells) {
        auto const* relations     = bulk_data.begin(cell, face_rank);
        auto const  num_relations = bulk_data.num_connectivity(cell, face_rank);
        ALBANY_ASSERT(num_relations > 0);
        for (auto i = 0; i < num_relations; ++i) {
          auto face = relations[i];
          if (topo_rcp->is_erodible_face(face) == true) {
            auto const elem_gid      = topo_rcp->get_gid(cell) - 1;
            auto const wslid         = elem_gid_to_wslid[elem_gid];
            auto const ws            = wslid.ws;
            auto const elem_lid      = wslid.LID;
            auto const elem_eb_index = stk_mesh_specs_rcp->ebNameToIndex[ws_eb_names[ws]];
            auto const side_local_id = stk_disc.determine_local_side_id(cell, face);
            auto const side_gid      = topo_rcp->get_gid(face) - 1;
            entry.side_GID           = side_gid;
            entry.elem_GID           = elem_gid;
            entry.elem_LID           = elem_lid;
            entry.elem_ebIndex       = elem_eb_index;
            entry.side_local_id      = side_local_id;
            side_set.emplace_back(entry);
          }
        }
      }
    }
  }
#endif

// #define DEBUG
#if defined(DEBUG)
  if (is_erodible == true) {
    auto const num_ss = side_set.size();
    ALBANY_DUMP("===============================================\n");
    ALBANY_DUMP("**** Side set name     : " << ss_id << '\n');
    ALBANY_DUMP("**** Number of entries : " << num_ss << '\n');
    for (auto i = 0; i < num_ss; ++i) {
      auto& ss = side_set[i];
      ALBANY_DUMP("-----------------------------------------------\n");
      ALBANY_DUMP("* entry         : " << i << '\n');
      ALBANY_DUMP("* side_GID      : " << ss.side_GID << '\n');
      ALBANY_DUMP("* elem_GID      : " << ss.elem_GID << '\n');
      ALBANY_DUMP("* elem_LID      : " << ss.elem_LID << '\n');
      ALBANY_DUMP("* elem_ebIndex  : " << ss.elem_ebIndex << '\n');
      ALBANY_DUMP("* side_local_id : " << ss.side_local_id << '\n');
    }
    ALBANY_DUMP("===============================================\n");
    exit(0);
  }
#endif

  using DynRankViewRealT       = Kokkos::DynRankView<RealType, PHX::Device>;
  using DynRankViewMeshScalarT = Kokkos::DynRankView<MeshScalarT, PHX::Device>;
  using DynRankViewScalarT     = Kokkos::DynRankView<ScalarT, PHX::Device>;

  DynRankViewRealT cubPointsSide;
  DynRankViewRealT refPointsSide;
  DynRankViewRealT cubWeightsSide;
  DynRankViewRealT basis_refPointsSide;

  DynRankViewMeshScalarT physPointsSide;
  DynRankViewMeshScalarT jacobianSide;
  DynRankViewMeshScalarT jacobianSide_det;
  DynRankViewMeshScalarT weighted_measure;
  DynRankViewMeshScalarT trans_basis_refPointsSide;
  DynRankViewMeshScalarT weighted_trans_basis_refPointsSide;
  DynRankViewMeshScalarT physPointsCell;

  DynRankViewScalarT dofSide;
  DynRankViewScalarT dofSideVec;
  DynRankViewScalarT dofCell;
  DynRankViewScalarT dofCellVec;
  DynRankViewScalarT data;

  //! For each element block, and for each local side id (e.g. side_id=0,1,2,3,4
  //! for a Prism) we want to identify all the physical cells associated to that
  //! side id and block. In this way we can group them and call Intrepid2
  //! function for a group of cells, which is more effective. At this point we
  //! do not know the number of blocks in this workset (If we assumed to have
  //! elements of the same block in a workset we could skip some of this). Also
  //! we do not know before the evaluator how many cells are associated to a
  //! local side id.

  std::map<int, int>                                              ordinalEbIndex;
  std::vector<int>                                                ebIndexVec;
  std::vector<std::vector<int>>                                   numCellsOnSidesOnBlocks;
  std::vector<std::vector<Kokkos::DynRankView<int, PHX::Device>>> cellsOnSidesOnBlocks;
  for (auto const& it_side : side_set) {
    int const ebIndex   = it_side.elem_ebIndex;
    int const elem_side = it_side.side_local_id;

    if (ordinalEbIndex.insert(std::pair<int, int>(ebIndex, ordinalEbIndex.size())).second) {
      numCellsOnSidesOnBlocks.push_back(std::vector<int>(numSidesOnElem, 0));
      ebIndexVec.push_back(ebIndex);
    }

    numCellsOnSidesOnBlocks[ordinalEbIndex[ebIndex]][elem_side]++;
  }
  cellsOnSidesOnBlocks.resize(ordinalEbIndex.size());
  for (int ib = 0; ib < ordinalEbIndex.size(); ib++) {
    cellsOnSidesOnBlocks[ib].resize(numSidesOnElem);
    for (int is = 0; is < numSidesOnElem; is++) {
      cellsOnSidesOnBlocks[ib][is]    = Kokkos::DynRankView<int, PHX::Device>("cellOnSide_i", numCellsOnSidesOnBlocks[ib][is]);
      numCellsOnSidesOnBlocks[ib][is] = 0;
    }
  }

  for (auto const& it_side : side_set) {
    int const iBlock    = ordinalEbIndex[it_side.elem_ebIndex];
    int const elem_LID  = it_side.elem_LID;
    int const elem_side = it_side.side_local_id;

    cellsOnSidesOnBlocks[iBlock][elem_side](numCellsOnSidesOnBlocks[iBlock][elem_side]++) = elem_LID;
  }

  numBlocks = ordinalEbIndex.size();
  // Loop over the sides that form the boundary condition
  for (int iblock = 0; iblock < numBlocks; ++iblock) {
    for (int side = 0; side < numSidesOnElem; ++side) {
      int numCells_ = numCellsOnSidesOnBlocks[iblock][side];
      if (numCells_ == 0) continue;

      // Get the data that corresponds to the side

      int sideDims   = sideType[side]->getDimension();
      int numQPsSide = cubatureSide[side]->getNumPoints();

      Kokkos::DynRankView<int, PHX::Device> cellVec = cellsOnSidesOnBlocks[iblock][side];

      // need to resize containers because they depend on side topology
      cubPointsSide       = DynRankViewRealT(cubPointsSide_buffer.data(), numQPsSide, sideDims);
      refPointsSide       = DynRankViewRealT(refPointsSide_buffer.data(), numQPsSide, cellDims);
      cubWeightsSide      = DynRankViewRealT(cubWeightsSide_buffer.data(), numQPsSide);
      basis_refPointsSide = DynRankViewRealT(basis_refPointsSide_buffer.data(), numNodes, numQPsSide);

      physPointsSide = Kokkos::createViewWithType<DynRankViewMeshScalarT>(physPointsSide_buffer, physPointsSide_buffer.data(), numCells_, numQPsSide, cellDims);
      jacobianSide =
          Kokkos::createViewWithType<DynRankViewMeshScalarT>(jacobianSide_buffer, jacobianSide_buffer.data(), numCells_, numQPsSide, cellDims, cellDims);
      jacobianSide_det = Kokkos::createViewWithType<DynRankViewMeshScalarT>(jacobianSide_det_buffer, jacobianSide_det_buffer.data(), numCells_, numQPsSide);
      weighted_measure = Kokkos::createViewWithType<DynRankViewMeshScalarT>(weighted_measure_buffer, weighted_measure_buffer.data(), numCells_, numQPsSide);
      trans_basis_refPointsSide = Kokkos::createViewWithType<DynRankViewMeshScalarT>(
          trans_basis_refPointsSide_buffer, trans_basis_refPointsSide_buffer.data(), numCells_, numNodes, numQPsSide);
      weighted_trans_basis_refPointsSide = Kokkos::createViewWithType<DynRankViewMeshScalarT>(
          weighted_trans_basis_refPointsSide_buffer, weighted_trans_basis_refPointsSide_buffer.data(), numCells_, numNodes, numQPsSide);
      physPointsCell = Kokkos::createViewWithType<DynRankViewMeshScalarT>(physPointsCell_buffer, physPointsCell_buffer.data(), numCells_, numNodes, cellDims);

      cubatureSide[side]->getCubature(cubPointsSide, cubWeightsSide);

      // Copy the coordinate data over to a temp container
      for (std::size_t iCell = 0; iCell < numCells_; ++iCell) {
        for (std::size_t node = 0; node < numNodes; ++node) {
          for (std::size_t dim = 0; dim < cellDims; ++dim) {
            physPointsCell(iCell, node, dim) = coordVec(cellVec(iCell), node, dim);
          }
        }
      }

      // Map side cubature points to the reference parent cell based on the
      // appropriate side (elem_side)
      ICT::mapToReferenceSubcell(refPointsSide, cubPointsSide, sideDims, side, *cellType);

      // Calculate side geometry
      ICT::setJacobian(jacobianSide, refPointsSide, physPointsCell, *cellType);

      ICT::setJacobianDet(jacobianSide_det, jacobianSide);

      if (sideDims < 2) {  // for 1 and 2D, get weighted edge measure
        IFST::computeEdgeMeasure(weighted_measure, jacobianSide, cubWeightsSide, side, *cellType, temporary_buffer);
      } else {  // for 3D, get weighted face measure
        IFST::computeFaceMeasure(weighted_measure, jacobianSide, cubWeightsSide, side, *cellType, temporary_buffer);
      }

      // Values of the basis functions at side cubature points, in the reference
      // parent cell domain
      intrepidBasis->getValues(basis_refPointsSide, refPointsSide, Intrepid2::OPERATOR_VALUE);

      // Transform values of the basis functions
      IFST::HGRADtransformVALUE(trans_basis_refPointsSide, basis_refPointsSide);

      // Multiply with weighted measure
      IFST::multiplyMeasure(weighted_trans_basis_refPointsSide, weighted_measure, trans_basis_refPointsSide);

      // Map cell (reference) cubature points to the appropriate side
      // (elem_side) in physical space
      ICT::mapToPhysicalFrame(physPointsSide, refPointsSide, physPointsCell, intrepidBasis);

      // Map cell (reference) degree of freedom points to the appropriate side
      // (elem_side)
      if (bc_type == ROBIN || bc_type == STEFAN_BOLTZMANN) {
        dofCell = Kokkos::createViewWithType<DynRankViewScalarT>(dofCell_buffer, dofCell_buffer.data(), numCells_, numNodes, numDOFsSet);
        dofSide = Kokkos::createViewWithType<DynRankViewScalarT>(dofSide_buffer, dofSide_buffer.data(), numCells_, numQPsSide, numDOFsSet);

        Kokkos::deep_copy(dofCell, 0.0);
        for (std::size_t iCell = 0; iCell < numCells_; ++iCell) {
          for (std::size_t node = 0; node < numNodes; ++node) {
            for (std::size_t icomp = 0; icomp < numDOFsSet; ++icomp) {
              if (vectorDOF) {
                dofCell(iCell, node, icomp) = dof(cellVec(iCell), node, this->offset[icomp]);
              } else {
                dofCell(iCell, node, icomp) = dof(cellVec(iCell), node);
              }
            }
          }
        }

        // This is needed, since evaluate currently sums into
        Kokkos::deep_copy(dofSide, 0.0);

        for (std::size_t icomp = 0; icomp < numDOFsSet; ++icomp) {
          IFST::evaluate(
              Kokkos::subview(dofSide, Kokkos::ALL(), Kokkos::ALL(), icomp),
              Kokkos::subview(dofCell, Kokkos::ALL(), Kokkos::ALL(), icomp),
              trans_basis_refPointsSide);
        }
      }

      // Transform the given BC data to the physical space QPs in each side
      // (elem_side)
      data = Kokkos::createViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device>>(data_buffer, data_buffer.data(), numCells_, numQPsSide, numDOFsSet);

      // Note: if you add a BC here, you need to add it above as well
      // to allocate neumann correctly.
      switch (bc_type) {
        case INTJUMP: {
          ScalarT const elem_scale = matScaling[ebIndexVec[iblock]];
          calc_dudn_const(data, elem_scale);
          break;
        }

        case ROBIN: calc_dudn_robin(data, dofSide); break;

        case STEFAN_BOLTZMANN: calc_dudn_radiate(data, dofSide); break;

        case NORMAL: calc_dudn_const(data); break;

        case PRESS: calc_press(data, jacobianSide, *cellType, side); break;

        case ACEPRESS: calc_ace_press(data, physPointsSide, jacobianSide, *cellType, side, worksetNum, workset.current_time); break;

        case ACEPRESS_HYDROSTATIC: calc_ace_press_hydrostatic(data, physPointsSide, jacobianSide, *cellType, side, worksetNum, workset.current_time); break;

        case TRACTION: calc_traction_components(data); break;
        case CLOSED_FORM: calc_closed_form(data, physPointsSide, jacobianSide, *cellType, side, workset); break;
        default: calc_gradu_dotn_const(data, jacobianSide, *cellType, side); break;
      }

      // Put this side's contribution into the vector
      for (std::size_t iCell = 0; iCell < numCells_; ++iCell) {
        int cell = cellVec(iCell);
        for (std::size_t node = 0; node < numNodes; ++node) {
          for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
            for (std::size_t dim = 0; dim < numDOFsSet; ++dim) {
              neumann(cell, node, dim) += data(iCell, qp, dim) * weighted_trans_basis_refPointsSide(iCell, node, qp);
            }
          }
        }
      }
    }
  }
}

template <typename EvalT, typename Traits>
typename NeumannBase<EvalT, Traits>::ScalarT&
NeumannBase<EvalT, Traits>::getValue(std::string const& n)
{
  if (std::string::npos != n.find("robin")) {
    for (int i = 0; i < 2; i++) {
      std::stringstream ss;
      ss << name << "[" << i << "]";
      if (n == ss.str()) return robin_vals[i];
    }
  } else if (std::string::npos != n.find("radiate")) {
    for (int i = 0; i < 2; i++) {
      std::stringstream ss;
      ss << name << "[" << i << "]";
      if (n == ss.str()) return robin_vals[i];
    }
  } else {
    for (int i = 0; i < dudx.size(); i++) {
      std::stringstream ss;
      ss << name << "[" << i << "]";
      if (n == ss.str()) return dudx[i];
    }
  }

  //  if (n == name) return const_val;
  return const_val;
}

template <typename EvalT, typename Traits>
void
NeumannBase<EvalT, Traits>::calc_traction_components(Kokkos::DynRankView<ScalarT, PHX::Device>& qp_data_returned) const
{
  int numPoints = qp_data_returned.extent(1);  // How many QPs per cell?
  int numCells_ = qp_data_returned.extent(0);  // How many cell's worth of data is being computed?

  for (int cell = 0; cell < numCells_; cell++)
    for (int pt = 0; pt < numPoints; pt++)
      for (int dim = 0; dim < numDOFsSet; dim++) qp_data_returned(cell, pt, dim) = -dudx[dim];
}

template <typename EvalT, typename Traits>
void
NeumannBase<EvalT, Traits>::calc_gradu_dotn_const(
    Kokkos::DynRankView<ScalarT, PHX::Device>&           qp_data_returned,
    const Kokkos::DynRankView<MeshScalarT, PHX::Device>& jacobian_side_refcell,
    const shards::CellTopology&                          celltopo,
    int                                                  local_side_id) const
{
  int numPoints = qp_data_returned.extent(1);  // How many QPs per cell?
  int numCells_ = qp_data_returned.extent(0);  // How many cell's worth of data is being computed?

  Kokkos::DynRankView<ScalarT, PHX::Device> grad_T = Kokkos::createDynRankView(qp_data_returned, "grad_T", numCells_, numPoints, cellDims);
  using DynRankViewMeshScalarT                     = Kokkos::DynRankView<MeshScalarT, PHX::Device>;
  DynRankViewMeshScalarT side_normals =
      Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(side_normals_buffer, side_normals_buffer.data(), numCells_, numPoints, cellDims);
  DynRankViewMeshScalarT normal_lengths =
      Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(normal_lengths_buffer, normal_lengths_buffer.data(), numCells_, numPoints);

  /*
    double kdTdx[3];
    kdTdx[0] = 1.0; // Neumann component in the x direction
    kdTdx[1] = 0.0; // Neumann component in the y direction
    kdTdx[2] = 0.0; // Neumann component in the z direction
  */

  for (int side = 0; side < numCells_; side++) {
    for (int pt = 0; pt < numPoints; pt++) {
      for (int dim = 0; dim < cellDims; dim++) {
        grad_T(side, pt, dim) = dudx[dim];  // k grad T in the x direction goes
                                            // in the x spot, and so on
      }
    }
  }

  // for this side in the reference cell, get the components of the normal
  // direction vector
  ICT::getPhysicalSideNormals(side_normals, jacobian_side_refcell, local_side_id, celltopo);

  // scale normals (unity)
  IRST::vectorNorm(normal_lengths, side_normals, Intrepid2::NORM_TWO);
  IFST::scalarMultiplyDataData<MeshScalarT>(side_normals, normal_lengths, side_normals, true);

  // take grad_T dotted with the unit normal
  IFST::dotMultiplyDataData(qp_data_returned, grad_T, side_normals);
  // for(int cell = 0; cell < numCells; cell++)
  //   for(int pt = 0; pt < numPoints; pt++)
  //     for(int dim = 0; dim < numDOFsSet; dim++)
  //       qp_data_returned(cell, pt, dim) = grad_T(cell, pt, dim) *
  //       side_normals(cell, pt, dim);
}

template <typename EvalT, typename Traits>
void
NeumannBase<EvalT, Traits>::calc_dudn_const(Kokkos::DynRankView<ScalarT, PHX::Device>& qp_data_returned, ScalarT scale) const
{
  int numCells_ = qp_data_returned.extent(0);  // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.extent(1);  // How many QPs per cell?

  // std::cout << "DEBUG: applying const dudn to sideset " << this->sideSetID <<
  // ": " << (const_val * scale) << std::endl;

  for (int side = 0; side < numCells_; side++) {
    for (int pt = 0; pt < numPoints; pt++) {
      for (int dim = 0; dim < numDOFsSet; dim++) {
        qp_data_returned(side, pt, dim) = -const_val * scale;  // User directly specified dTdn, just use it
      }
    }
  }
}

template <typename EvalT, typename Traits>
void
NeumannBase<EvalT, Traits>::calc_dudn_robin(
    Kokkos::DynRankView<ScalarT, PHX::Device>&       qp_data_returned,
    const Kokkos::DynRankView<ScalarT, PHX::Device>& dof_side) const
{
  int numCells_ = qp_data_returned.extent(0);  // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.extent(1);  // How many QPs per cell?

  ScalarT const& dof_value = robin_vals[0];
  ScalarT const& coeff     = robin_vals[1];

  for (int side = 0; side < numCells_; side++) {
    for (int pt = 0; pt < numPoints; pt++) {
      for (int dim = 0; dim < numDOFsSet; dim++) {
        qp_data_returned(side, pt, dim) = coeff * (dof_side(side, pt, dim) - dof_value);
      }
    }
  }
}

template <typename EvalT, typename Traits>
void
NeumannBase<EvalT, Traits>::calc_dudn_radiate(
    Kokkos::DynRankView<ScalarT, PHX::Device>&       qp_data_returned,
    const Kokkos::DynRankView<ScalarT, PHX::Device>& dof_side) const
{
  int numPoints = qp_data_returned.extent(1);  // How many QPs per cell?
  int numCells_ = qp_data_returned.extent(0);  // How many cell's worth of data is being computed?

  ScalarT const& dof_value  = robin_vals[0];
  ScalarT const  dof_value4 = dof_value * dof_value * dof_value * dof_value;
  ScalarT const& coeff      = robin_vals[1];

  for (int cell = 0; cell < numCells_; cell++) {
    for (int pt = 0; pt < numPoints; pt++) {
      for (int dim = 0; dim < numDOFsSet; dim++) {
        qp_data_returned(cell, pt, dim) =
            coeff * (dof_side(cell, pt, dim) * dof_side(cell, pt, dim) * dof_side(cell, pt, dim) * dof_side(cell, pt, dim) - dof_value4);
      }
    }
  }
}

template <typename EvalT, typename Traits>
void
NeumannBase<EvalT, Traits>::calc_press(
    Kokkos::DynRankView<ScalarT, PHX::Device>&           qp_data_returned,
    const Kokkos::DynRankView<MeshScalarT, PHX::Device>& jacobian_side_refcell,
    const shards::CellTopology&                          celltopo,
    int                                                  local_side_id) const
{
  int numCells_ = qp_data_returned.extent(0);  // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.extent(1);  // How many QPs per cell?

  using DynRankViewMeshScalarT = Kokkos::DynRankView<MeshScalarT, PHX::Device>;
  DynRankViewMeshScalarT side_normals =
      Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(side_normals_buffer, side_normals_buffer.data(), numCells_, numPoints, cellDims);
  DynRankViewMeshScalarT normal_lengths =
      Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(normal_lengths_buffer, normal_lengths_buffer.data(), numCells_, numPoints);

  // for this side in the reference cell, get the components of the normal
  // direction vector
  ICT::getPhysicalSideNormals(side_normals, jacobian_side_refcell, local_side_id, celltopo);

  // scale normals (unity)
  IRST::vectorNorm(normal_lengths, side_normals, Intrepid2::NORM_TWO);
  IFST::scalarMultiplyDataData(side_normals, normal_lengths, side_normals, true);

  for (int cell = 0; cell < numCells_; cell++)
    for (int pt = 0; pt < numPoints; pt++)
      for (int dim = 0; dim < numDOFsSet; dim++) qp_data_returned(cell, pt, dim) = const_val * side_normals(cell, pt, dim);
}

template <typename EvalT, typename Traits>
void
NeumannBase<EvalT, Traits>::calc_ace_press(
    Kokkos::DynRankView<ScalarT, PHX::Device>&           qp_data_returned,
    const Kokkos::DynRankView<MeshScalarT, PHX::Device>& physPointsSide,
    const Kokkos::DynRankView<MeshScalarT, PHX::Device>& jacobian_side_refcell,
    const shards::CellTopology&                          celltopo,
    int                                                  local_side_id,
    const int                                            workset_num,
    const double                                         current_time) const
{
  int numCells_ = qp_data_returned.extent(0);  // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.extent(1);  // How many QPs per cell?

  using DynRankViewMeshScalarT = Kokkos::DynRankView<MeshScalarT, PHX::Device>;
  DynRankViewMeshScalarT side_normals =
      Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(side_normals_buffer, side_normals_buffer.data(), numCells_, numPoints, cellDims);
  DynRankViewMeshScalarT normal_lengths =
      Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(normal_lengths_buffer, normal_lengths_buffer.data(), numCells_, numPoints);

  // for this side in the reference cell, get the components of the normal
  // direction vector
  ICT::getPhysicalSideNormals(side_normals, jacobian_side_refcell, local_side_id, celltopo);

  // scale normals (unity)
  IRST::vectorNorm(normal_lengths, side_normals, Intrepid2::NORM_TWO);
  IFST::scalarMultiplyDataData(side_normals, normal_lengths, side_normals, true);

  const ScalarT L     = wave_length_val;  // wave length interpolated in time
  const ScalarT k     = wave_number_val;  // wave number interpolated in time
  const ScalarT s     = s_val;            // still water level interpolated in time
  const ScalarT w     = w_val;            // wave height interpolated in time
  const double  tm    = inputValues[0];   // impact duration
  const double  g     = inputValues[1];   // gravitational constant
  const double  rho   = inputValues[2];   // density
  const double  zmin  = inputValues[3];   // min value of z coord in mesh
  const double  delta = inputValues[4];   // critical wave ratio - for
                                          // determining when to switch b/w
                                          // different forms of wave press NBC
  const bool dump_wave_press_nbc_data = inputValues[5];

#ifdef ACE_WAVE_PRESS_EXTREME_DEBUG_OUTPUT
  std::cout << "DEBUG: zmin = " << zmin << "\n";
#endif

  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  if (std::abs(2 * M_PI / k / w) >= delta) {  // case 1: hydrostatic formulation
    // IKT, 9/4/2022: FIXME - remove the following print statement once we
    // are confident new NBC is working properly.
    //  *out << "ACE Wave Pressure NBC: hydrostatic pressure case, as 2*pi/k/w = "
    //  << 2*M_PI / k / w << " < delta = " << delta <<".\n";
    for (int cell = 0; cell < numCells_; cell++) {
      for (int qp = 0; qp < numPoints; qp++) {
        for (int dim = 0; dim < numDOFsSet; dim++) {
          MeshScalarT z = physPointsSide(cell, qp, 2);
#ifdef ACE_WAVE_PRESS_DEBUG_OUTPUT
          MeshScalarT x = physPointsSide(cell, qp, 0);
          MeshScalarT y = physPointsSide(cell, qp, 1);
#endif
          MeshScalarT ztilde = z - zmin;
#ifdef ACE_WAVE_PRESS_EXTREME_DEBUG_OUTPUT
          std::cout << "DEBUG: z, ztilde = " << z << ", " << ztilde << "\n";
#endif
          const ScalarT pval_qp = this->calc_ace_press_at_z_point(rho, g, s, w, k, ztilde);
#ifdef ACE_WAVE_PRESS_DEBUG_OUTPUT
          if (dim == 0) {
            std::cout << "DEBUG: cell, qp, x, y, z, pval_qp = " << cell << ", " << qp << ", " << x << ", " << y << ", " << z << ", " << pval_qp << "\n";
          }
#endif
          qp_data_returned(cell, qp, dim) = pval_qp * side_normals(cell, qp, dim);
        }
      }
    }
  } else {  // case 2: breaking wave formulation
    *out << "WARNING: Resorting to breaking wave formulation of ACE wave pressure NBC, as 2*pi/k/w = " << 2 * M_PI / k / w << " < delta = " << delta << "!\n";
    for (int cell = 0; cell < numCells_; cell++) {
      for (int qp = 0; qp < numPoints; qp++) {
        for (int dim = 0; dim < numDOFsSet; dim++) {
          MeshScalarT z = physPointsSide(cell, qp, 2);
#ifdef ACE_WAVE_PRESS_DEBUG_OUTPUT
          MeshScalarT x = physPointsSide(cell, qp, 0);
          MeshScalarT y = physPointsSide(cell, qp, 1);
#endif
          MeshScalarT ztilde = z - zmin;
#ifdef ACE_WAVE_PRESS_EXTREME_DEBUG_OUTPUT
          std::cout << "DEBUG: z, ztilde = " << z << ", " << ztilde << "\n";
#endif
          const ScalarT pval_qp = this->calc_ace_press_at_z_point(rho, g, tm, s, w, k, L, ztilde);
#ifdef ACE_WAVE_PRESS_DEBUG_OUTPUT
          if (dim == 0) {
            std::cout << "DEBUG: cell, qp, x, y, z, pval_qp = " << cell << ", " << qp << ", " << x << ", " << y << ", " << z << ", " << pval_qp << "\n";
          }
#endif
          qp_data_returned(cell, qp, dim) = pval_qp * side_normals(cell, qp, dim);
        }
      }
    }
  }

  if (dump_wave_press_nbc_data == true) {
    if (commT->getSize() > 1) {
      ALBANY_ABORT("PHAL_Neumann::calc_ace_press: dumping of ACE pressure BC data not implemented for parallel runs!\n");
    }
    if (numBlocks > 1) {
      ALBANY_ABORT(
          "PHAL_Neumann::calc_ace_press: dumping of ACE pressure BC data not implemented for >1 element blocks!\n"
          << "Please contact Irina Tezaur if this capability is of interest.\n");
    }
    if ((current_time != previous_times[workset_num]) || (is_initial_time[workset_num] == true)) {
#ifdef ACE_WAVE_PRESS_DEBUG_OUTPUT
      std::cout << "DEBUG: workset_num, ace_press_index, time = " << workset_num << ", " << ace_press_index[workset_num] << ", " << current_time << "\n";
#endif
      std::ofstream outfile;
      char          str[80];
      strcpy(str, "ace_press_nbc_");
      strcat(str, std::to_string(ace_press_index[workset_num]).c_str());
      strcat(str, "-");
      strcat(str, std::to_string(workset_num).c_str());
      strcat(str, ".txt");
      outfile.open(str);
      for (int cell = 0; cell < numCells_; cell++) {
        for (int node = 0; node < numNodes; node++) {
          const auto    x         = coordVec(cell, node, 0);
          const auto    y         = coordVec(cell, node, 1);
          const auto    z         = coordVec(cell, node, 2);
          const auto    ztilde    = z - zmin;
          const ScalarT pval_node = (2 * M_PI / k / w >= delta) ? this->calc_ace_press_at_z_point(rho, g, s, w, k, ztilde) :
                                                                  this->calc_ace_press_at_z_point(rho, g, tm, s, w, k, L, ztilde);
#ifdef ACE_WAVE_PRESS_DEBUG_OUTPUT
          std::cout << "DEBUG: workset_num, cell, node, x, y, z, pval_node = " << workset_num << ", " << cell << ", " << node << ", " << x << ", " << y << ", "
                    << z << ", " << pval_node << "\n";
#endif
          outfile << cell << "  " << node << "  " << x << "  " << y << "  " << z << "  " << pval_node << "\n";
        }
      }
      outfile.close();
      ace_press_index[workset_num]++;
    }
  }
  previous_times[workset_num]  = current_time;
  is_initial_time[workset_num] = false;
}

template <typename EvalT, typename Traits>
void
NeumannBase<EvalT, Traits>::calc_ace_press_hydrostatic(
    Kokkos::DynRankView<ScalarT, PHX::Device>&           qp_data_returned,
    const Kokkos::DynRankView<MeshScalarT, PHX::Device>& physPointsSide,
    const Kokkos::DynRankView<MeshScalarT, PHX::Device>& jacobian_side_refcell,
    const shards::CellTopology&                          celltopo,
    int                                                  local_side_id,
    const int                                            workset_num,
    const double                                         current_time) const
{
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  //*out << "IKT in calc_ace_press_hydrostatic!\n";
  int numCells_ = qp_data_returned.extent(0);  // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.extent(1);  // How many QPs per cell?

  using DynRankViewMeshScalarT = Kokkos::DynRankView<MeshScalarT, PHX::Device>;
  DynRankViewMeshScalarT side_normals =
      Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(side_normals_buffer, side_normals_buffer.data(), numCells_, numPoints, cellDims);
  DynRankViewMeshScalarT normal_lengths =
      Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(normal_lengths_buffer, normal_lengths_buffer.data(), numCells_, numPoints);

  // for this side in the reference cell, get the components of the normal
  // direction vector
  ICT::getPhysicalSideNormals(side_normals, jacobian_side_refcell, local_side_id, celltopo);

  // scale normals (unity)
  IRST::vectorNorm(normal_lengths, side_normals, Intrepid2::NORM_TWO);
  IFST::scalarMultiplyDataData(side_normals, normal_lengths, side_normals, true);

  const ScalarT waterH = waterH_val;      // waterH interpolated in time
  const double  g      = inputValues[1];   // gravitational constant
  const double  rho    = inputValues[2];   // density
  const bool dump_wave_press_nbc_data = inputValues[5];

  for (int cell = 0; cell < numCells_; cell++) {
    for (int qp = 0; qp < numPoints; qp++) {
      for (int dim = 0; dim < numDOFsSet; dim++) {
        const ScalarT pval_qp = rho*g*waterH; 
        qp_data_returned(cell, qp, dim) = pval_qp * side_normals(cell, qp, dim);
      }
    }
  }
}

// The following is for the breaking wave formulation of the ACE wave pressure NBC
template <typename EvalT, typename Traits>
typename NeumannBase<EvalT, Traits>::ScalarT
NeumannBase<EvalT, Traits>::calc_ace_press_at_z_point(
    const double  rho,
    const double  g,
    const double  tm,
    const ScalarT s,
    const ScalarT w,
    const ScalarT k,
    const ScalarT L,
    const ScalarT zval) const
{
  ScalarT       pval = 0.0;
  const ScalarT hs   = s;                            // water height
  const ScalarT hc   = w;                            // height above water of the max pressure
  const ScalarT Hb   = hc / 0.7;                     // breaking height of wave
  ScalarT       p0, pc, ps, m1, m2, m3, b1, b2, b3;  // temporary variables
  if (hs > 0.0) {
    p0 = M_PI * rho * Hb * Hb / tm / L * sqrt(g * hs);
    pc = rho * Hb / 2.0 / tm * sqrt(g * hs);
    ps = M_PI * rho * Hb * Hb / (tm * L * cosh(k * hs)) * sqrt(g * hs);
    m1 = (p0 - ps) / hs;
  } else {  // no pressure applied if wave height <=0
    pval = 0.0;
    return pval;
  }
  m2 = (pc - p0) / hc;
  m3 = -2.0 * pc / Hb;
  b1 = ps;
  b2 = m1 * hs + b1 - m2 * hs;
  b3 = m3 * (hs + hc + 0.5 * Hb);

#ifdef ACE_WAVE_PRESS_EXTREME_DEBUG_OUTPUT
  std::cout << "DEBUG: p0, pc, ps = " << p0 << ", " << pc << ", " << ps << "\n";
  std::cout << "DEBUG: m1, m2, m3 = " << m1 << ", " << m2 << ", " << m3 << "\n";
#endif
  if (w + s <= 0.0) {
    pval = 0.0;
    return pval;
  }
  if ((zval >= 0) && (zval <= hs)) {
#ifdef ACE_WAVE_PRESS_EXTREME_DEBUG_OUTPUT
    std::cout << "DEBUG: case 1!\n";
#endif
    pval = m1 * zval + b1;
  } else if ((zval > hs) && (zval <= hs + hc)) {
#ifdef ACE_WAVE_PRESS_EXTREME_DEBUG_OUTPUT
    std::cout << "DEBUG: case 2!\n";
#endif
    pval = m2 * zval + b2;
  } else if ((zval > hs + hc) && (zval <= hs + hc + 0.5 * Hb)) {
#ifdef ACE_WAVE_PRESS_EXTREME_DEBUG_OUTPUT
    std::cout << "DEBUG: case 3!\n";
#endif
    pval = m3 * zval - b3;
  } else {
#ifdef ACE_WAVE_PRESS_EXTREME_DEBUG_OUTPUT
    std::cout << "DEBUG: case 4!\n";
#endif
    pval = 0.0;
  }
  return pval;
}

// The following is for the hydrostatic pressure formulation of the ACE wave pressure NBC
template <typename EvalT, typename Traits>
typename NeumannBase<EvalT, Traits>::ScalarT
NeumannBase<EvalT, Traits>::calc_ace_press_at_z_point(
    const double  rho,
    const double  g,
    const ScalarT s,  // s = still water level
    const ScalarT w,  // w = wave height
    const ScalarT k,  // k = wave number
    const ScalarT zval) const
{
  ScalarT pval = 0.0;
  if (w + s <= 0.0) {
    pval = 0.0;
    return pval;
  }
  if (zval <= w + s) {
    pval = rho * g * (s - zval) + rho * g * w * cosh(k * zval) / cosh(k * s);
  } else {
    pval = 0.0;
  }
  return pval;
}

template <typename EvalT, typename Traits>
void
NeumannBase<EvalT, Traits>::calc_closed_form(
    Kokkos::DynRankView<ScalarT, PHX::Device>&           qp_data_returned,
    const Kokkos::DynRankView<MeshScalarT, PHX::Device>& physPointsSide,
    const Kokkos::DynRankView<MeshScalarT, PHX::Device>& jacobian_side_refcell,
    const shards::CellTopology&                          celltopo,
    int                                                  local_side_id,
    typename Traits::EvalData                            workset) const
{
  // How many QPs per cell?
  int numCells_ = qp_data_returned.extent(0);  // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.extent(1);

  using DynRankViewMeshScalarT = Kokkos::DynRankView<MeshScalarT, PHX::Device>;
  DynRankViewMeshScalarT side_normals =
      Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(side_normals_buffer, side_normals_buffer.data(), numCells_, numPoints, cellDims);
  DynRankViewMeshScalarT normal_lengths =
      Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(normal_lengths_buffer, normal_lengths_buffer.data(), numCells_, numPoints);

  // for this side in the reference cell, get the components of the normal
  // direction vector
  ICT::getPhysicalSideNormals(side_normals, jacobian_side_refcell, local_side_id, celltopo);
  // scale normals (unity)
  IRST::vectorNorm(normal_lengths, side_normals, Intrepid2::NORM_TWO);
  IFST::scalarMultiplyDataData(side_normals, normal_lengths, side_normals, true);

  for (int cell = 0; cell < numCells_; cell++) {
    for (int pt = 0; pt < numPoints; pt++) {
      MeshScalarT x = physPointsSide(cell, pt, 0);
      MeshScalarT y = physPointsSide(cell, pt, 1);
      MeshScalarT z = physPointsSide(cell, pt, 2);
      double      t = workset.current_time;
      for (int dim = 0; dim < numDOFsSet; dim++) {
        // Your closed form equation here!
        ScalarT value                   = 0.0 * t + 0.0 * x + 0.0 * y + 0.0 * z;
        qp_data_returned(cell, pt, dim) = value;
      }
    }
  }
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template <typename Traits>
Neumann<PHAL::AlbanyTraits::Residual, Traits>::Neumann(Teuchos::ParameterList& p) : NeumannBase<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

template <typename Traits>
void
Neumann<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  auto       rcp_disc    = workset.disc;
  auto       stk_disc    = dynamic_cast<Albany::STKDiscretization*>(rcp_disc.get());
  auto       nodeID      = workset.wsElNodeEqID;
  auto       node_gids   = workset.wsElNodeID;
  auto       f           = workset.f;
  auto       f_view      = Albany::getNonconstLocalData(f);
  auto const has_nbi     = stk_disc->hasNodeBoundaryIndicator();
  auto const ss_id       = this->sideSetID;
  auto const is_erodible = ss_id.find("erodible") != std::string::npos;

#if defined(DEBUG)
  {
    if (is_erodible == true) {
      ALBANY_ASSERT(has_nbi == true);
      auto const& bi_field = stk_disc->getNodeBoundaryIndicator();
      ALBANY_DUMP("**** GLOBAL BOUNDARY INDICATOR MAP :\n");
      for (auto&& kv : bi_field) {
        ALBANY_DUMP("NODE GID : " << kv.first << ", BI : " << *kv.second << "\n");
      }
      std::cout << "*** SIDESET BOUNDARY INDICATOR : " << ss_id << " ***\n";
      for (auto cell = 0; cell < workset.numCells; ++cell) {
        for (auto ss_node = 0; ss_node < this->numNodes; ++ss_node) {
          ALBANY_ASSERT(has_nbi == true);
          auto&       stk_mesh_struct = *(stk_disc->getSTKMeshStruct());
          auto const& bi_field        = stk_disc->getNodeBoundaryIndicator();
          auto const  gid             = node_gids[cell][ss_node] + 1;
          auto const  it              = bi_field.find(gid);
          ALBANY_ASSERT(it != bi_field.end());
          auto const    bi                 = *(it->second);
          auto          overlap_node_vs    = stk_disc->getOverlapNodeVectorSpace();
          auto          ov_node_vs_indexer = Albany::createGlobalLocalIndexer(overlap_node_vs);
          auto const    local_node_id      = ov_node_vs_indexer->getLocalElement(gid - 1);
          auto const&   coordinates        = stk_disc->getCoordinates();
          double* const pc                 = &(coordinates[3 * local_node_id]);
          auto const    x                  = pc[0];
          auto const    y                  = pc[1];
          auto const    z                  = pc[2];
          std::cout << "CELL: " << std::setw(4) << cell << ", NODE GID: " << std::setw(4) << gid;
          std::cout << ", BI : " << std::setw(2) << bi << ", ";
          std::cout << "X : " << std::setw(24) << std::setprecision(16) << x << ", ";
          std::cout << "Y : " << std::setw(24) << std::setprecision(16) << y << ", ";
          std::cout << "Z : " << std::setw(24) << std::setprecision(16) << z << "\n";
        }
      }
    }
  }
#endif  // DEBUG

  // Fill in "neumann" array
  this->evaluateNeumannContribution(workset);

  // Place it at the appropriate offset into F
  for (auto cell = 0; cell < workset.numCells; ++cell) {
    for (auto node = 0; node < this->numNodes; ++node) {
      if (has_nbi == true) {
        auto const& bi_field = stk_disc->getNodeBoundaryIndicator();
        auto const  gid      = node_gids[cell][node] + 1;
        auto const  it       = bi_field.find(gid);
        if (it == bi_field.end()) continue;
        auto const nbi = *(it->second);
        if (is_erodible == true && nbi != 2.0) continue;
        if (nbi == 0.0) continue;
      }
      for (auto dim = 0; dim < this->numDOFsSet; ++dim) {
        auto const dof = nodeID(cell, node, this->offset[dim]);
        f_view[dof] += this->neumann(cell, node, dim);
      }
    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template <typename Traits>
Neumann<PHAL::AlbanyTraits::Jacobian, Traits>::Neumann(Teuchos::ParameterList& p) : NeumannBase<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}

// **********************************************************************
template <typename Traits>
void
Neumann<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  auto       rcp_disc    = workset.disc;
  auto       stk_disc    = dynamic_cast<Albany::STKDiscretization*>(rcp_disc.get());
  auto       nodeID      = workset.wsElNodeEqID;
  auto       node_gids   = workset.wsElNodeID;
  auto       f           = workset.f;
  auto       jac         = workset.Jac;
  auto const has_nbi     = stk_disc->hasNodeBoundaryIndicator();
  auto const ss_id       = this->sideSetID;
  auto const is_erodible = ss_id.find("erodible") != std::string::npos;
  auto const fill        = f != Teuchos::null;
  auto       f_view      = fill ? Albany::getNonconstLocalData(f) : Teuchos::null;

  // Fill in "neumann" array
  this->evaluateNeumannContribution(workset);
  Teuchos::Array<LO> row(1);
  Teuchos::Array<LO> col(1);
  Teuchos::Array<ST> value(1);

  for (auto cell = 0; cell < workset.numCells; ++cell) {
    for (auto node = 0; node < this->numNodes; ++node) {
      if (has_nbi == true) {
        auto const& bi_field = stk_disc->getNodeBoundaryIndicator();
        auto const  gid      = node_gids[cell][node] + 1;
        auto const  it       = bi_field.find(gid);
        if (it == bi_field.end()) continue;
        auto const nbi = *(it->second);
        if (is_erodible == true && nbi != 2.0) continue;
        if (nbi == 0.0) continue;
      }
      for (auto dim = 0; dim < this->numDOFsSet; ++dim) {
        row[0]         = nodeID(cell, node, this->offset[dim]);
        auto const neq = nodeID.extent(2);
        if (fill == true) {
          f_view[row[0]] += this->neumann(cell, node, dim).val();
        }

        // Check derivative array is nonzero
        if (this->neumann(cell, node, dim).hasFastAccess()) {
          // Loop over nodes in element
          for (auto node_col = 0; node_col < this->numNodes; node_col++) {
            // Loop over equations per node
            for (auto eq_col = 0; eq_col < neq; eq_col++) {
              auto const lcol = neq * node_col + eq_col;

              // Global column
              col[0]   = nodeID(cell, node_col, eq_col);
              value[0] = this->neumann(cell, node, dim).fastAccessDx(lcol);
              if (workset.is_adjoint) {
                // Sum Jacobian transposed
                Albany::addToLocalRowValues(jac, col[0], row(), value());
              } else {
                // Sum Jacobian
                Albany::addToLocalRowValues(jac, row[0], col(), value());
              }
            }  // column equations
          }    // column nodes
        }      // has fast access
      }
    }
  }
}

// **********************************************************************
// Simple evaluator to aggregate all Neumann BCs into one "field"
// **********************************************************************

template <typename EvalT, typename Traits>
NeumannAggregator<EvalT, Traits>::NeumannAggregator(Teuchos::ParameterList const& p)
{
  Teuchos::RCP<PHX::DataLayout> dl = p.get<Teuchos::RCP<PHX::DataLayout>>("Data Layout");

  std::vector<std::string> const& nbcs = *p.get<Teuchos::RCP<std::vector<std::string>>>("NBC Names");

  for (unsigned int i = 0; i < nbcs.size(); i++) {
    PHX::Tag<ScalarT> fieldTag(nbcs[i], dl);
    this->addDependentField(fieldTag);
  }

  PHX::Tag<ScalarT> fieldTag(p.get<std::string>("NBC Aggregator Name"), dl);
  this->addEvaluatedField(fieldTag);

  this->setName("Neumann Aggregator" + PHX::print<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
NeumannAggregator<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm)
{
  d.fill_field_dependencies(this->dependentFields(), this->evaluatedFields());
}

}  // namespace PHAL
