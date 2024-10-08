// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <Albany_STKNodeSharing.hpp>
#include <Shards_BasicTopologies.hpp>
#include <cinttypes>
#include <iostream>
#include <stk_io/IossBridge.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Selector.hpp>

#include "Albany_TmplSTKMeshStruct.hpp"
#include "Albany_Utils.hpp"
#include "Teuchos_VerboseObject.hpp"

#define ST_LLU "%" PRId64

// Test 64 bit Tpetra address space
const GO StartIndex = 0;
// const GO StartIndex = 2147483647L; // 2^31 - 1

template <unsigned Dim, class traits>
Albany::TmplSTKMeshStruct<Dim, traits>::TmplSTKMeshStruct(
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    const Teuchos::RCP<Teuchos::ParameterList>& adaptParams_,
    const Teuchos::RCP<Teuchos_Comm const>&     commT)
    : GenericSTKMeshStruct(params, adaptParams_, traits_type::size),
      periodic_x(params->get("Periodic_x BC", false)),
      periodic_y(params->get("Periodic_y BC", false)),
      periodic_z(params->get("Periodic_z BC", false)),
      triangles(false)
{
  /*
    There are two use cases of interest here.

    1. Do not specify element block information ("Element Blocks"). In this
    case, numEB = 1 and the scale (domain dimensions) and discretization is set
    globally across the mesh.

    2. "Element Blocks" is specified in the input file. In this case, the
    logical size of the element blocks must be specified, as well as the number
    of elements in each block and their size (block length).

  */

  GO total_elems;

  int input_nEB = params->get<int>("Element Blocks",
                                   -1);  // Read the number of element blocks. -1 if not listed

  if (input_nEB <= 0 || Dim == 0)  // If "Element Blocks" are not present in input file

    numEB = 1;

  else

    numEB = input_nEB;

  params->validateParameters(*this->getValidDiscretizationParameters(), 0, Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  nelem[0] = 1;    // One element in the case of a single point 0D mesh (the below
                   // isn't executed)
  scale[0] = 1.0;  // one unit length in the x dimension

  // The number of element blocks. Always at least one.

  EBSpecs.resize(numEB);

  switch (Dim) {
    case 0:  // 0D
      default_element_type      = stk::topology::PARTICLE;
      optional_element_type     = stk::topology::PARTICLE;
      default_element_side_type = stk::topology::PARTICLE;  // No sides in 0D
      break;
    case 1:  // 1D
      default_element_type      = stk::topology::LINE_2_1D;
      optional_element_type     = stk::topology::LINE_2_1D;
      default_element_side_type = stk::topology::PARTICLE;  // No sides in 1D
      break;
    case 2:  // 2D
      default_element_type      = stk::topology::QUAD_4_2D;
      optional_element_type     = stk::topology::TRI_3_2D;
      default_element_side_type = stk::topology::LINE_2;
      break;
    case 3:  // 3D
      default_element_type      = stk::topology::HEX_8;
      optional_element_type     = stk::topology::HEX_8;
      default_element_side_type = stk::topology::QUAD_4;
      break;
  }
  scales.resize(Dim);
  for (unsigned i = 0; i < Dim; i++) {  // Get the number of elements in each dimension from params
                                        // Note that nelem will default to 0 and scale to 1 if element
                                        // blocks are specified

    // Read the values for "1D Elements", "2D Elements", "3D Elements"
    std::stringstream buf;
    buf << i + 1 << "D Elements";
    nelem[i] = params->get<int>(buf.str(), 0);

    // Read the values for "1D Scale", "2D Scale", "3D Scale"
    std::stringstream buf2;
    buf2 << i + 1 << "D Scale";

    scale[i]  = params->get<double>(buf2.str(), 1.0);
    scales[i] = scale[i];
  }

  if (input_nEB <= 0 || Dim == 0) {  // If "Element Blocks" are not present in input file

    EBSpecs[0].Initialize(nelem, scale);

    // Format and print out information about the mesh that is being generated

    if (commT->getRank() == 0 && Dim > 0) {  // Not reached for 0D problems

      std::cout << "TmplSTKMeshStruct:: Creating " << Dim << "D mesh of size ";

      std::stringstream nelem_txt, scale_txt;

      for (unsigned idx = 0; idx < Dim - 1; idx++) {
        nelem_txt << nelem[idx] << "x";
        scale_txt << scale[idx] << "x";
      }

      nelem_txt << nelem[Dim - 1];
      scale_txt << scale[Dim - 1];

      std::cout << nelem_txt.str() << " elements and scaled to " << scale_txt.str() << std::endl;

      if (triangles) std::cout << " Quad elements cut to make twice as many triangles " << std::endl;
    }

    // Calculate total number of elements
    total_elems = nelem[0];

    for (unsigned i = 1; i < Dim; i++) total_elems *= nelem[i];

  } else {  // Element blocks are present in input

    std::vector<GO> min(Dim), max(Dim);

    for (unsigned i = 0; i < Dim; i++) {
      min[i] = std::numeric_limits<GO>::max();
      max[i] = std::numeric_limits<GO>::min();

      nelem[i] = 0;
    }

    // Read the EB extents from the parameter list and initialize the EB structs
    for (unsigned int eb = 0; eb < numEB; eb++) {
      EBSpecs[eb].Initialize(eb, params);

      //      for(int i = 0; i < Dim; i++)

      //        nelem[i] += EBSpecs[eb].numElems(i);

      for (unsigned i = 0; i < Dim; i++) {
        min[i] = (min[i] < EBSpecs[eb].min[i]) ? min[i] : EBSpecs[eb].min[i];
        max[i] = (max[i] > EBSpecs[eb].max[i]) ? max[i] : EBSpecs[eb].max[i];
      }
    }

    for (unsigned i = 0; i < Dim; i++) nelem[i] = max[i] - min[i];

    // Calculate total number of elements
    total_elems = nelem[0];

    for (unsigned i = 1; i < Dim; i++) total_elems *= nelem[i];
  }

  std::vector<std::string> nsNames;

  // Construct the nodeset names

  for (unsigned idx = 0; idx < Dim * 2; idx++) {  // 2 nodesets per dimension (one at beginning, one at end)
    std::stringstream buf;
    buf << "NodeSet" << idx;
    nsNames.push_back(buf.str());
  }
  // For 2D and 3D, and extra node set of a single node for setting Pressure in
  // confined incompressible flows
  if (Dim == 2 || Dim == 3) nsNames.push_back("NodeSet99");

  std::vector<std::string> ssNames;

  // Construct the sideset names

  if (Dim > 1)  // Sidesets present only for 2 and 3D problems

    for (unsigned idx = 0; idx < Dim * 2; idx++) {  // 2 sidesets per dimension (one at beginning, one at end)
      std::stringstream buf;
      buf << "SideSet" << idx;
      ssNames.push_back(buf.str());
    }

  DeclareParts(EBSpecs, ssNames, nsNames);

  // Different element topologies. Note that there is only a choice for
  // triangles/quads in 2D (all other dimensions select the default element type
  // for now)

  std::string cellTopo = params->get("Cell Topology", "Quad");
  if (cellTopo == "Tri" || cellTopo == "Triangle") triangles = true;
  //    "\nUnknown Cell Topology entry in STK2D(not \'Tri\' or \'Quad\'): "
  //     << cellTopo);

  // Set the element types in the EBs

  // get the type of transformation of STK mesh
  xShift = params->get("x-shift", 0.0);
  yShift = params->get("y-shift", 0.0);
  zShift = params->get("z-shift", 0.0);

  // boolean specifying if ascii mesh has contiguous IDs; only used for ascii
  // meshes on 1 processor
  contigIDs = params->get("Contiguous IDs", true);

  // Does user want to write coordinates to matrix market file (e.g., for ML
  // analysis)?
  writeCoordsToMMFile = params->get("Write Coordinates to MatrixMarket", false);

  for (unsigned int i = 0; i < numEB; i++) {
    if (triangles) {
      stk::mesh::set_topology(*partVec[i], optional_element_type);
    } else {
      stk::mesh::set_topology(*partVec[i], default_element_type);
    }
  }

  for (std::map<std::string, stk::mesh::Part*>::const_iterator it = ssPartVec.begin(); it != ssPartVec.end(); ++it)

    stk::mesh::set_topology(*it->second, default_element_side_type);

  int cub            = params->get("Cubature Degree", 3);
  int worksetSizeMax = params->get<int>("Workset Size", DEFAULT_WORKSET_SIZE);

  // Get Cubature Rule
  std::string const    cub_rule_string = params->get("Cubature Rule", "GAUSS");
  Intrepid2::EPolyType cub_rule;
  if (cub_rule_string == "GAUSS")
    cub_rule = static_cast<Intrepid2::EPolyType>(Intrepid2::POLYTYPE_GAUSS);
  else if (cub_rule_string == "GAUSS_RADAU_LEFT")
    cub_rule = static_cast<Intrepid2::EPolyType>(Intrepid2::POLYTYPE_GAUSS_RADAU_LEFT);
  else if (cub_rule_string == "GAUSS_RADAU_RIGHT")
    cub_rule = static_cast<Intrepid2::EPolyType>(Intrepid2::POLYTYPE_GAUSS_RADAU_RIGHT);
  else if (cub_rule_string == "GAUSS_LOBATTO")
    cub_rule = static_cast<Intrepid2::EPolyType>(Intrepid2::POLYTYPE_GAUSS_LOBATTO);
  else
    ALBANY_ABORT(
        "Invalid Cubature Rule: " << cub_rule_string
                                  << "; valid options are GAUSS, GAUSS_RADAU_LEFT, "
                                     "GAUSS_RADAU_RIGHT, and GAUSS_LOBATTO");

  // Create just enough of the mesh to figure out number of owned elements
  // so that the problem setup can know the worksetSize

  // Distribute the elems equally. Build total_elems elements, with nodeIDs
  // starting at StartIndex
  elem_map = Teuchos::rcp(new Tpetra_Map(total_elems, StartIndex, commT, Tpetra::GloballyDistributed));

  int worksetSize = this->computeWorksetSize(worksetSizeMax, elem_map->getLocalNumElements() * (triangles ? 2 : 1));

  // Build a map to get the EB name given the index
  std::map<std::string, int> ebNameToIndex;
  for (unsigned int eb = 0; eb < numEB; eb++) ebNameToIndex[partVec[eb]->name()] = eb;

  // Construct MeshSpecsStruct
  if (!params->get("Separate Evaluators by Element Block", false)) {
    stk::topology           stk_topo_data = metaData->get_topology(*partVec[0]);
    shards::CellTopology    shards_ctd    = stk::mesh::get_cell_topology(stk_topo_data);
    const CellTopologyData& ctd           = *shards_ctd.getCellTopologyData();

    this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(
        ctd, numDim, cub, nsNames, ssNames, worksetSize, partVec[0]->name(), ebNameToIndex, this->interleavedOrdering, false, cub_rule));
  } else {
    meshSpecs.resize(numEB);

    this->allElementBlocksHaveSamePhysics = false;

    for (unsigned int eb = 0; eb < numEB; eb++) {
      // MeshSpecs holds all info needed to set up an Albany problem

      stk::topology           stk_topo_data = metaData->get_topology(*partVec[eb]);
      shards::CellTopology    shards_ctd    = stk::mesh::get_cell_topology(stk_topo_data);
      const CellTopologyData& ctd           = *shards_ctd.getCellTopologyData();

      this->meshSpecs[eb] = Teuchos::rcp(new Albany::MeshSpecsStruct(
          ctd, numDim, cub, nsNames, ssNames, worksetSize, partVec[eb]->name(), ebNameToIndex, this->interleavedOrdering, true, cub_rule));
    }
  }

  // Upon request, add a nodeset for each sideset
  if (params->get<bool>("Build Node Sets From Side Sets", false)) {
    this->addNodeSetsFromSideSets();
  }

  // Create a mesh specs object for EACH side set
  this->initializeSideSetMeshSpecs(commT);

  // Initialize the requested sideset mesh struct in the mesh
  this->initializeSideSetMeshStructs(commT);
}

template <unsigned Dim, class traits>
void
Albany::TmplSTKMeshStruct<Dim, traits>::setFieldAndBulkData(
    Teuchos::RCP<Teuchos_Comm const> const&                                          commT,
    Teuchos::RCP<Teuchos::ParameterList> const&                                      params,
    unsigned int const                                                               neq_,
    AbstractFieldContainer::FieldContainerRequirements const&                        req,
    Teuchos::RCP<Albany::StateInfoStruct> const&                                     sis,
    unsigned int const                                                               worksetSize,
    std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> const&              side_set_sis,
    std::map<std::string, AbstractFieldContainer::FieldContainerRequirements> const& side_set_req)
{
  // Create global mesh: Dim-D structured, rectangular

  //  std::vector<std::vector<double> > h_dim;
  std::vector<double> h_dim[traits_type::size];
  //  h_dim.resize(traits_type::size);
  //  x.resize(traits_type::size);
  //  x.resize(Dim);

  for (unsigned idx = 0; idx < Dim; idx++) {
    // Allocate the storage

    x[idx].resize(nelem[idx] + 1);
    h_dim[idx].resize(nelem[idx] + 1);
  }

  for (unsigned int eb = 0; eb < numEB; eb++) EBSpecs[eb].calcElemSizes(h_dim);

  for (unsigned idx = 0; idx < Dim; idx++) {
    x[idx][0] = 0;

    for (unsigned int i = 1; i <= nelem[idx]; i++) x[idx][i] = x[idx][i - 1] + h_dim[idx][i - 1];  // place the coordinates of the element nodes
  }

  SetupFieldData(commT, neq_, req, sis, worksetSize);

  metaData->commit();

  // STK
  bulkData->modification_begin();  // Begin modifying the mesh

  buildMesh(commT);

  // STK
  Albany::fix_node_sharing(*bulkData);
  bulkData->modification_end();

  this->setDefaultCoordinates3d();
  this->loadRequiredInputFields(req, commT);

  // Refine the mesh before starting the simulation if indicated
  uniformRefineMesh(commT);

  // Rebalance the mesh before starting the simulation if indicated
  rebalanceInitialMeshT(commT);

  // Build additional mesh connectivity needed for mesh fracture (if indicated)
  computeAddlConnectivity();

  fieldAndBulkDataSet = true;

  // Finally, setup the side set meshes (if any)
  this->finalizeSideSetMeshStructs(commT, side_set_req, side_set_sis, worksetSize);
}

template <unsigned Dim, class traits>
void
Albany::TmplSTKMeshStruct<Dim, traits>::DeclareParts(
    std::vector<EBSpecsStruct<Dim, traits>> ebStructArray,
    std::vector<std::string>                ssNames,
    std::vector<std::string>                nsNames)
{
  // Element blocks
  for (std::size_t i = 0; i < ebStructArray.size(); i++) {
    // declare each element block present in the mesh
    partVec[i] = &metaData->declare_part(ebStructArray[i].name, stk::topology::ELEMENT_RANK);
    stk::io::put_io_part_attribute(*partVec[i]);
  }

  // SideSets
  for (std::size_t i = 0; i < ssNames.size(); i++) {
    std::string ssn = ssNames[i];
    ssPartVec[ssn]  = &metaData->declare_part(ssn, metaData->side_rank());
    stk::io::put_io_part_attribute(*ssPartVec[ssn]);
  }

  // NodeSets
  for (std::size_t i = 0; i < nsNames.size(); i++) {
    std::string nsn = nsNames[i];
    nsPartVec[nsn]  = &metaData->declare_part(nsn, stk::topology::NODE_RANK);
    stk::io::put_io_part_attribute(*nsPartVec[nsn]);
  }
}

template <unsigned Dim, class traits>
void
Albany::EBSpecsStruct<Dim, traits>::Initialize(GO nnelems[], double blLen[])
{
  name = "Block0";

  for (unsigned i = 0; i < Dim; i++) {
    min[i]      = 0;
    max[i]      = nnelems[i];
    blLength[i] = blLen[i];
  }
}

// Template specialization functions for the different dimensions

// Specializations to read the element block information for each dimension

template <>
GO
Albany::EBSpecsStruct<0>::numElems(int i)
{
  return 1L;
}

template <>
void
Albany::EBSpecsStruct<0>::calcElemSizes(std::vector<double> h[])
{
  h[0][0] = 1.0;
}

template <>
void
Albany::EBSpecsStruct<0>::Initialize(GO nelems[], double blLen[])
{
  // Never more than one element block in a 0D problem
  name        = "Block0";
  blLength[0] = blLen[0];
}

template <>
void
Albany::EBSpecsStruct<0>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  // Never more than one element block in a 0D problem
  name        = "Block0";
  blLength[0] = 1.0;
}

template <>
void
Albany::EBSpecsStruct<1>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  // Read element block specs from input file. Note that this is called only for
  // the multiple element block case, once per element block.

  // Construct the name of the element block desired
  std::stringstream ss;
  ss << "Block " << i;

  // Get the parameter string for this block. Note the default (below) applies
  // if there is no block information in the xml file.

  std::string blkinfo = params->get<std::string>(ss.str(), "begins at 0 ends at 100 length 1.0 named Block0");

  std::string junk;

  // Parse object to break up the line
  std::stringstream parsess(blkinfo);

  //       "begins"  "at"      0      "ends"   "at"     100   "length"     1.0
  //       "named" "Bck0"
  parsess >> junk >> junk >> min[0] >> junk >> junk >> max[0] >> junk >> blLength[0] >> junk >> name;
}

template <>
void
Albany::EBSpecsStruct<2>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  // Read element block specs from input file, or set defaults
  char buf[256];

  // Construct the name of the element block desired
  std::stringstream ss;
  ss << "Block " << i;

  // Get the parameter string for this block. Note the default (below) applies
  // if there is no block information in the xml file.

  std::string blkinfo = params->get<std::string>(ss.str(), "begins at (0, 0) ends at (100, 100) length (1.0, 1.0) named Block0");

  // Parse it
  sscanf(
      &blkinfo[0],
      "begins at (" ST_LLU "," ST_LLU ") ends at (" ST_LLU "," ST_LLU ") length (%lf,%lf) named %s",
      &min[0],
      &min[1],
      &max[0],
      &max[1],
      &blLength[0],
      &blLength[1],
      buf);

  name = buf;
}

template <>
void
Albany::EBSpecsStruct<3>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  // Read element block specs from input file, or set defaults
  char buf[256];

  // Construct the name of the element block desired
  std::stringstream ss;
  ss << "Block " << i;

  // Get the parameter string for this block. Note the default (below) applies
  // if there is no block information in the xml file.

  std::string blkinfo = params->get<std::string>(
      ss.str(),
      "begins at (0, 0, 0) ends at (100, 100, 100) length (1.0, 1.0, 1.0) "
      "named Block0");

  // Parse it
  sscanf(
      &blkinfo[0],
      "begins at (" ST_LLU "," ST_LLU "," ST_LLU ") ends at (" ST_LLU "," ST_LLU "," ST_LLU ") length (%lf,%lf,%lf) named %s",
      &min[0],
      &min[1],
      &min[2],
      &max[0],
      &max[1],
      &max[2],
      &blLength[0],
      &blLength[1],
      &blLength[2],
      buf);

  name = buf;
}

// Specializations to build the mesh for each dimension
template <>
void
Albany::TmplSTKMeshStruct<0>::buildMesh(const Teuchos::RCP<Teuchos_Comm const>& commT)
{
  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);

  singlePartVec[0] = partVec[0];  // Get the element block part to put the
                                  // element in. Only one block in 0D mesh

  // Declare element 1 is in that block
  stk::mesh::Entity pt = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1, singlePartVec);
  // Declare node 1 is in the node part vector
  stk::mesh::Entity node = bulkData->declare_entity(stk::topology::NODE_RANK, 1, nodePartVec);
  // Declare that the node belongs to the element "pt"
  // "node" is the zeroth node of this element
  bulkData->declare_relation(pt, node, 0);

  // No node sets or side sets in 0D
}

// Specialized for 0 D

template <>
void
Albany::TmplSTKMeshStruct<0, Albany::albany_stk_mesh_traits<0>>::setFieldAndBulkData(
    const Teuchos::RCP<Teuchos_Comm const>&                   commT,
    const Teuchos::RCP<Teuchos::ParameterList>&               params,
    const unsigned int                                        neq_,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    const Teuchos::RCP<Albany::StateInfoStruct>&              sis,
    const unsigned int                                        worksetSize,
    std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> const& /*side_set_sis*/,
    std::map<std::string, AbstractFieldContainer::FieldContainerRequirements> const& /*side_set_req*/)
{
  SetupFieldData(commT, neq_, req, sis, worksetSize);

  metaData->commit();

  // STK
  bulkData->modification_begin();  // Begin modifying the mesh

  TmplSTKMeshStruct<0, albany_stk_mesh_traits<0>>::buildMesh(commT);

  // STK
  Albany::fix_node_sharing(*bulkData);
  bulkData->modification_end();
}

template <>
void
Albany::TmplSTKMeshStruct<1>::buildMesh(const Teuchos::RCP<Teuchos_Comm const>& commT)
{
  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);

  std::vector<GO> elemNumber(1);
  unsigned int    ebNo;

  AbstractSTKFieldContainer::STKFieldType* coordinates_field = fieldContainer->getCoordinatesField();

  if (periodic_x) {
    this->PBCStruct.periodic[0] = true;
    this->PBCStruct.scale[0]    = scale[0];
  }

  // Create elements and node IDs
  for (int i = 0; i < elem_map->getLocalNumElements(); i++) {
    const GO elem_GID   = elem_map->getGlobalElement(i);
    const GO left_node  = elem_GID;
    GO       right_node = left_node + 1;
    if (periodic_x) right_node %= elem_map->getGlobalNumElements();
    //    if (rightNode < right_node) rightNode = right_node;

    stk::mesh::EntityId elem_id = (stk::mesh::EntityId)elem_GID;

    // The left node number is the same as the element number
    elemNumber[0] = elem_GID;

    // find out which EB the element is in

    if (numEB == 1)  // assume all elements are in element block if there is only one
      ebNo = 0;
    else {
      for (ebNo = 0; ebNo < numEB; ebNo++)
        // Does the elemNumber lie in the EB?
        if (EBSpecs[ebNo].inEB(elemNumber)) break;

      if (ebNo == numEB) {  // error, we didn't find an element block that this
                            // element should fit in

        ALBANY_ABORT(std::endl << "Error: Could not place element " << elem_GID << " in its corresponding element block." << std::endl);
      }
    }

    singlePartVec[0] = partVec[ebNo];

    // Build element "1+elem_id" and put it in the element block
    stk::mesh::Entity edge = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1 + elem_id, singlePartVec);
    // Build the left and right nodes of this element and put them in the node
    // part Vec
    stk::mesh::Entity lnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1 + left_node, nodePartVec);
    stk::mesh::Entity rnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1 + right_node, nodePartVec);
    // node number 0 of this element
    bulkData->declare_relation(edge, lnode, 0);
    // node number 1 of this element
    bulkData->declare_relation(edge, rnode, 1);

    // set the coordinate values for these nodes
    double* lnode_coord = stk::mesh::field_data(*coordinates_field, lnode);
    lnode_coord[0]      = x[0][left_node];
    double* rnode_coord = stk::mesh::field_data(*coordinates_field, rnode);
    rnode_coord[0]      = x[0][right_node];

    /*
        if(proc_rank_field){
          int* p_rank = stk::mesh::field_data(*proc_rank_field, edge);
          p_rank[0] = comm->MyPID();
        }
    */

    // Set node sets. There are no side sets currently with 1D problems (only 2D
    // and 3D)
    if (left_node == StartIndex) {
      singlePartVec[0] = nsPartVec["NodeSet0"];
      bulkData->change_entity_parts(lnode, singlePartVec);
    }
    if (right_node == elem_map->getGlobalNumElements() + StartIndex) {
      singlePartVec[0] = nsPartVec["NodeSet1"];
      bulkData->change_entity_parts(rnode, singlePartVec);
    }
    // Periodic case -- right node is wrapped to left side
    if (right_node == StartIndex) {
      singlePartVec[0] = nsPartVec["NodeSet0"];
      bulkData->change_entity_parts(rnode, singlePartVec);
    }

    // Note: Side_ranks are not currently registered for 1D elements, see
    // $TRILINOS_DIR/packages/stk/stk_mesh/stk_mesh/fem/MetaData.cpp line 104.
    // Skip sideset construction in 1D for this reason (GAH)
  }
}

template <>
void
Albany::TmplSTKMeshStruct<2>::buildMesh(const Teuchos::RCP<Teuchos_Comm const>& commT)
{
  // STK
  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  std::vector<GO>       elemNumber(2);
  unsigned int          ebNo;

  // Create elements and node IDs

  const GO nodes_x = periodic_x ? nelem[0] : nelem[0] + 1;
  const GO mod_x   = periodic_x ? nelem[0] : std::numeric_limits<GO>::max();
  const GO mod_y   = periodic_y ? nelem[1] : std::numeric_limits<GO>::max();

  AbstractSTKFieldContainer::STKFieldType* coordinates_field = fieldContainer->getCoordinatesField();

  if (periodic_x) {
    this->PBCStruct.periodic[0] = true;
    this->PBCStruct.scale[0]    = scale[0];
  }
  if (periodic_y) {
    this->PBCStruct.periodic[1] = true;
    this->PBCStruct.scale[1]    = scale[1];
  }

  for (int i = 0; i < elem_map->getLocalNumElements(); i++) {
    const GO elem_GID = elem_map->getGlobalElement(i);
    const GO x_GID    = elem_GID % nelem[0];  // mesh column number
    const GO y_GID    = elem_GID / nelem[0];  // mesh row number

    const GO x_GIDplus1 = (x_GID + 1) % mod_x;  // = x_GID+1 unless last element of periodic system
    const GO y_GIDplus1 = (y_GID + 1) % mod_y;  // = y_GID+1 unless last element of periodic system

    const GO lower_left  = x_GID + nodes_x * y_GID;
    const GO lower_right = x_GIDplus1 + nodes_x * y_GID;
    const GO upper_right = x_GIDplus1 + nodes_x * y_GIDplus1;
    const GO upper_left  = x_GID + nodes_x * y_GIDplus1;

    // get ID of quadrilateral -- will be doubled for trianlges below
    stk::mesh::EntityId elem_id = (stk::mesh::EntityId)elem_GID;

    // Calculate elemNumber of element
    elemNumber[0] = x_GID;
    elemNumber[1] = y_GID;

    // find out which EB the element is in

    if (numEB == 1)  // assume all elements are in element block if there is only one
      ebNo = 0;
    else {
      for (ebNo = 0; ebNo < numEB; ebNo++) {
        // Does the elemNumber lie in the EB?
        // std::cout << elem_GID << " " << i << " " << ebNo << std::endl;
        if (EBSpecs[ebNo].inEB(elemNumber)) break;
      }

      if (ebNo == numEB) {  // error, we didn't find an element block that this
                            // element should fit in

        ALBANY_ABORT(std::endl << "Error: Could not place element " << elem_GID << " in its corresponding element block." << std::endl);
      }
    }

    singlePartVec[0] = partVec[ebNo];

    // if (x_GID==0 && y_GID==0) cout << " FOUND global node " << lower_left <<
    // endl;

    // Declare Nodes = (Add one to IDs because STK requires 1-based
    stk::mesh::Entity llnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1 + lower_left, nodePartVec);
    stk::mesh::Entity lrnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1 + lower_right, nodePartVec);
    stk::mesh::Entity urnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1 + upper_right, nodePartVec);
    stk::mesh::Entity ulnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1 + upper_left, nodePartVec);

    if (triangles) {  // pair of 3-node triangles

      stk::mesh::Entity elem = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1 + 2 * elem_id, singlePartVec);
      bulkData->declare_relation(elem, llnode, 0);
      bulkData->declare_relation(elem, lrnode, 1);
      bulkData->declare_relation(elem, urnode, 2);
      stk::mesh::Entity elem2 = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1 + 2 * elem_id + 1, singlePartVec);
      bulkData->declare_relation(elem2, llnode, 0);
      bulkData->declare_relation(elem2, urnode, 1);
      bulkData->declare_relation(elem2, ulnode, 2);

      // Triangle sideset construction
      if (x_GID == 0) {  // left edge of mesh, elem2 has side 2 on left boundary

        singlePartVec[0]            = ssPartVec["SideSet0"];
        stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID));

        stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
        bulkData->declare_relation(elem2, side, 2 /*local side id*/);

        bulkData->declare_relation(side, ulnode, 0);
        bulkData->declare_relation(side, llnode, 1);
      }
      if (x_GIDplus1 == nelem[0]) {  // right edge of mesh, elem has side 1 on right boundary

        singlePartVec[0]            = ssPartVec["SideSet1"];
        stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID) + 1);

        stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
        bulkData->declare_relation(elem, side, 1 /*local side id*/);

        bulkData->declare_relation(side, lrnode, 0);
        bulkData->declare_relation(side, urnode, 1);
      }
      if (y_GID == 0) {  // bottom edge of mesh, elem has side 0 on lower boundary

        singlePartVec[0]            = ssPartVec["SideSet2"];
        stk::mesh::EntityId side_id = (stk::mesh::EntityId)(x_GID);

        stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
        bulkData->declare_relation(elem, side, 0 /*local side id*/);

        bulkData->declare_relation(side, llnode, 0);
        bulkData->declare_relation(side, lrnode, 1);
      }
      if (y_GIDplus1 == nelem[1]) {  // top edge of mesh, elem2 has side 1 on upper boundary

        singlePartVec[0]            = ssPartVec["SideSet3"];
        stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * nelem[1]) + x_GID);

        stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
        bulkData->declare_relation(elem2, side, 1 /*local side id*/);

        bulkData->declare_relation(side, urnode, 0);
        bulkData->declare_relation(side, ulnode, 1);
      }
    }  // end pair of triangles

    else {  // 4-node quad

      stk::mesh::Entity elem = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1 + elem_id, singlePartVec);
      bulkData->declare_relation(elem, llnode, 0);
      bulkData->declare_relation(elem, lrnode, 1);
      bulkData->declare_relation(elem, urnode, 2);
      bulkData->declare_relation(elem, ulnode, 3);

      // Quad sideset construction
      if (x_GID == 0) {  // left edge of mesh, elem has side 3 on left boundary

        singlePartVec[0] = ssPartVec["SideSet0"];

        stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID));

        stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
        bulkData->declare_relation(elem, side, 3 /*local side id*/);

        bulkData->declare_relation(side, ulnode, 0);
        bulkData->declare_relation(side, llnode, 1);
      }
      if (x_GIDplus1 == nelem[0]) {  // right edge of mesh, elem has side 1 on right boundary

        singlePartVec[0] = ssPartVec["SideSet1"];

        stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID) + 1);

        stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
        bulkData->declare_relation(elem, side, 1 /*local side id*/);

        bulkData->declare_relation(side, lrnode, 0);
        bulkData->declare_relation(side, urnode, 1);
      }
      if (y_GID == 0) {  // bottom edge of mesh, elem has side 0 on lower boundary

        singlePartVec[0] = ssPartVec["SideSet2"];

        stk::mesh::EntityId side_id = (stk::mesh::EntityId)(x_GID);

        stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
        bulkData->declare_relation(elem, side, 0 /*local side id*/);

        bulkData->declare_relation(side, llnode, 0);
        bulkData->declare_relation(side, lrnode, 1);
      }
      if (y_GIDplus1 == nelem[1]) {  // tope edge of mesh, elem has side 2 on upper boundary

        singlePartVec[0] = ssPartVec["SideSet3"];

        stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * nelem[1]) + x_GID);

        stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
        bulkData->declare_relation(elem, side, 2 /*local side id*/);

        bulkData->declare_relation(side, urnode, 0);
        bulkData->declare_relation(side, ulnode, 1);
      }
    }  // end 4 node quad

    double* llnode_coord = stk::mesh::field_data(*coordinates_field, llnode);
    llnode_coord[0]      = x[0][x_GID];
    llnode_coord[1]      = x[1][y_GID];
    double* lrnode_coord = stk::mesh::field_data(*coordinates_field, lrnode);
    lrnode_coord[0]      = x[0][x_GIDplus1];
    lrnode_coord[1]      = x[1][y_GID];
    double* urnode_coord = stk::mesh::field_data(*coordinates_field, urnode);
    urnode_coord[0]      = x[0][x_GIDplus1];
    urnode_coord[1]      = x[1][y_GIDplus1];
    double* ulnode_coord = stk::mesh::field_data(*coordinates_field, ulnode);
    ulnode_coord[0]      = x[0][x_GID];
    ulnode_coord[1]      = x[1][y_GIDplus1];

    // Nodesets

    if (x_GID == 0) {  // left of elem is on left bdy
      singlePartVec[0] = nsPartVec["NodeSet0"];
      bulkData->change_entity_parts(llnode, singlePartVec);
      bulkData->change_entity_parts(ulnode, singlePartVec);
    }
    if ((x_GIDplus1) == nelem[0]) {  // right of elem is on right bdy
      singlePartVec[0] = nsPartVec["NodeSet1"];
      bulkData->change_entity_parts(lrnode, singlePartVec);
      bulkData->change_entity_parts(urnode, singlePartVec);
    }
    if (y_GID == 0) {  // bottom of elem is on bottom bdy
      singlePartVec[0] = nsPartVec["NodeSet2"];
      bulkData->change_entity_parts(llnode, singlePartVec);
      bulkData->change_entity_parts(lrnode, singlePartVec);
    }
    if ((y_GIDplus1) == nelem[1]) {  // top of elem is on top bdy
      singlePartVec[0] = nsPartVec["NodeSet3"];
      bulkData->change_entity_parts(ulnode, singlePartVec);
      bulkData->change_entity_parts(urnode, singlePartVec);
    }

    // Periodic cases -- last node wraps around to first
    if ((x_GIDplus1) == 0) {  // right of elem is on right bdy
      singlePartVec[0] = nsPartVec["NodeSet0"];
      bulkData->change_entity_parts(lrnode, singlePartVec);
      bulkData->change_entity_parts(urnode, singlePartVec);
    }
    if ((y_GIDplus1) == 0) {  // top of elem is on top bdy
      singlePartVec[0] = nsPartVec["NodeSet2"];
      bulkData->change_entity_parts(ulnode, singlePartVec);
      bulkData->change_entity_parts(urnode, singlePartVec);
    }

    // Single node at origin
    if (x_GID == 0 && y_GID == 0) {
      singlePartVec[0] = nsPartVec["NodeSet99"];
      bulkData->change_entity_parts(llnode, singlePartVec);
    }
    // Periodic cases
    if (x_GIDplus1 == 0 && y_GIDplus1 == 0) {
      singlePartVec[0] = nsPartVec["NodeSet99"];
      bulkData->change_entity_parts(urnode, singlePartVec);
    }
    if (x_GID == 0 && y_GIDplus1 == 0) {
      singlePartVec[0] = nsPartVec["NodeSet99"];
      bulkData->change_entity_parts(ulnode, singlePartVec);
    }
    if (x_GIDplus1 == 0 && y_GID == 0) {  // single node at bottom corner
      singlePartVec[0] = nsPartVec["NodeSet99"];
      bulkData->change_entity_parts(lrnode, singlePartVec);
    }
  }
}

template <>
void
Albany::TmplSTKMeshStruct<3>::buildMesh(const Teuchos::RCP<Teuchos_Comm const>& commT)
{
  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  std::vector<GO>       elemNumber(3);
  unsigned int          ebNo;

  const GO nodes_x  = periodic_x ? nelem[0] : nelem[0] + 1;
  const GO nodes_xy = (periodic_y && periodic_y) ? nodes_x * nelem[1] : nodes_x * (nelem[1] + 1);
  const GO mod_x    = periodic_x ? nelem[0] : std::numeric_limits<GO>::max();
  const GO mod_y    = periodic_y ? nelem[1] : std::numeric_limits<GO>::max();
  const GO mod_z    = periodic_z ? nelem[2] : std::numeric_limits<GO>::max();

  AbstractSTKFieldContainer::STKFieldType* coordinates_field = fieldContainer->getCoordinatesField();

  if (periodic_x) {
    this->PBCStruct.periodic[0] = true;
    this->PBCStruct.scale[0]    = scale[0];
  }
  if (periodic_y) {
    this->PBCStruct.periodic[1] = true;
    this->PBCStruct.scale[1]    = scale[1];
  }
  if (periodic_z) {
    this->PBCStruct.periodic[2] = true;
    this->PBCStruct.scale[2]    = scale[2];
  }

  // Create elements and node IDs
  for (int i = 0; i < elem_map->getLocalNumElements(); i++) {
    const GO elem_GID = elem_map->getGlobalElement(i);
    const GO z_GID    = elem_GID / (nelem[0] * nelem[1]);  // mesh column number
    const GO xy_plane = elem_GID % (nelem[0] * nelem[1]);
    const GO x_GID    = xy_plane % nelem[0];  // mesh column number
    const GO y_GID    = xy_plane / nelem[0];  // mesh row number

    const GO x_GIDplus1 = (x_GID + 1) % mod_x;  // = x_GID+1 unless last element of periodic system
    const GO y_GIDplus1 = (y_GID + 1) % mod_y;  // = y_GID+1 unless last element of periodic system
    const GO z_GIDplus1 = (z_GID + 1) % mod_z;  // = z_GID+1 unless last element of periodic system

    const GO lower_left  = x_GID + nodes_x * y_GID + nodes_xy * z_GID;
    const GO lower_right = x_GIDplus1 + nodes_x * y_GID + nodes_xy * z_GID;
    const GO upper_right = x_GIDplus1 + nodes_x * y_GIDplus1 + nodes_xy * z_GID;
    const GO upper_left  = x_GID + nodes_x * y_GIDplus1 + nodes_xy * z_GID;

    const GO lower_left_back  = x_GID + nodes_x * y_GID + nodes_xy * z_GIDplus1;
    const GO lower_right_back = x_GIDplus1 + nodes_x * y_GID + nodes_xy * z_GIDplus1;
    const GO upper_right_back = x_GIDplus1 + nodes_x * y_GIDplus1 + nodes_xy * z_GIDplus1;
    const GO upper_left_back  = x_GID + nodes_x * y_GIDplus1 + nodes_xy * z_GIDplus1;

    stk::mesh::EntityId elem_id = (stk::mesh::EntityId)elem_GID;

    // Calculate elemNumber of element
    elemNumber[0] = x_GID;
    elemNumber[1] = y_GID;
    elemNumber[2] = z_GID;

    // find out which EB the element is in

    if (numEB == 1)  // assume all elements are in element block if there is only one
      ebNo = 0;
    else {
      for (ebNo = 0; ebNo < numEB; ebNo++)
        // Does the elemNumber lie in the EB?
        if (EBSpecs[ebNo].inEB(elemNumber)) break;

      if (ebNo == numEB) {  // error, we didn't find an element block that this
                            // element should fit in

        ALBANY_ABORT(std::endl << "Error: Could not place element " << elem_GID << " in its corresponding element block." << std::endl);
      }
    }

    singlePartVec[0] = partVec[ebNo];

    // Add one to IDs because STK requires 1-based
    stk::mesh::Entity elem    = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1 + elem_id, singlePartVec);
    stk::mesh::Entity llnode  = bulkData->declare_entity(stk::topology::NODE_RANK, 1 + lower_left, nodePartVec);
    stk::mesh::Entity lrnode  = bulkData->declare_entity(stk::topology::NODE_RANK, 1 + lower_right, nodePartVec);
    stk::mesh::Entity urnode  = bulkData->declare_entity(stk::topology::NODE_RANK, 1 + upper_right, nodePartVec);
    stk::mesh::Entity ulnode  = bulkData->declare_entity(stk::topology::NODE_RANK, 1 + upper_left, nodePartVec);
    stk::mesh::Entity llnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, 1 + lower_left_back, nodePartVec);
    stk::mesh::Entity lrnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, 1 + lower_right_back, nodePartVec);
    stk::mesh::Entity urnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, 1 + upper_right_back, nodePartVec);
    stk::mesh::Entity ulnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, 1 + upper_left_back, nodePartVec);
    bulkData->declare_relation(elem, llnode, 0);
    bulkData->declare_relation(elem, lrnode, 1);
    bulkData->declare_relation(elem, urnode, 2);
    bulkData->declare_relation(elem, ulnode, 3);
    bulkData->declare_relation(elem, llnodeb, 4);
    bulkData->declare_relation(elem, lrnodeb, 5);
    bulkData->declare_relation(elem, urnodeb, 6);
    bulkData->declare_relation(elem, ulnodeb, 7);

    /*
        if(proc_rank_field){
          int* p_rank = stk::mesh::field_data(*proc_rank_field, elem);
          p_rank[0] = comm->MyPID();
        }
    */

    double* coord;
    coord    = stk::mesh::field_data(*coordinates_field, llnode);
    coord[0] = x[0][x_GID];
    coord[1] = x[1][y_GID];
    coord[2] = x[2][z_GID];
    coord    = stk::mesh::field_data(*coordinates_field, lrnode);
    coord[0] = x[0][x_GIDplus1];
    coord[1] = x[1][y_GID];
    coord[2] = x[2][z_GID];
    coord    = stk::mesh::field_data(*coordinates_field, urnode);
    coord[0] = x[0][x_GIDplus1];
    coord[1] = x[1][y_GIDplus1];
    coord[2] = x[2][z_GID];
    coord    = stk::mesh::field_data(*coordinates_field, ulnode);
    coord[0] = x[0][x_GID];
    coord[1] = x[1][y_GIDplus1];
    coord[2] = x[2][z_GID];

    coord    = stk::mesh::field_data(*coordinates_field, llnodeb);
    coord[0] = x[0][x_GID];
    coord[1] = x[1][y_GID];
    coord[2] = x[2][z_GIDplus1];
    coord    = stk::mesh::field_data(*coordinates_field, lrnodeb);
    coord[0] = x[0][x_GIDplus1];
    coord[1] = x[1][y_GID];
    coord[2] = x[2][z_GIDplus1];
    coord    = stk::mesh::field_data(*coordinates_field, urnodeb);
    coord[0] = x[0][x_GIDplus1];
    coord[1] = x[1][y_GIDplus1];
    coord[2] = x[2][z_GIDplus1];
    coord    = stk::mesh::field_data(*coordinates_field, ulnodeb);
    coord[0] = x[0][x_GID];
    coord[1] = x[1][y_GIDplus1];
    coord[2] = x[2][z_GIDplus1];

    /*
     * Do sidesets
     */

    // Hex sideset construction

    if (x_GID == 0) {  // left edge of mesh, elem has side 3 on left boundary

      // Sideset construction
      // elem has side 3 (0473) on left boundary

      singlePartVec[0] = ssPartVec["SideSet0"];

      stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID) + (nelem[0] + nelem[1]) * 2 * z_GID);

      stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
      bulkData->declare_relation(elem, side, 3 /*local side id*/);

      bulkData->declare_relation(side, llnode, 0);
      bulkData->declare_relation(side, llnodeb, 2);
      bulkData->declare_relation(side, ulnodeb, 3);
      bulkData->declare_relation(side, ulnode, 1);
    }
    if ((x_GIDplus1) == nelem[0]) {  // right edge of mesh, elem has side 1 on right boundary

      // elem has side 1 (1265) on right boundary

      singlePartVec[0] = ssPartVec["SideSet1"];

      stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID) + 1 + (nelem[0] + nelem[1]) * 2 * z_GID);

      stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
      bulkData->declare_relation(elem, side, 1 /*local side id*/);

      bulkData->declare_relation(side, lrnode, 0);
      bulkData->declare_relation(side, urnode, 1);
      bulkData->declare_relation(side, urnodeb, 3);
      bulkData->declare_relation(side, lrnodeb, 2);
    }
    if (y_GID == 0) {  // bottom edge of mesh, elem has side 0 on lower boundary

      // elem has side 0 (0154) on bottom boundary

      singlePartVec[0] = ssPartVec["SideSet2"];

      stk::mesh::EntityId side_id = (stk::mesh::EntityId)(x_GID + (nelem[0] + nelem[1]) * 2 * z_GID);

      stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
      bulkData->declare_relation(elem, side, 0 /*local side id*/);

      bulkData->declare_relation(side, llnode, 0);
      bulkData->declare_relation(side, lrnode, 1);
      bulkData->declare_relation(side, lrnodeb, 3);
      bulkData->declare_relation(side, llnodeb, 2);
    }
    if ((y_GIDplus1) == nelem[1]) {  // tope edge of mesh, elem has side 2 on upper boundary

      // elem has side 2 (2376) on top boundary

      singlePartVec[0] = ssPartVec["SideSet3"];

      stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * nelem[1]) + x_GID + (nelem[0] + nelem[1]) * 2 * z_GID);

      stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
      bulkData->declare_relation(elem, side, 2 /*local side id*/);

      bulkData->declare_relation(side, urnode, 0);
      bulkData->declare_relation(side, ulnode, 1);
      bulkData->declare_relation(side, ulnodeb, 3);
      bulkData->declare_relation(side, urnodeb, 2);
    }
    if (z_GID == 0) {
      // elem has side 4 (0321) on front boundary

      singlePartVec[0] = ssPartVec["SideSet4"];

      stk::mesh::EntityId side_id = (stk::mesh::EntityId)((nelem[0] + nelem[1]) * 2 * nelem[2] + x_GID + (2 * nelem[0]) * y_GID);

      stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
      bulkData->declare_relation(elem, side, 4 /*local side id*/);

      bulkData->declare_relation(side, llnode, 0);
      bulkData->declare_relation(side, ulnode, 3);
      bulkData->declare_relation(side, urnode, 2);
      bulkData->declare_relation(side, lrnode, 1);
    }
    if ((z_GIDplus1) == nelem[2]) {
      // elem has side 5 (4567) on back boundary

      singlePartVec[0]            = ssPartVec["SideSet5"];
      stk::mesh::EntityId side_id = (stk::mesh::EntityId)((nelem[0] + nelem[1]) * 2 * nelem[2] + x_GID + (2 * nelem[0]) * y_GID + nelem[0]);

      stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
      bulkData->declare_relation(elem, side, 5 /*local side id*/);

      bulkData->declare_relation(side, llnodeb, 0);
      bulkData->declare_relation(side, lrnodeb, 1);
      bulkData->declare_relation(side, urnodeb, 2);
      bulkData->declare_relation(side, ulnodeb, 3);
    }

    /*
     * Do Nodesets
     */

    if (x_GID == 0) {
      singlePartVec[0] = nsPartVec["NodeSet0"];
      bulkData->change_entity_parts(llnode, singlePartVec);   // node 0
      bulkData->change_entity_parts(ulnode, singlePartVec);   // 3
      bulkData->change_entity_parts(llnodeb, singlePartVec);  // 4
      bulkData->change_entity_parts(ulnodeb, singlePartVec);  // 7
    }
    if ((x_GIDplus1) == nelem[0]) {
      singlePartVec[0] = nsPartVec["NodeSet1"];
      bulkData->change_entity_parts(lrnode, singlePartVec);   // 1
      bulkData->change_entity_parts(urnode, singlePartVec);   // 2
      bulkData->change_entity_parts(lrnodeb, singlePartVec);  // 5
      bulkData->change_entity_parts(urnodeb, singlePartVec);  // 6
    }
    if (y_GID == 0) {
      singlePartVec[0] = nsPartVec["NodeSet2"];
      bulkData->change_entity_parts(llnode, singlePartVec);   // 0
      bulkData->change_entity_parts(lrnode, singlePartVec);   // 1
      bulkData->change_entity_parts(llnodeb, singlePartVec);  // 4
      bulkData->change_entity_parts(lrnodeb, singlePartVec);  // 5
    }
    if ((y_GIDplus1) == nelem[1]) {
      singlePartVec[0] = nsPartVec["NodeSet3"];
      bulkData->change_entity_parts(ulnode, singlePartVec);   // 3
      bulkData->change_entity_parts(urnode, singlePartVec);   // 2
      bulkData->change_entity_parts(ulnodeb, singlePartVec);  // 7
      bulkData->change_entity_parts(urnodeb, singlePartVec);  // 6
    }
    if (z_GID == 0) {
      singlePartVec[0] = nsPartVec["NodeSet4"];
      bulkData->change_entity_parts(llnode, singlePartVec);  // 0
      bulkData->change_entity_parts(lrnode, singlePartVec);  // 1
      bulkData->change_entity_parts(ulnode, singlePartVec);  // 3
      bulkData->change_entity_parts(urnode, singlePartVec);  // 2
    }
    if ((z_GIDplus1) == nelem[2]) {
      singlePartVec[0] = nsPartVec["NodeSet5"];
      bulkData->change_entity_parts(llnodeb, singlePartVec);  // 4
      bulkData->change_entity_parts(lrnodeb, singlePartVec);  // 5
      bulkData->change_entity_parts(ulnodeb, singlePartVec);  // 7
      bulkData->change_entity_parts(urnodeb, singlePartVec);  // 6
    }

    // Periodic cases -- last node wraps around to first
    if ((x_GIDplus1) == 0) {
      singlePartVec[0] = nsPartVec["NodeSet0"];
      bulkData->change_entity_parts(lrnode, singlePartVec);   // 1
      bulkData->change_entity_parts(urnode, singlePartVec);   // 2
      bulkData->change_entity_parts(lrnodeb, singlePartVec);  // 5
      bulkData->change_entity_parts(urnodeb, singlePartVec);  // 6
    }
    if ((y_GIDplus1) == 0) {
      singlePartVec[0] = nsPartVec["NodeSet2"];
      bulkData->change_entity_parts(ulnode, singlePartVec);   // 3
      bulkData->change_entity_parts(urnode, singlePartVec);   // 2
      bulkData->change_entity_parts(ulnodeb, singlePartVec);  // 7
      bulkData->change_entity_parts(urnodeb, singlePartVec);  // 6
    }
    if ((z_GIDplus1) == 0) {
      singlePartVec[0] = nsPartVec["NodeSet4"];
      bulkData->change_entity_parts(llnodeb, singlePartVec);  // 4
      bulkData->change_entity_parts(lrnodeb, singlePartVec);  // 5
      bulkData->change_entity_parts(ulnodeb, singlePartVec);  // 7
      bulkData->change_entity_parts(urnodeb, singlePartVec);  // 6
    }

    // Single node at origin
    if (x_GID == 0 && y_GID == 0 && z_GID == 0) {
      singlePartVec[0] = nsPartVec["NodeSet99"];
      bulkData->change_entity_parts(llnode, singlePartVec);  // node 0
    }
    // Seven Periodic cases
    if (x_GIDplus1 == 0 && y_GID == 0 && z_GID == 0) {
      singlePartVec[0] = nsPartVec["NodeSet99"];
      bulkData->change_entity_parts(lrnode, singlePartVec);  // node 0
    }
    if (x_GID == 0 && y_GIDplus1 == 0 && z_GID == 0) {
      singlePartVec[0] = nsPartVec["NodeSet99"];
      bulkData->change_entity_parts(ulnode, singlePartVec);  // node 0
    }
    if (x_GIDplus1 == 0 && y_GIDplus1 == 0 && z_GID == 0) {
      singlePartVec[0] = nsPartVec["NodeSet99"];
      bulkData->change_entity_parts(urnode, singlePartVec);  // node 0
    }
    if (x_GID == 0 && y_GID == 0 && z_GIDplus1 == 0) {
      singlePartVec[0] = nsPartVec["NodeSet99"];
      bulkData->change_entity_parts(llnodeb, singlePartVec);  // node 0
    }
    if (x_GIDplus1 == 0 && y_GID == 0 && z_GIDplus1 == 0) {
      singlePartVec[0] = nsPartVec["NodeSet99"];
      bulkData->change_entity_parts(lrnodeb, singlePartVec);  // node 0
    }
    if (x_GID == 0 && y_GIDplus1 == 0 && z_GIDplus1 == 0) {
      singlePartVec[0] = nsPartVec["NodeSet99"];
      bulkData->change_entity_parts(ulnodeb, singlePartVec);  // node 0
    }
    if (x_GIDplus1 == 0 && y_GIDplus1 == 0 && z_GIDplus1 == 0) {
      singlePartVec[0] = nsPartVec["NodeSet99"];
      bulkData->change_entity_parts(urnodeb, singlePartVec);  // node 0
    }

  }  // end element loop
}

template <>
Teuchos::RCP<Teuchos::ParameterList const>
Albany::TmplSTKMeshStruct<0>::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getValidGenericSTKParameters("ValidSTK0D_DiscParams");

  return validPL;
}

template <>
Teuchos::RCP<Teuchos::ParameterList const>
Albany::TmplSTKMeshStruct<1>::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getValidGenericSTKParameters("ValidSTK1D_DiscParams");
  validPL->set<bool>("Periodic_x BC", false, "Flag to indicate periodic mesh in x-dimesnsion");
  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  validPL->sublist("Required Fields Info", false, "Info for the loading of the required fields");

  // Multiple element blocks parameters
  validPL->set<int>("Element Blocks", 1, "Number of elements blocks that span the X domain");

  for (unsigned int i = 0; i < numEB; i++) {
    std::stringstream ss;
    ss << "Block " << i;

    validPL->set<std::string>(ss.str(), "begins at 0 ends at 100 length 1.0 named Bck0", "Beginning and ending parametric coordinates of block, block name");
  }

  return validPL;
}

template <>
Teuchos::RCP<Teuchos::ParameterList const>
Albany::TmplSTKMeshStruct<2>::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getValidGenericSTKParameters("ValidSTK2D_DiscParams");

  validPL->set<bool>("Periodic_x BC", false, "Flag to indicate periodic mesh in x-dimesnsion");
  validPL->set<bool>("Periodic_y BC", false, "Flag to indicate periodic mesh in y-dimesnsion");
  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<int>("2D Elements", 0, "Number of Elements in Y discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  validPL->set<double>("2D Scale", 1.0, "Height of Y discretization");
  validPL->set<std::string>("Cell Topology", "Quad", "Quad or Tri Cell Topology");
  validPL->sublist("Required Fields Info", false, "Info for the loading of the required fields");
  validPL->set<bool>("Write points coordinates to ascii file", false, "If true, writes the mesh points coordinates on file");

  // Multiple element blocks parameters
  validPL->set<int>("Element Blocks", 1, "Number of elements blocks that span the X-Y domain");

  for (unsigned int i = 0; i < numEB; i++) {
    std::stringstream ss;
    ss << "Block " << i;

    validPL->set<std::string>(
        ss.str(), "begins at (0, 0) ends at (100, 100) length (1.0, 1.0) named Bck0", "Beginning and ending parametric coordinates of block, block name");
  }

  return validPL;
}

template <>
Teuchos::RCP<Teuchos::ParameterList const>
Albany::TmplSTKMeshStruct<3>::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getValidGenericSTKParameters("ValidSTK3D_DiscParams");

  validPL->set<bool>("Periodic_x BC", false, "Flag to indicate periodic mesh in x-dimesnsion");
  validPL->set<bool>("Periodic_y BC", false, "Flag to indicate periodic mesh in y-dimesnsion");
  validPL->set<bool>("Periodic_z BC", false, "Flag to indicate periodic mesh in z-dimesnsion");
  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<int>("2D Elements", 0, "Number of Elements in Y discretization");
  validPL->set<int>("3D Elements", 0, "Number of Elements in Z discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  validPL->set<double>("2D Scale", 1.0, "Depth of Y discretization");
  validPL->set<double>("3D Scale", 1.0, "Height of Z discretization");
  validPL->sublist("Required Fields Info", false, "Info for the loading of the required fields");
  validPL->set<int>("Number Of Time Derivatives", 1, "Number of time derivatives in use in the problem");

  // Multiple element blocks parameters
  validPL->set<int>("Element Blocks", 1, "Number of elements blocks that span the X-Y-Z domain");

  for (unsigned int i = 0; i < numEB; i++) {
    std::stringstream ss;
    ss << "Block " << i;

    validPL->set<std::string>(
        ss.str(),
        "begins at (0, 0, 0) ends at (100, 100, 100) length (1.0, 1.0, 1.0) "
        "named Bck0",
        "Beginning and ending parametric coordinates of block, block name");
  }

  return validPL;
}
