// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_DiscretizationFactory.hpp"

#include "Albany_AsciiSTKMesh2D.hpp"
#include "Albany_AsciiSTKMeshStruct.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_GmshSTKMeshStruct.hpp"
#include "Albany_IossSTKMeshStruct.hpp"
#include "Albany_Macros.hpp"
#include "Albany_STK3DPointStruct.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_SideSetSTKMeshStruct.hpp"
#include "Albany_TmplSTKMeshStruct.hpp"

Albany::DiscretizationFactory::DiscretizationFactory(const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams, const Teuchos::RCP<Teuchos_Comm const>& commT_)
    : commT(commT_)
{
  discParams = Teuchos::sublist(topLevelParams, "Discretization", true);

  if (topLevelParams->isSublist("Piro")) {
    piroParams = Teuchos::sublist(topLevelParams, "Piro", true);
  }

  if (topLevelParams->isSublist("Problem")) {
    Teuchos::RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(topLevelParams, "Problem", true);

    if (problemParams->isSublist("Adaptation")) {
      adaptParams = Teuchos::sublist(problemParams, "Adaptation", true);
    }
    if (problemParams->isSublist("Catalyst")) {
      catalystParams = Teuchos::sublist(problemParams, "Catalyst", true);
    }
  }
}

Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>
Albany::DiscretizationFactory::createMeshSpecs()
{
  meshStruct = createMeshStruct(discParams, adaptParams, commT);
  return meshStruct->getMeshSpecs();
}

Teuchos::RCP<Albany::AbstractMeshStruct>
Albany::DiscretizationFactory::createMeshStruct(
    Teuchos::RCP<Teuchos::ParameterList> disc_params,
    Teuchos::RCP<Teuchos::ParameterList> adapt_params,
    Teuchos::RCP<Teuchos_Comm const>     comm)
{
  std::string& method = disc_params->get("Method", "STK1D");
  if (method == "STK1D") {
    return Teuchos::rcp(new Albany::TmplSTKMeshStruct<1>(disc_params, adapt_params, comm));
  } else if (method == "STK0D") {
    return Teuchos::rcp(new Albany::TmplSTKMeshStruct<0>(disc_params, adapt_params, comm));
  } else if (method == "STK2D") {
    return Teuchos::rcp(new Albany::TmplSTKMeshStruct<2>(disc_params, adapt_params, comm));
  } else if (method == "STK3D") {
    return Teuchos::rcp(new Albany::TmplSTKMeshStruct<3>(disc_params, adapt_params, comm));
  } else if (method == "STK3DPoint") {
    return Teuchos::rcp(new Albany::STK3DPointStruct(disc_params, comm));
  } else if (method == "Ioss" || method == "Exodus" || method == "Pamgen") {
    return Teuchos::rcp(new Albany::IossSTKMeshStruct(disc_params, adapt_params, comm));
  } else if (method == "Ascii") {
    return Teuchos::rcp(new Albany::AsciiSTKMeshStruct(disc_params, comm));
  } else if (method == "Ascii2D") {
    return Teuchos::rcp(new Albany::AsciiSTKMesh2D(disc_params, comm));
  } else if (method == "Hacky Ascii2D") {
    // FixME very hacky! needed for printing 2d mesh
    Teuchos::RCP<Albany::GenericSTKMeshStruct> meshStruct2D;
    meshStruct2D                                                   = Teuchos::rcp(new Albany::AsciiSTKMesh2D(disc_params, comm));
    Teuchos::RCP<Albany::StateInfoStruct>                      sis = Teuchos::rcp(new Albany::StateInfoStruct);
    Albany::AbstractFieldContainer::FieldContainerRequirements req;
    int                                                        neq = 2;
    meshStruct2D->setFieldAndBulkData(comm, disc_params, neq, req, sis, meshStruct2D->getMeshSpecs()[0]->worksetSize);
    Ioss::Init::Initializer                io;
    Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data = Teuchos::rcp(new stk::io::StkMeshIoBroker(MPI_COMM_WORLD));
    mesh_data->set_bulk_data(*meshStruct2D->bulkData);
    std::string const& output_filename = disc_params->get("Exodus Output File Name", "ice_mesh.2d.exo");
    size_t             idx             = mesh_data->create_output_mesh(output_filename, stk::io::WRITE_RESULTS);
    mesh_data->process_output_request(idx, 0.0);
  } else if (method == "Gmsh") {
    return Teuchos::rcp(new Albany::GmshSTKMeshStruct(disc_params, comm));
  } else

    ALBANY_ABORT(
        std::endl
        << "Error!  Unknown discretization method in "
           "DiscretizationFactory: "
        << method << "!" << std::endl
        << "Supplied parameter list is " << std::endl
        << *disc_params << "\nValid Methods are: STK1D, STK2D, STK3D, STK3DPoint, Ioss, "
        << " Exodus, Sim, Ascii,"
        << " Ascii2D, Extruded" << std::endl);
  return Teuchos::null;
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretization(
    unsigned int                                              neq,
    const Teuchos::RCP<Albany::StateInfoStruct>&              sis,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    const Teuchos::RCP<Albany::RigidBodyModes>&               rigidBodyModes)
{
  return createDiscretization(neq, empty_side_set_equations, sis, empty_side_set_sis, req, empty_side_set_req, rigidBodyModes);
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretization(
    unsigned int                                                                     neq,
    std::map<int, std::vector<std::string>> const&                                   sideSetEquations,
    const Teuchos::RCP<Albany::StateInfoStruct>&                                     sis,
    std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> const&              side_set_sis,
    const AbstractFieldContainer::FieldContainerRequirements&                        req,
    std::map<std::string, AbstractFieldContainer::FieldContainerRequirements> const& side_set_req,
    const Teuchos::RCP<Albany::RigidBodyModes>&                                      rigidBodyModes)
{
  ALBANY_PANIC(meshStruct == Teuchos::null, "meshStruct accessed, but it has not been constructed" << std::endl);

  setupInternalMeshStruct(neq, sis, side_set_sis, req, side_set_req);
  Teuchos::RCP<Albany::AbstractDiscretization> result = createDiscretizationFromInternalMeshStruct(sideSetEquations, rigidBodyModes);

  // Wrap the discretization in the catalyst decorator if needed.

  return result;
}

Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>
Albany::DiscretizationFactory::createMeshSpecs(Teuchos::RCP<Albany::AbstractMeshStruct> mesh)
{
  meshStruct = mesh;
  return meshStruct->getMeshSpecs();
}

void
Albany::DiscretizationFactory::setupInternalMeshStruct(
    unsigned int                                              neq,
    const Teuchos::RCP<Albany::StateInfoStruct>&              sis,
    const AbstractFieldContainer::FieldContainerRequirements& req)
{
  setupInternalMeshStruct(neq, sis, empty_side_set_sis, req, empty_side_set_req);
}

void
Albany::DiscretizationFactory::setupInternalMeshStruct(
    unsigned int                                                                     neq,
    const Teuchos::RCP<Albany::StateInfoStruct>&                                     sis,
    std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> const&              side_set_sis,
    const AbstractFieldContainer::FieldContainerRequirements&                        req,
    std::map<std::string, AbstractFieldContainer::FieldContainerRequirements> const& side_set_req)
{
  meshStruct->setFieldAndBulkData(commT, discParams, neq, req, sis, meshStruct->getMeshSpecs()[0]->worksetSize, side_set_sis, side_set_req);
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretizationFromInternalMeshStruct(const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes)
{
  return createDiscretizationFromInternalMeshStruct(empty_side_set_equations, rigidBodyModes);
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretizationFromInternalMeshStruct(
    std::map<int, std::vector<std::string>> const& sideSetEquations,
    const Teuchos::RCP<Albany::RigidBodyModes>&    rigidBodyModes)
{
  if (!piroParams.is_null() && !rigidBodyModes.is_null()) {
    rigidBodyModes->setPiroPL(piroParams);
  }
  auto ms   = Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(meshStruct);
  auto disc = Teuchos::rcp(new Albany::STKDiscretization(discParams, ms, commT, rigidBodyModes, sideSetEquations));
  if (!deferUpdateMesh) {
    disc->updateMesh();
  }
  return disc;
}

/* This function overwrite previous discretization parameter list */
void
Albany::DiscretizationFactory::setDiscretizationParameters(Teuchos::RCP<Teuchos::ParameterList> disc_params)
{
  discParams = disc_params;
}
