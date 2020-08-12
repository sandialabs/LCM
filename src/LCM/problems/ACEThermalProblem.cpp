// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "ACEThermalProblem.hpp"

#include "Albany_BCUtils.hpp"
#include "Albany_Utils.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Shards_CellTopology.hpp"

Albany::ACEThermalProblem::ACEThermalProblem(
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    const Teuchos::RCP<ParamLib>&               param_lib,
    int const                                   num_dim,
    Teuchos::RCP<Teuchos::Comm<int> const>&     comm)
    : Albany::AbstractProblem(params, param_lib /*, distParamLib_*/),
      params_(params),
      num_dim_(num_dim),
      comm_(comm),
      use_sdbcs_(false)
{
  this->setNumEquations(1);
  // We just have 1 PDE/node
  neq = 1;

  if (params->isType<std::string>("MaterialDB Filename")) {
    std::string mtrl_db_filename = params->get<std::string>("MaterialDB Filename");
    // Create Material Database
    material_db_ = Teuchos::rcp(new Albany::MaterialDatabase(mtrl_db_filename, comm_));
  }
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  if (params->isSublist("Stabilization")) {
    Teuchos::ParameterList& stab_pl = params->sublist("Stabilization");
    use_stab_                       = stab_pl.get<bool>("Use Stabilization", false);
    stab_value_                     = stab_pl.get<double>("Stabilization Parameter Value", 1.0);
    // Maximum value of x and z in the domain
    // IKT 6/21/20 TODO: figure these out from the geometry
    x_max_ = stab_pl.get<double>("Max Value of x-Coord", 0.0);
    z_max_ = stab_pl.get<double>("Max Value of z-Coord", 0.0);
    // Maximum time up t0 which you want to apply stabilization
    // IKT 6/21/20 TODO? set this to the maximum time in the input file
    max_time_stab_ = stab_pl.get<double>("Max Stabilization Time", 1.0e10);
    tau_type_      = stab_pl.get<std::string>("Tau Type", "Proportional to Mesh Size");
    stab_type_     = stab_pl.get<std::string>("Stabilization Type", "Laplacian");
    if ((stab_type_ != "SUPG") && (stab_type_ != "Laplacian")) {
      ALBANY_ASSERT(false, "Invalid Stabilization Type!  Valid options are 'SUPG' and 'Laplacian'.");
    }
    if (use_stab_ == true) {
      *out << "Thermal problem: using stabilization.\n";
      *out << "   Stabilization Type = " << stab_type_ << "\n";
      *out << "   Tau Type = " << tau_type_ << "\n";
      *out << "   Stabilization Parameter Value = " << stab_value_ << "\n";
    }
  } else {
    tau_type_   = "None";
    stab_value_ = 0.0;
  }
}

Albany::ACEThermalProblem::~ACEThermalProblem() {}

void
Albany::ACEThermalProblem::buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> mesh_specs,
    Albany::StateManager&                                    state_mgr)
{
  /* Construct All Phalanx Evaluators */
  int                                 phys_sets = mesh_specs.size();  // number of blocks
  Teuchos::RCP<Teuchos::FancyOStream> out       = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "ACE Thermal Problem Num MeshSpecs: " << phys_sets << "\n";
  fm.resize(phys_sets);
  eb_names_.resize(phys_sets);
  bool init_step = true;
  for (int ps = 0; ps < phys_sets; ps++) {
    if (ps < phys_sets - 1) {
      eb_names_.resize(ps + 1);
    }
    std::string element_block_name = mesh_specs[ps]->ebName;
    eb_names_[ps]                  = element_block_name;
    fm[ps]                         = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *mesh_specs[ps], state_mgr, BUILD_RESID_FM, Teuchos::null);

    if (mesh_specs[ps]->nsNames.size() > 0) {  // Build a nodeset evaluator if nodesets are present
      constructDirichletEvaluators(mesh_specs[ps]->nsNames);
    }

    // Check if have Neumann sublist; throw error if attempting to specify
    // Neumann BCs, but there are no sidesets in the input mesh
    bool is_neumann_pl = params->isSublist("Neumann BCs");
    if (is_neumann_pl && !(mesh_specs[ps]->ssNames.size() > 0)) {
      ALBANY_ABORT("You are attempting to set Neumann BCs on a mesh with no sidesets!");
    }

    if (mesh_specs[ps]->ssNames.size() > 0) {  // Build a sideset evaluator if sidesets are present
      constructNeumannEvaluators(mesh_specs[ps]);
    }
    eb_names_.resize(phys_sets);
  }
}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
Albany::ACEThermalProblem::buildEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits>&      fm0,
    Albany::MeshSpecsStruct const&              mesh_specs,
    Albany::StateManager&                       state_mgr,
    Albany::FieldManagerChoice                  fmchoice,
    const Teuchos::RCP<Teuchos::ParameterList>& response_list)
{
  // Call constructEvaluators<EvalT>(*rfm[0], *mesh_specs[0], state_mgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<ACEThermalProblem>              op(*this, fm0, mesh_specs, state_mgr, fmchoice, response_list);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

// Dirichlet BCs
void
Albany::ACEThermalProblem::constructDirichletEvaluators(std::vector<std::string> const& node_set_ids)
{
  // Construct BC evaluators for all node sets and names
  std::vector<std::string> bc_names(neq);
  bc_names[0] = "T";
  Albany::BCUtils<Albany::DirichletTraits> bc_utils;
  dfm         = bc_utils.constructBCEvaluators(node_set_ids, bc_names, this->params, this->paramLib);
  use_sdbcs_  = bc_utils.useSDBCs();
  offsets_    = bc_utils.getOffsets();
  nodeSetIDs_ = bc_utils.getNodeSetIDs();
}

// Neumann BCs
void
Albany::ACEThermalProblem::constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& mesh_specs)
{
  // Note: we only enter this function if sidesets are defined in the mesh file
  // i.e. mesh_specs.ssNames.size() > 0

  Albany::BCUtils<Albany::NeumannTraits> bc_utils;

  // Check to make sure that Neumann BCs are given in the input file

  if (!bc_utils.haveBCSpecified(this->params)) return;

  // Construct BC evaluators for all side sets and names
  // Note that the string index sets up the equation offset, so ordering is
  // important
  std::vector<std::string>            bc_names(neq);
  Teuchos::ArrayRCP<std::string>      dof_names(neq);
  Teuchos::Array<Teuchos::Array<int>> offsets;
  offsets.resize(neq);

  bc_names[0]  = "T";
  dof_names[0] = "Temperature";
  offsets[0].resize(1);
  offsets[0][0] = 0;

  // Construct BC evaluators for all possible names of conditions
  // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not
  // both
  std::vector<std::string> cond_names(5);
  // dudx, dudy, dudz, dudn, scaled jump (internal surface), or robin (like DBC
  // plus scaled jump)

  // Note that sidesets are only supported for two and 3D currently
  if (num_dim_ == 2)
    cond_names[0] = "(dudx, dudy)";
  else if (num_dim_ == 3)
    cond_names[0] = "(dudx, dudy, dudz)";
  else
    ALBANY_ABORT("\nError: Sidesets only supported in 2 and 3D.\n");

  cond_names[1] = "dudn";
  cond_names[2] = "scaled jump";
  cond_names[3] = "robin";
  cond_names[4] = "radiate";

  nfm.resize(1);  // Heat problem only has one physics set
  nfm[0] = bc_utils.constructBCEvaluators(
      mesh_specs, bc_names, dof_names, false, 0, cond_names, offsets, dl_, this->params, this->paramLib);
}

Teuchos::RCP<Teuchos::ParameterList const>
Albany::ACEThermalProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> valid_pl = this->getGenericProblemParams("ValidACEThermalProblemParams");

  valid_pl->set<std::string>("MaterialDB Filename", "materials.xml", "Filename of material database xml file");
  valid_pl->sublist("Stabilization", false, "Parameter list with stabilization parameters");
  // valid_pl->set<bool>("Use Stabilization", false, "Flag to turn on stabilization");
  // valid_pl->set<double>("Stabilization Parameter Value", 1.0, "Value of stabilization parameter");
  return valid_pl;
}
