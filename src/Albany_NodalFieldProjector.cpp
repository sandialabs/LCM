// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_NodalFieldProjector.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_BasicTopologies.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractMeshStruct.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_MaterialDatabase.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_Utils.hpp"
#include "PHAL_LoadStateField.hpp"
#include "ProjectIPtoNodalField.hpp"

namespace Albany {

namespace {

Teuchos::RCP<PHX::DataLayout>
qp_layout_for(Teuchos::RCP<Layouts> const& dl, std::string const& layout)
{
  if (layout == "Scalar") return dl->qp_scalar;
  if (layout == "Vector") return dl->qp_vector;
  if (layout == "Tensor") return dl->qp_tensor;
  ALBANY_ABORT("NodalFieldProjector: unknown IP field layout '" << layout << "' (use Scalar, Vector, or Tensor).");
  return Teuchos::null;
}

}  // anonymous namespace

NodalFieldProjector::NodalFieldProjector(
    Teuchos::RCP<Application> const& app,
    std::vector<FieldSpec> const&    fields,
    std::string const&               mass_matrix_type,
    bool const                       output_to_exodus)
    : app_(app)
{
  using EvalT  = PHAL::AlbanyTraits::Residual;
  using Traits = PHAL::AlbanyTraits;

  auto  disc          = app_->getDiscretization();
  auto& state_manager = app_->getStateMgr();
  auto  mesh_specs    = disc->getMeshStruct()->getMeshSpecs();
  auto  phx_setup     = app_->getPhxSetup();

  // The "Use Composite Tet 10" flag is a per-element-block material property,
  // so the saved IP states of a composite block were written on the composite
  // quadrature. Read the same material database the assembly used, so the
  // projector integrates on the matching quadrature (see per-block handling
  // below). Absent a material database (non-mechanics problems), no block is
  // composite.
  Teuchos::RCP<MaterialDatabase> material_db;
  {
    auto problem_pl = app_->getProblemPL();
    if (problem_pl->isType<std::string>("MaterialDB Filename")) {
      auto comm   = app_->getComm();
      material_db = createMaterialDatabase(problem_pl, comm);
    }
  }

  int const num_blocks = mesh_specs.size();
  fms_.resize(num_blocks);
  eval_names_.resize(num_blocks);

  for (int eb = 0; eb < num_blocks; ++eb) {
    auto& ms = *mesh_specs[eb];

    // "Use Composite Tet 10" is a per-block material property. A 10-node tet
    // (composite or standard) is projected via the linear Tet<4> down-cast in
    // the projection evaluator (issue #12); the flag is forwarded so it selects
    // the composite path. Independently, the field manager here must integrate
    // on the block's OWN quadrature so the loaded IP states line up with where
    // assembly saved them: a composite block uses the composite rule (via the
    // Tetrahedron<11> topology and COMP12 basis, mirroring assembly), otherwise
    // the standard rule.
    bool const is_ct10 =
        (material_db != Teuchos::null) && ms.ctd.dimension == 3 && ms.ctd.node_count == 10 &&
        material_db->getElementBlockParam<bool>(ms.ebName, "Use Composite Tet 10", false);

    // Cubature and basis for this block's cell topology (volume elements).
    auto intrepid_basis = Albany::getIntrepid2Basis(ms.ctd, is_ct10);

    Teuchos::RCP<shards::CellTopology> const cell_type =
        is_ct10 ? Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Tetrahedron<11>>())) :
                  Teuchos::rcp(new shards::CellTopology(&ms.ctd));

    Intrepid2::DefaultCubatureFactory cub_factory;

    Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature =
        cub_factory.create<PHX::Device, RealType, RealType>(*cell_type, ms.cubatureDegree);

    int const num_nodes    = intrepid_basis->getCardinality();
    int const num_pts      = cubature->getNumPoints();
    int const num_dims     = cubature->getDimension();
    int const workset_size = ms.worksetSize;

    Teuchos::RCP<Layouts> dl = Teuchos::rcp(new Layouts(workset_size, num_nodes, num_nodes, num_pts, num_dims));

    EvaluatorUtils<EvalT, Traits> eval_utils(dl);

    auto fm = Teuchos::rcp(new PHX::FieldManager<Traits>);

    // Geometry -> basis functions (BF, wBF, weighted measure) and the
    // quadrature-point physical coordinates ("Coord Vec" at QP), which the
    // projection evaluator consumes.
    fm->registerEvaluator<EvalT>(eval_utils.constructGatherCoordinateVectorEvaluator());
    fm->registerEvaluator<EvalT>(eval_utils.constructMapToPhysicalFrameEvaluator(cell_type, cubature));
    fm->registerEvaluator<EvalT>(eval_utils.constructComputeBasisFunctionsEvaluator(cell_type, intrepid_basis, cubature));

    // Load each saved QP state into a Phalanx field of the same name. This is
    // the key difference from the response path: the IP field comes from the
    // stored state, NOT a fresh constitutive evaluation.
    for (auto const& f : fields) {
      auto p = Teuchos::rcp(new Teuchos::ParameterList("Load " + f.name));
      p->set<std::string>("Field Name", f.name);
      p->set<std::string>("State Name", f.name);
      p->set<Teuchos::RCP<PHX::DataLayout>>("State Field Layout", qp_layout_for(dl, f.layout));
      fm->registerEvaluator<EvalT>(Teuchos::rcp(new PHAL::LoadStateFieldBase<EvalT, Traits, typename EvalT::ScalarT>(*p)));
    }

    // The projection itself (L2 "Full" or "Lumped"), reusing the existing
    // evaluator. It writes the proj_nodal_<name> nodal states.
    //
    // ProjectIPtoNodalField reads its field configuration from a nested
    // "Parameter List" (mirroring the response path, where this is the deck's
    // ResponseParams sublist). It holds a raw pointer to that list and reads it
    // again during postRegistrationSetup, so the list must outlive the field
    // manager -- hence it is stored in param_lists_.
    auto config = Teuchos::rcp(new Teuchos::ParameterList("Project IP to Nodal Field"));
    config->set<std::string>("Name", "Project IP to Nodal Field");
    config->set<int>("Number of Fields", static_cast<int>(fields.size()));
    for (std::size_t i = 0; i < fields.size(); ++i) {
      config->set<std::string>(Albany::strint("IP Field Name", i), fields[i].name);
      config->set<std::string>(Albany::strint("IP Field Layout", i), fields[i].layout);
    }
    config->set<std::string>("Mass Matrix Type", mass_matrix_type);
    config->set<bool>("Output to File", output_to_exodus);
    // Reuse the projection manager the response path already created (it carries
    // the correct nodal-database offset); do not add a worker, so the manager's
    // worker count keeps matching the evaluate sweeps this projector drives.
    config->set<bool>("Skip Worker Registration", true);
    param_lists_.push_back(config);

    // Forward "Use Composite Tet 10" so the projection evaluator can fail loudly
    // on composite blocks instead of emitting a silently wrong nodal field
    // (mirrors the response path, which passes this through "Parameters From
    // Problem"). Kept alive in param_lists_ for the field manager's lifetime.
    auto pfp = Teuchos::rcp(new Teuchos::ParameterList("Parameters From Problem"));
    pfp->set<bool>("Use Composite Tet 10", is_ct10);
    param_lists_.push_back(pfp);

    auto pp = Teuchos::rcp(new Teuchos::ParameterList("Project IP to Nodal Field"));
    pp->set<Teuchos::ParameterList*>("Parameter List", config.get());
    pp->set<Teuchos::RCP<Teuchos::ParameterList>>("Parameters From Problem", pfp);
    pp->set<std::string>("BF Name", "BF");
    pp->set<std::string>("Weighted BF Name", "wBF");
    pp->set<std::string>("Coordinate Vector Name", "Coord Vec");
    pp->set<Albany::StateManager*>("State Manager Ptr", &state_manager);
    pp->set<Teuchos::RCP<PHX::DataLayout>>("Dummy Data Layout", dl->dummy);

    auto proj = Teuchos::rcp(new LCM::ProjectIPtoNodalField<EvalT, Traits>(*pp, dl, &ms));
    fm->registerEvaluator<EvalT>(proj);
    fm->requireField<EvalT>(*proj->evaluatedFields()[0]);

    // Post-registration setup (Residual-only; the projector is built after the
    // discretization exists, so this is valid here).
    std::string const eval_name = PHAL::evalName<EvalT>("NFP", eb);
    phx_setup->insert_eval(eval_name);
    fm->postRegistrationSetupForType<EvalT>(*phx_setup);
    phx_setup->check_fields(fm->getFieldTagsForSizing<EvalT>());
    phx_setup->update_fields();

    fms_[eb]        = fm;
    eval_names_[eb] = eval_name;
  }
}

void
NodalFieldProjector::project(double const time) const
{
  using EvalT = PHAL::AlbanyTraits::Residual;

  auto        disc          = app_->getDiscretization();
  auto const& ws_phys_index = disc->getWsPhysIndex();
  int const   num_worksets  = app_->getNumWorksets();

  PHAL::Workset workset;
  app_->loadBasicWorksetInfo(workset, time);

  for (std::size_t eb = 0; eb < fms_.size(); ++eb) {
    auto& fm = *fms_[eb];
    fm.preEvaluate<EvalT>(workset);
    for (int ws = 0; ws < num_worksets; ++ws) {
      if (static_cast<std::size_t>(ws_phys_index[ws]) != eb) continue;
      app_->loadWorksetBucketInfo<EvalT>(workset, ws, eval_names_[eb]);
      fm.evaluateFields<EvalT>(workset);
    }
    fm.postEvaluate<EvalT>(workset);
  }
}

}  // namespace Albany
