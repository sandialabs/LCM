// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

//
// Verification test for Albany::NodalFieldProjector.
//
// A single linear-elastic hex is pulled in uniaxial tension. The exact solution
// is a homogeneous uniaxial-stress state, sigma = diag(E * u / L, 0, 0), which a
// trilinear element reproduces exactly and which both the "Full" (L2) and
// "Lumped" projections recover exactly at every node (projecting a constant
// field returns that constant). The projector reads the *saved* Cauchy_Stress
// quadrature-point state -- it never re-runs the constitutive model -- so as a
// stronger check we also scale the saved state and confirm the projected nodal
// field tracks the scaled state.
//

#include <fstream>

#include "Teuchos_RCP.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_YamlParameterListCoreHelpers.hpp"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>

#include "Albany_Application.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_NodalFieldProjector.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Albany_Utils.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "disc/stk/Albany_AbstractSTKMeshStruct.hpp"

#include "Piro_PerformSolve.hpp"

namespace {

std::string const proj_state = "proj_nodal_Cauchy_Stress";
std::string const ip_state   = "Cauchy_Stress";

std::string const material_filename = "utNFP_material.yaml";

std::string const material_yaml =
    "%YAML 1.1\n---\n"
    "LCM:\n"
    "  ElementBlocks:\n"
    "    Block0:\n"
    "      material: mat\n"
    "  Materials:\n"
    "    mat:\n"
    "      Material Model:\n"
    "        Model Name: Linear Elastic\n"
    "      Elastic Modulus:\n"
    "        Elastic Modulus Type: Constant\n"
    "        Value: 1.0\n"
    "      Poissons Ratio:\n"
    "        Poissons Ratio Type: Constant\n"
    "        Value: 0.3\n"
    "      Output Cauchy Stress: true\n"
    "...\n";

// Mechanics 3D, single STK hex, uniaxial tension in X with symmetry planes on
// the -X/-Y/-Z faces and a prescribed X displacement on +X. The "Project IP to
// Nodal Field" response is declared only so that proj_nodal_Cauchy_Stress is
// registered and allocated before the discretization is built; the projector
// under test reuses that state.
std::string const input_yaml =
    "%YAML 1.1\n---\n"
    "LCM:\n"
    "  Problem:\n"
    "    Name: Mechanics 3D\n"
    "    Solution Method: Steady\n"
    "    MaterialDB Filename: utNFP_material.yaml\n"
    "    Dirichlet BCs:\n"
    "      SDBC on NS NodeSet1 for DOF X: 1.0e-3\n"
    "      SDBC on NS NodeSet0 for DOF X: 0.0\n"
    "      SDBC on NS NodeSet2 for DOF Y: 0.0\n"
    "      SDBC on NS NodeSet5 for DOF Z: 0.0\n"
    "    Parameters:\n"
    "      Number: 1\n"
    "      Parameter 0: Time\n"
    "    Response Functions:\n"
    "      Number: 1\n"
    "      Response 0: Project IP to Nodal Field\n"
    "      ResponseParams 0:\n"
    "        Number of Fields: 1\n"
    "        IP Field Name 0: Cauchy_Stress\n"
    "        IP Field Layout 0: Tensor\n"
    "        Mass Matrix Type: Full\n"
    "        Output to File: true\n"
    "  Discretization:\n"
    "    1D Elements: 1\n"
    "    2D Elements: 1\n"
    "    3D Elements: 1\n"
    "    Method: STK3D\n"
    "    Exodus Output File Name: utNFP.e\n"
    "  Piro:\n"
    "    LOCA:\n"
    "      Bifurcation: { }\n"
    "      Constraints: { }\n"
    "      Predictor:\n"
    "        Method: Constant\n"
    "      Stepper:\n"
    "        Continuation Method: Natural\n"
    "        Continuation Parameter: Time\n"
    "        Initial Value: 0.0\n"
    "        Min Value: 0.0\n"
    "        Max Value: 10.0\n"
    "        Max Steps: 10\n"
    "      Step Size:\n"
    "        Initial Step Size: 10.0\n"
    "        Method: Constant\n"
    "    NOX:\n"
    "      Direction:\n"
    "        Method: Newton\n"
    "        Newton:\n"
    "          Forcing Term Method: Constant\n"
    "          Rescue Bad Newton Solve: true\n"
    "          Stratimikos Linear Solver:\n"
    "            NOX Stratimikos Options: { }\n"
    "            Stratimikos:\n"
    "              Linear Solver Type: Belos\n"
    "              Linear Solver Types:\n"
    "                Belos:\n"
    "                  Solver Type: Block GMRES\n"
    "                  Solver Types:\n"
    "                    Block GMRES:\n"
    "                      Convergence Tolerance: 1.0e-12\n"
    "                      Output Frequency: 0\n"
    "                      Output Style: 0\n"
    "                      Verbosity: 0\n"
    "                      Maximum Iterations: 500\n"
    "                      Block Size: 1\n"
    "                      Num Blocks: 500\n"
    "                      Flexible Gmres: false\n"
    "              Preconditioner Type: Ifpack2\n"
    "              Preconditioner Types:\n"
    "                Ifpack2:\n"
    "                  Overlap: 2\n"
    "                  Prec Type: ILUT\n"
    "                  Ifpack2 Settings:\n"
    "                    'fact: drop tolerance': 0.0\n"
    "                    'fact: ilut level-of-fill': 1.0\n"
    "      Line Search:\n"
    "        Full Step:\n"
    "          Full Step: 1.0\n"
    "        Method: Full Step\n"
    "      Nonlinear Solver: Line Search Based\n"
    "      Printing:\n"
    "        Output Precision: 3\n"
    "        Output Processor: 0\n"
    "        Output Information:\n"
    "          Error: true\n"
    "          Warning: true\n"
    "          Outer Iteration: false\n"
    "          Parameters: false\n"
    "          Details: false\n"
    "          Linear Solver Details: false\n"
    "          Stepper Iteration: false\n"
    "          Stepper Details: false\n"
    "          Stepper Parameters: false\n"
    "      Status Tests:\n"
    "        Test Type: Combo\n"
    "        Combo Type: OR\n"
    "        Number of Tests: 2\n"
    "        Test 0:\n"
    "          Test Type: NormF\n"
    "          Norm Type: Two Norm\n"
    "          Scale Type: Scaled\n"
    "          Tolerance: 1.0e-10\n"
    "        Test 1:\n"
    "          Test Type: MaxIters\n"
    "          Maximum Iterations: 15\n"
    "      Solver Options:\n"
    "        Status Test Check Type: Complete\n"
    "...\n";

// Locate the STK node-vector field holding the projected Cauchy stress.
stk::mesh::Field<double>*
getProjField(Albany::AbstractSTKMeshStruct& ms)
{
  return ms.metaData->get_field<double>(stk::topology::NODE_RANK, proj_state);
}

// Apply f(data, ncomp) at every locally owned node of the projected field.
template <typename F>
void
forEachOwnedNode(Albany::AbstractSTKMeshStruct& ms, stk::mesh::Field<double>* fld, F f)
{
  stk::mesh::Selector              sel     = ms.metaData->locally_owned_part();
  const stk::mesh::BucketVector&   buckets = ms.bulkData->get_buckets(stk::topology::NODE_RANK, sel);
  for (auto const* bptr : buckets) {
    stk::mesh::Bucket const& bucket = *bptr;
    int const                ncomp  = stk::mesh::field_scalars_per_entity(*fld, bucket);
    for (std::size_t i = 0; i < bucket.size(); ++i) {
      double* data = stk::mesh::field_data(*fld, bucket[i]);
      f(data, ncomp);
    }
  }
}

}  // namespace

TEUCHOS_UNIT_TEST(NodalFieldProjector, UniaxialTension)
{
  using Teuchos::RCP;

  // Write the material database the input deck references.
  {
    std::ofstream out(material_filename.c_str());
    out << material_yaml;
  }

  RCP<Teuchos_Comm const>          comm  = Albany::getDefaultComm();
  RCP<Teuchos::ParameterList>      input = Teuchos::getParametersFromYamlString(input_yaml);

  // Build and solve the uniaxial-tension problem. The deck declares a
  // "Project IP to Nodal Field" response, which registers the
  // proj_nodal_Cauchy_Stress nodal state and its projection manager during
  // setup; the projector under test reuses both.
  Albany::SolverFactory                                slvrfctry(input, comm);
  RCP<Albany::Application>                              app;
  RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>> const solver = slvrfctry.createAndGetAlbanyApp(app, comm, comm);

  Teuchos::ParameterList& solveParams = slvrfctry.getAnalysisParameters().sublist("Solve", /*mustAlreadyExist=*/false);
  Teuchos::Array<RCP<Thyra_Vector const>>                      responses;
  Teuchos::Array<Teuchos::Array<RCP<const Thyra_MultiVector>>> sensitivities;
  Piro::PerformSolve(*solver, solveParams, responses, sensitivities);

  auto disc = app->getDiscretization();
  auto ms   = Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(disc->getMeshStruct(), /*throw=*/true);

  stk::mesh::Field<double>* fld = getProjField(*ms);
  TEUCHOS_TEST_FOR_EXCEPTION(fld == nullptr, std::runtime_error, "Field '" << proj_state << "' was not registered.");

  auto checkAgainst = [&](double const* expect, double tol, std::string const& label) {
    int n_nodes = 0;
    forEachOwnedNode(*ms, fld, [&](double const* data, int ncomp) {
      TEST_EQUALITY(ncomp, 9);
      for (int c = 0; c < ncomp; ++c) {
        // Absolute comparison: several expected components are exactly zero
        // (uniaxial / off-diagonal), where a relative test is meaningless.
        if (std::abs(data[c] - expect[c]) > tol) {
          out << label << ": node " << n_nodes << " comp " << c << " = " << data[c] << " expected " << expect[c] << "\n";
        }
        TEST_COMPARE(std::abs(data[c] - expect[c]), <=, tol);
      }
      ++n_nodes;
    });
    TEST_COMPARE(n_nodes, >, 0);
  };

  auto zeroField = [&]() {
    forEachOwnedNode(*ms, fld, [](double* data, int ncomp) {
      for (int c = 0; c < ncomp; ++c) data[c] = 0.0;
    });
  };

  // ---- 1. Project the actually-solved uniaxial-tension stress -------------
  // With E=1, nu=0.3, an applied axial strain of 1e-3 on a unit cube and free
  // lateral faces, the exact solution is homogeneous uniaxial stress
  // sigma = diag(E*eps, 0, 0), which a single trilinear hex reproduces exactly.
  // A uniform field is recovered exactly by both the L2 ("Full") and "Lumped"
  // projections, so every node's projected tensor must equal sigma. (Tolerance
  // is loose enough to absorb the off-diagonal/transverse solver round-off but
  // tight enough to catch a wrong or zero projection.)
  double const E = 1.0, eps = 1.0e-3;
  double const sigma_solved[9] = {E * eps, 0, 0, 0, 0, 0, 0, 0, 0};
  double const solve_tol       = 1.0e-7;

  zeroField();
  {
    Albany::NodalFieldProjector projector(app, {{ip_state, "Tensor"}}, "Full");
    projector.project(0.0);
  }
  checkAgainst(sigma_solved, solve_tol, "Full(solved)");

  zeroField();
  {
    Albany::NodalFieldProjector projector(app, {{ip_state, "Tensor"}}, "Lumped");
    projector.project(0.0);
  }
  checkAgainst(sigma_solved, solve_tol, "Lumped(solved)");

  // ---- 2. Project a prescribed full symmetric tensor ---------------------
  // Overwrite the saved Cauchy_Stress quadrature-point state with a full
  // symmetric tensor and confirm the projection tracks it. This proves the
  // projector sources the STORED state (re-running the constitutive model would
  // instead reproduce the solved stress) and exercises every component and the
  // row-major (i,j) -> component mapping. The state is laid out
  // (Cell, QuadPoint, Dim, Dim) row-major, so the trailing nine (i,j) entries
  // repeat per quadrature point. A uniform field projects exactly (tol 1e-10).
  double const sigma_set[9] = {1.0, 4.0, 5.0, 4.0, 2.0, 6.0, 5.0, 6.0, 3.0};
  {
    Albany::StateArrayVec& esa = app->getStateMgr().getStateArrays().elemStateArrays;
    for (std::size_t ws = 0; ws < esa.size(); ++ws) {
      auto& mda = esa[ws][ip_state];
      for (int idx = 0; idx < static_cast<int>(mda.size()); ++idx) mda[idx] = sigma_set[idx % 9];
    }
  }
  zeroField();
  {
    Albany::NodalFieldProjector projector(app, {{ip_state, "Tensor"}}, "Full");
    projector.project(0.0);
  }
  checkAgainst(sigma_set, 1.0e-10, "Full(prescribed)");
}
