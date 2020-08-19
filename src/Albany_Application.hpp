// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_APPLICATION_HPP
#define ALBANY_APPLICATION_HPP

#include <set>

#include "AAdapt_AdaptiveSolutionManager.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractProblem.hpp"
#include "Albany_AbstractResponseFunction.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_config.h"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Setup.hpp"
#include "PHAL_Workset.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Sacado_ScalarParameterLibrary.hpp"
#include "Sacado_ScalarParameterVector.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"

// Forward declarations.
namespace AAdapt {
namespace rc {
class Manager;
}
}  // namespace AAdapt

namespace Albany {

class Application : public Sacado::ParameterAccessor<PHAL::AlbanyTraits::Residual, SPL_Traits>
{
 public:
  enum SolutionMethod
  {
    Steady,
    Transient,
    TransientTempus,
    Continuation,
    Eigensolve,
    Invalid
  };

  //! Constructor(s) and Destructor
  Application(
      const Teuchos::RCP<Teuchos_Comm const>&     comm,
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      Teuchos::RCP<Thyra_Vector const> const&     initial_guess = Teuchos::null,
      bool const                                  schwarz       = false);

  Application(const Teuchos::RCP<Teuchos_Comm const>& comm);

  Application(const Application&) = delete;

  ~Application() = default;

  //! Prohibit copying/moving
  Application&
  operator=(const Application&) = delete;
  Application&
  operator=(Application&&) = delete;

  void
  initialSetUp(const Teuchos::RCP<Teuchos::ParameterList>& params);
  void
  createMeshSpecs();
  void
  createMeshSpecs(Teuchos::RCP<Albany::AbstractMeshStruct> mesh);
  void
  buildProblem();
  void
  createDiscretization();
  void
  finalSetUp(
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      Teuchos::RCP<Thyra_Vector const> const&     initial_guess = Teuchos::null);

  //! Get underlying abstract discretization
  Teuchos::RCP<Albany::AbstractDiscretization>
  getDiscretization() const;

  //! Get problem object
  Teuchos::RCP<Albany::AbstractProblem>
  getProblem() const;

  //! Get communicator
  Teuchos::RCP<Teuchos_Comm const>
  getComm() const;

  //! Get Thyra DOF vector space
  Teuchos::RCP<Thyra_VectorSpace const>
  getVectorSpace() const;

  //! Create Jacobian operator
  Teuchos::RCP<Thyra_LinearOp>
  createJacobianOp() const;

  //! Get Preconditioner Operator
  Teuchos::RCP<Thyra_LinearOp>
  getPreconditioner();

  bool
  observeResponses() const
  {
    return observe_responses;
  }

  int
  observeResponsesFreq() const
  {
    return response_observ_freq;
  }

  Teuchos::Array<unsigned int>
  getMarkersForRelativeResponses() const
  {
    return relative_responses;
  }

  Teuchos::RCP<AAdapt::AdaptiveSolutionManager>
  getAdaptSolMgr()
  {
    return solMgr;
  }

  //! Get parameter library
  Teuchos::RCP<ParamLib>
  getParamLib() const;

  //! Get distributed parameter library
  Teuchos::RCP<DistributedParameterLibrary>
  getDistributedParameterLibrary() const;

  //! Get solution method
  SolutionMethod
  getSolutionMethod() const
  {
    return solMethod;
  }

  //! Get number of responses
  int
  getNumResponses() const;

  int
  getNumEquations() const
  {
    return neq;
  }
  int
  getSpatialDimension() const
  {
    return spatial_dimension;
  }
  int
  getTangentDerivDimension() const
  {
    return tangent_deriv_dim;
  }

  Teuchos::RCP<Albany::AbstractDiscretization>
  getDisc() const
  {
    return disc;
  }

  //! Get response function
  Teuchos::RCP<AbstractResponseFunction>
  getResponse(int i) const;

  //! Return whether problem wants to use its own preconditioner
  bool
  suppliesPreconditioner() const;

  void
  computeGlobalResidual(
      double const                            current_time,
      Teuchos::RCP<Thyra_Vector const> const& x,
      Teuchos::RCP<Thyra_Vector const> const& x_dot,
      Teuchos::RCP<Thyra_Vector const> const& x_dotdot,
      const Teuchos::Array<ParamVec>&         p,
      Teuchos::RCP<Thyra_Vector> const&       f,
      double const                            dt = 0.0);

 private:
  void
  computeGlobalResidualImpl(
      double const                           current_time,
      Teuchos::RCP<Thyra_Vector const> const x,
      Teuchos::RCP<Thyra_Vector const> const x_dot,
      Teuchos::RCP<Thyra_Vector const> const x_dotdot,
      const Teuchos::Array<ParamVec>&        p,
      Teuchos::RCP<Thyra_Vector> const&      f,
      double const                           dt = 0.0);

  PHAL::Workset
  set_dfm_workset(
      double const                            current_time,
      Teuchos::RCP<Thyra_Vector const> const  x,
      Teuchos::RCP<Thyra_Vector const> const  x_dot,
      Teuchos::RCP<Thyra_Vector const> const  x_dotdot,
      Teuchos::RCP<Thyra_Vector> const&       f,
      Teuchos::RCP<Thyra_Vector const> const& x_post_SDBCs = Teuchos::null);

 public:
  //! Compute global Jacobian
  /*!
   * Set xdot to NULL for steady-state problems
   */
  void
  computeGlobalJacobian(
      double const                            alpha,
      double const                            beta,
      double const                            omega,
      double const                            current_time,
      Teuchos::RCP<Thyra_Vector const> const& x,
      Teuchos::RCP<Thyra_Vector const> const& xdot,
      Teuchos::RCP<Thyra_Vector const> const& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      Teuchos::RCP<Thyra_Vector> const&       f,
      const Teuchos::RCP<Thyra_LinearOp>&     jac,
      double const                            dt = 0.0);

 private:
  void
  computeGlobalJacobianImpl(
      double const                            alpha,
      double const                            beta,
      double const                            omega,
      double const                            current_time,
      Teuchos::RCP<Thyra_Vector const> const& x,
      Teuchos::RCP<Thyra_Vector const> const& xdot,
      Teuchos::RCP<Thyra_Vector const> const& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      Teuchos::RCP<Thyra_Vector> const&       f,
      const Teuchos::RCP<Thyra_LinearOp>&     jac,
      double const                            dt = 0.0);

 public:
  //! Evaluate response functions
  /*!
   * Set xdot to NULL for steady-state problems
   */
  void
  evaluateResponse(
      int                                     response_index,
      double const                            current_time,
      Teuchos::RCP<Thyra_Vector const> const& x,
      Teuchos::RCP<Thyra_Vector const> const& xdot,
      Teuchos::RCP<Thyra_Vector const> const& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      Teuchos::RCP<Thyra_Vector> const&       g);

  //! Provide access to shapeParameters -- no AD
  PHAL::AlbanyTraits::Residual::ScalarT&
  getValue(std::string const& n);

  //! Class to manage state variables (a.k.a. history)
  StateManager&
  getStateMgr()
  {
    return stateMgr;
  }

  //! Evaluate state field manager
  void
  evaluateStateFieldManager(
      double const                     current_time,
      Thyra_Vector const&              x,
      Teuchos::Ptr<Thyra_Vector const> xdot,
      Teuchos::Ptr<Thyra_Vector const> xdotdot);

  void
  evaluateStateFieldManager(double const current_time, const Thyra_MultiVector& x);

  //! Access to number of worksets - needed for working with StateManager
  int
  getNumWorksets()
  {
    return disc->getWsElNodeEqID().size();
  }

  //! Const access to problem parameter list
  Teuchos::RCP<Teuchos::ParameterList const>
  getProblemPL() const
  {
    return problemParams;
  }

  //! Access to problem parameter list
  Teuchos::RCP<Teuchos::ParameterList>
  getProblemPL()
  {
    return problemParams;
  }

  //! Const access to app parameter list
  Teuchos::RCP<Teuchos::ParameterList const>
  getAppPL() const
  {
    return params_;
  }

  //! Access to app parameter list
  Teuchos::RCP<Teuchos::ParameterList>
  getAppPL()
  {
    return params_;
  }

  bool is_adjoint;

 private:
  //! Utility function to set up ShapeParameters through Sacado
  void
  registerShapeParameters();

  void
  defineTimers();

 public:
  //! Routine to get workset (bucket) size info needed by all Evaluation types
  template <typename EvalT>
  void
  loadWorksetBucketInfo(PHAL::Workset& workset, int const& ws, std::string const& evalName);

  void
  loadBasicWorksetInfo(PHAL::Workset& workset, double current_time);

  void
  loadBasicWorksetInfoSDBCs(
      PHAL::Workset&                          workset,
      Teuchos::RCP<Thyra_Vector const> const& owned_sol,
      double const                            current_time);

  void
  loadWorksetJacobianInfo(PHAL::Workset& workset, double const alpha, double const beta, double const omega);

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>
  getEnrichedMeshSpecs() const
  {
    return meshSpecs;
  }

  //! Routine to load common nodeset info into workset
  void
  loadWorksetNodesetInfo(PHAL::Workset& workset);

  //! Routine to load common sideset info into workset
  void
  loadWorksetSidesetInfo(PHAL::Workset& workset, int const ws);

  //! Routines for setting a scaling to be applied to the Jacobian/resdiual
  void
  setScale(Teuchos::RCP<const Thyra_LinearOp> jac = Teuchos::null);
  void
  setScaleBCDofs(PHAL::Workset& workset, Teuchos::RCP<const Thyra_LinearOp> jac = Teuchos::null);

  void
  setupBasicWorksetInfo(
      PHAL::Workset&                          workset,
      double                                  current_time,
      Teuchos::RCP<Thyra_Vector const> const& x,
      Teuchos::RCP<Thyra_Vector const> const& xdot,
      Teuchos::RCP<Thyra_Vector const> const& xdotdot,
      const Teuchos::Array<ParamVec>&         p);

 private:
  template <typename EvalT>
  void
  postRegSetup();

  template <typename EvalT>
  void
  postRegSetupDImpl();

  template <typename EvalT>
  void
  writePhalanxGraph(
      Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>> fm,
      std::string const&                                  evalName,
      int const&                                          phxGraphVisDetail);

 public:
  double
  fixTime(double const current_time) const
  {
    bool const use_time_param =
        (paramLib->isParameter("Time") == true) && (getSchwarzAlternating() == false) && (solMethod != TransientTempus);

    double const this_time =
        use_time_param == true ? paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time") : current_time;

    return this_time;
  }

  void
  setScaling(const Teuchos::RCP<Teuchos::ParameterList>& params);

  // Needed for coupled Schwarz

  void
  setApplications(Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> ca)
  {
    apps_ = ca;
  }

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
  getApplications() const
  {
    return apps_;
  }

  void
  setAppIndex(int const i)
  {
    app_index_ = i;
  }

  int
  getAppIndex() const
  {
    return app_index_;
  }

  void
  setAppNameIndexMap(Teuchos::RCP<std::map<std::string, int>>& anim)
  {
    app_name_index_map_ = anim;
  }

  Teuchos::RCP<std::map<std::string, int>>
  getAppNameIndexMap() const
  {
    return app_name_index_map_;
  }

  void
  setCoupledAppBlockNodeset(
      std::string const& app_name,
      std::string const& block_name,
      std::string const& nodeset_name);

  std::string
  getCoupledBlockName(int const app_index) const
  {
    auto it = coupled_app_index_block_nodeset_names_map_.find(app_index);
    assert(it != coupled_app_index_block_nodeset_names_map_.end());
    return it->second.first;
  }

  std::string
  getNodesetName(int const app_index) const
  {
    auto it = coupled_app_index_block_nodeset_names_map_.find(app_index);
    assert(it != coupled_app_index_block_nodeset_names_map_.end());
    return it->second.second;
  }

  bool
  isCoupled(int const app_index) const
  {
    return coupled_app_index_block_nodeset_names_map_.find(app_index) !=
           coupled_app_index_block_nodeset_names_map_.end();
  }

  // Few coupled applications, so do this by brute force.
  std::string
  getAppName(int app_index = -1) const
  {
    if (app_index == -1) app_index = this->getAppIndex();

    std::string name;

    auto it = app_name_index_map_->begin();

    for (; it != app_name_index_map_->end(); ++it) {
      if (app_index == it->second) {
        name = it->first;
        break;
      }
    }

    assert(it != app_name_index_map_->end());

    return name;
  }

  Teuchos::RCP<Thyra_Vector const> const&
  getX() const
  {
    return x_;
  }

  Teuchos::RCP<Thyra_Vector const> const&
  getXdot() const
  {
    return xdot_;
  }

  Teuchos::RCP<Thyra_Vector const> const&
  getXdotdot() const
  {
    return xdotdot_;
  }

  void
  setX(Teuchos::RCP<Thyra_Vector const> const& x)
  {
    x_ = x;
  }

  void
  setXdot(Teuchos::RCP<Thyra_Vector const> const& xdot)
  {
    xdot_ = xdot;
  }

  void
  setXdotdot(Teuchos::RCP<Thyra_Vector const> const& xdotdot)
  {
    xdotdot_ = xdotdot;
  }

  void
  setSchwarzAlternating(bool const isa)
  {
    is_schwarz_alternating_ = isa;
  }

  bool
  getSchwarzAlternating() const
  {
    return is_schwarz_alternating_;
  }

  Teuchos::RCP<AAdapt::AdaptiveSolutionManager>
  getSolutionManager() const
  {
    return solMgr;
  }

 private:
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> apps_;

  int app_index_{-1};

  Teuchos::RCP<std::map<std::string, int>> app_name_index_map_{Teuchos::null};

  std::map<int, std::pair<std::string, std::string>> coupled_app_index_block_nodeset_names_map_;

  Teuchos::RCP<Thyra_Vector const> x_{Teuchos::null};
  Teuchos::RCP<Thyra_Vector const> xdot_{Teuchos::null};
  Teuchos::RCP<Thyra_Vector const> xdotdot_{Teuchos::null};

  bool is_schwarz_alternating_{false};

 public:
  //! Get Phalanx postRegistration data
  Teuchos::RCP<PHAL::Setup>
  getPhxSetup()
  {
    return phxSetup;
  }

 protected:
  bool is_schwarz_{false};
  bool no_dir_bcs_{false};
  bool requires_sdbcs_{false};
  bool requires_orig_dbcs_{false};

  Teuchos::RCP<Teuchos_Comm const>                         comm{Teuchos::null};
  Teuchos::RCP<Teuchos::FancyOStream>                      out{Teuchos::null};
  Teuchos::RCP<Albany::AbstractDiscretization>             disc{Teuchos::null};
  Teuchos::RCP<Albany::DiscretizationFactory>              discFactory{Teuchos::null};
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> meshSpecs{Teuchos::null};
  Teuchos::RCP<Albany::AbstractProblem>                    problem{Teuchos::null};
  Teuchos::RCP<Teuchos::ParameterList>                     problemParams{Teuchos::null};
  Teuchos::RCP<Teuchos::ParameterList>                     params_{Teuchos::null};
  Teuchos::RCP<ParamLib>                                   paramLib{Teuchos::null};
  Teuchos::RCP<DistributedParameterLibrary>                distParamLib{Teuchos::null};
  Teuchos::RCP<AAdapt::AdaptiveSolutionManager>            solMgr{Teuchos::null};

  // Reference configuration (update) manager
  Teuchos::RCP<AAdapt::rc::Manager> rc_mgr{Teuchos::null};

  // Response functions
  Teuchos::Array<Teuchos::RCP<Albany::AbstractResponseFunction>> responses;

  // Phalanx Field Manager for volumetric fills
  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>> fm;

  // Phalanx Field Manager for Dirichlet Boundary Conditions
  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>> dfm;

  // Phalanx Field Manager for Neumann Boundary Conditions
  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>> nfm;

  // Phalanx Field Manager for states
  Teuchos::Array<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>> sfm;

  // Data for Physics-Based Preconditioners
  bool                                 physicsBasedPreconditioner{false};
  Teuchos::RCP<Teuchos::ParameterList> precParams{Teuchos::null};
  std::string                          precType{""};

  //! Type of solution method
  SolutionMethod solMethod{Invalid};

  //! Integer specifying whether user wants to write Jacobian to MatrixMarket
  //! file
  // writeToMatrixMarketJac = 0: no writing to MatrixMarket (default)
  // writeToMatrixMarketJac =-1: write to MatrixMarket every time a Jacobian
  // arises writeToMatrixMarketJac = N: write N^th Jacobian to MatrixMarket
  // ...and similarly for writeToMatrixMarketRes (integer specifying whether
  // user wants to write residual to MatrixMarket file)
  int writeToMatrixMarketSol{0};
  int writeToMatrixMarketRes{0};
  int writeToMatrixMarketJac{0};
  //! Integer specifying whether user wants to write Jacobian and residual to
  //! Standard output (cout)
  int writeToCoutSol{0};
  int writeToCoutRes{0};
  int writeToCoutJac{0};

  // Value to scale Jacobian/Residual by to possibly improve conditioning
  double scale{0.0};
  double scaleBCdofs{0.0};
  // Scaling types
  enum SCALETYPE
  {
    CONSTANT,
    DIAG,
    ABSROWSUM,
    INVALID
  };
  SCALETYPE scale_type{INVALID};

  //! Shape Optimization data
  bool                     shapeParamsHaveBeenReset{false};
  std::vector<RealType>    shapeParams;
  std::vector<std::string> shapeParamNames;

  unsigned int neq{0}, spatial_dimension{0}, tangent_deriv_dim{0};

  //! Phalanx postRegistration data
  Teuchos::RCP<PHAL::Setup> phxSetup{Teuchos::null};
  mutable int               phxGraphVisDetail{0};
  mutable int               stateGraphVisDetail{0};

  StateManager stateMgr;

  bool morphFromInit{false};
  bool ignore_residual_in_jacobian{false};

  // To prevent a singular mass matrix associated with Dirichlet
  //  conditions, optionally add a small perturbation to the diag
  double perturbBetaForDirichlets{0.0};

  void
  determinePiroSolver(const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams);

  int derivatives_check_{0};
  int num_time_deriv{0};

  // The following are for Jacobian/residual scaling
  Teuchos::Array<Teuchos::Array<int>> offsets_;
  std::vector<std::string>            nodeSetIDs_;
  Teuchos::RCP<Thyra_Vector>          scaleVec_{Teuchos::null};

  // boolean read from input file telling code whether to compute/print
  // responses every step
  bool observe_responses{false};

  // how often one wants the responses to be computed/printed
  int response_observ_freq{0};

  // local responses
  Teuchos::Array<unsigned int> relative_responses;
};

template <typename EvalT>
void
Application::loadWorksetBucketInfo(PHAL::Workset& workset, int const& ws, std::string const& evalName)
{
  auto const& wsElNodeEqID            = disc->getWsElNodeEqID();
  auto const& wsElNodeID              = disc->getWsElNodeID();
  auto const& coords                  = disc->getCoords();
  auto const& wsEBNames               = disc->getWsEBNames();
  auto const& sphereVolume            = disc->getSphereVolume();
  auto const& latticeOrientation      = disc->getLatticeOrientation();
  auto const& cell_boundary_indicator = disc->getCellBoundaryIndicator();
  auto const& face_boundary_indicator = disc->getFaceBoundaryIndicator();
  auto const& edge_boundary_indicator = disc->getEdgeBoundaryIndicator();
  auto const& node_boundary_indicator = disc->getNodeBoundaryIndicator();

  auto const has_cell = cell_boundary_indicator != Teuchos::null && ws < cell_boundary_indicator.size();
  auto const has_face = face_boundary_indicator != Teuchos::null && ws < face_boundary_indicator.size();
  auto const has_edge = edge_boundary_indicator != Teuchos::null && ws < edge_boundary_indicator.size();
  auto const has_node = node_boundary_indicator.size() > 0;

  if (has_cell == true) {
    workset.cell_boundary_indicator = cell_boundary_indicator[ws];
  }
  if (has_face == true) {
    workset.face_boundary_indicator = face_boundary_indicator[ws];
  }
  if (has_edge == true) {
    workset.edge_boundary_indicator = edge_boundary_indicator[ws];
  }
  if (has_node == true) {
    workset.node_boundary_indicator = node_boundary_indicator;
  }

  workset.numCells             = wsElNodeEqID[ws].extent(0);
  workset.wsElNodeEqID         = wsElNodeEqID[ws];
  workset.wsElNodeID           = wsElNodeID[ws];
  workset.wsCoords             = coords[ws];
  workset.wsSphereVolume       = sphereVolume[ws];
  workset.wsLatticeOrientation = latticeOrientation[ws];
  workset.EBName               = wsEBNames[ws];
  workset.wsIndex              = ws;

  workset.local_Vp.resize(workset.numCells);

  workset.savedMDFields = phxSetup->get_saved_fields(evalName);

  // Sidesets are integrated within the Cells
  loadWorksetSidesetInfo(workset, ws);

  workset.stateArrayPtr = &stateMgr.getStateArray(Albany::StateManager::ELEM, ws);
}

}  // namespace Albany

#endif  // ALBANY_APPLICATION_HPP
