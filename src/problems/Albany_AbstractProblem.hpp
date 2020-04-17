// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_ABSTRACTPROBLEM_HPP
#define ALBANY_ABSTRACTPROBLEM_HPP

#include <string>
#include <vector>

#include "Albany_AbstractFieldContainer.hpp"  //has typedef needed to list the field requirements of the problem
#include "Albany_DataTypes.hpp"
#include "Albany_NullSpaceUtils.hpp"  // has defn of struct that holds null space info for ML
#include "Albany_StateInfoStruct.hpp"  // contains MeshSpecsStuct
#include "Albany_StateManager.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C2_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C2_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C2_FEM.hpp"
#include "Intrepid2_HGRAD_TET_COMP12_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C2_FEM.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "PHAL_Workset.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_VerboseObject.hpp"

namespace Albany {

// Define enumerator to store solution method name
enum SolutionMethodType
{
  Steady,
  Continuation,
  Transient,
  TransientTempus,
  Eigensolve,
  AerasHyperviscosity,
  Unknown
};

// Forward declaration needed for Schwarz coupling.
class Application;

enum FieldManagerChoice
{
  BUILD_RESID_FM,
  BUILD_RESPONSE_FM,
  BUILD_STATE_FM
};

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
 */
class AbstractProblem
{
 public:
  //! Only constructor
  AbstractProblem(
      const Teuchos::RCP<Teuchos::ParameterList>& params_,
      const Teuchos::RCP<ParamLib>&               paramLib_,
      // const Teuchos::RCP<DistributedParameterLibrary>& distParamLib_,
      int const neq_ = 0);

  //! Destructor
  virtual ~AbstractProblem(){};

  //! Get the number of equations
  unsigned int
  numEquations() const;
  std::map<int, std::vector<std::string>> const&
  getSideSetEquations() const;
  void
  setNumEquations(int const neq_);
  unsigned int
  numStates() const;

  // Get the solution method type name
  SolutionMethodType
  getSolutionMethod();

  //! Get spatial dimension
  virtual int
  spatialDimension() const = 0;

  //! Get boolean telling code if SDBCs are utilized
  virtual bool
  useSDBCs() const = 0;

  //! Build the PDE instantiations, boundary conditions, and initial solution
  //! And construct the evaluators and field managers
  virtual void
  buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> meshSpecs,
      StateManager&                                            stateMgr) = 0;

  // Build evaluators
  virtual Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
  buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>&      fm0,
      Albany::MeshSpecsStruct const&              meshSpecs,
      Albany::StateManager&                       stateMgr,
      Albany::FieldManagerChoice                  fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList) = 0;

  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>>
  getFieldManager();
  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
  getDirichletFieldManager();
  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>>
  getNeumannFieldManager();

  Teuchos::Array<Teuchos::Array<int>>
  getOffsets()
  {
    return offsets_;
  }

  std::vector<std::string>
  getNodeSetIDs()
  {
    return nodeSetIDs_;
  }

  //! Return the Null space object used to communicate with MP
  const Teuchos::RCP<Albany::RigidBodyModes>&
  getNullSpace()
  {
    return rigidBodyModes;
  }

  //! Each problem must generate it's list of valide parameters
  virtual Teuchos::RCP<Teuchos::ParameterList const>
  getValidProblemParameters() const
  {
    return getGenericProblemParams("Generic Problem List");
  };

  virtual void getAllocatedStates(
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<
          Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>>
      /* oldState_ */,
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<
          Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>>
      /* newState_ */) const {};

  //! Get a list of the Special fields needed to implement the problem
  const AbstractFieldContainer::FieldContainerRequirements
  getFieldRequirements()
  {
    return requirements;
  }

  const std::
      map<std::string, AbstractFieldContainer::FieldContainerRequirements>&
      getSideSetFieldRequirements() const
  {
    return ss_requirements;
  }

  //! Allow the Problem to modify the solver settings, for example by adding a
  //! custom status test.
  virtual void applyProblemSpecificSolverSettings(
      Teuchos::RCP<Teuchos::ParameterList>)
  {
  }

 protected:
  Teuchos::Array<Teuchos::Array<int>> offsets_;
  std::vector<std::string>            nodeSetIDs_;
  //! List of valid problem params common to all problems, as
  //! a starting point for the specific  getValidProblemParameters
  Teuchos::RCP<Teuchos::ParameterList>
  getGenericProblemParams(std::string listname = "ProblemList") const;

  //! Configurable output stream, defaults to printing on proc=0
  Teuchos::RCP<Teuchos::FancyOStream> out;

  //! Number of equations per node being solved
  unsigned int neq;

  //! Number of time derivatives used in the problem
  //! 0 = x, 1 = xdot, 2 = xdotdot
  int number_of_time_deriv;

  //! Equations that are defined ONLY on some part of the mesh
  std::map<int, std::vector<std::string>> sideSetEquations;

  // Variable use to store the solution method name.
  SolutionMethodType SolutionMethodName;

  //! Problem parameters
  Teuchos::RCP<Teuchos::ParameterList> params;

  //! Parameter library
  Teuchos::RCP<ParamLib> paramLib;

  //! Distributed parameter library
  // Teuchos::RCP<DistributedParameterLibrary> distParamLib;

  //! Field manager for Volumetric Fill
  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>> fm;

  //! Field manager for Dirchlet Conditions Fill
  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>> dfm;

  //! Field manager for Neumann Conditions Fill
  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>> nfm;

  //! Special fields needed to implement the problem
  AbstractFieldContainer::FieldContainerRequirements requirements;
  //! Special fields defined on the side sets needed to implement the problem
  std::map<std::string, AbstractFieldContainer::FieldContainerRequirements>
      ss_requirements;

  //! Null space object used to communicate with MP
  Teuchos::RCP<Albany::RigidBodyModes> rigidBodyModes;

 private:
  //! Private to prohibit default or copy constructor
  AbstractProblem();
  AbstractProblem(const AbstractProblem&);

  //! Private to prohibit copying
  AbstractProblem&
  operator=(const AbstractProblem&);

 public:
  void
  setApplication(Teuchos::RCP<Application> const& a)
  {
    app_ = a;
  }

  Teuchos::RCP<Application>
  getApplication() const
  {
    return app_;
  }

 private:
  Teuchos::RCP<Application> app_;
};

template <typename ProblemType>
struct ConstructEvaluatorsOp
{
  ProblemType&                                                    prob;
  PHX::FieldManager<PHAL::AlbanyTraits>&                          fm;
  Albany::MeshSpecsStruct const&                                  meshSpecs;
  Albany::StateManager&                                           stateMgr;
  Albany::FieldManagerChoice                                      fmchoice;
  Teuchos::RCP<Teuchos::ParameterList>                            responseList;
  Teuchos::RCP<Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>> tags;
  ConstructEvaluatorsOp(
      ProblemType&                                prob_,
      PHX::FieldManager<PHAL::AlbanyTraits>&      fm_,
      Albany::MeshSpecsStruct const&              meshSpecs_,
      Albany::StateManager&                       stateMgr_,
      Albany::FieldManagerChoice                  fmchoice_ = BUILD_RESID_FM,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList_ = Teuchos::null)
      : prob(prob_),
        fm(fm_),
        meshSpecs(meshSpecs_),
        stateMgr(stateMgr_),
        fmchoice(fmchoice_),
        responseList(responseList_)
  {
    tags = Teuchos::rcp(new Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>);
  }
  template <typename T>
  void operator()(T /* x */) const
  {
    tags->push_back(prob.template constructEvaluators<T>(
        fm, meshSpecs, stateMgr, fmchoice, responseList));
  }
};
}  // namespace Albany

#endif  // ALBANY_ABSTRACTPROBLEM_HPP
