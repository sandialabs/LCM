// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef AADAPT_ADAPTIVE_SOLUTION_MANAGER_HPP
#define AADAPT_ADAPTIVE_SOLUTION_MANAGER_HPP

#include "AAdapt_AbstractAdapter.hpp"
#include "AAdapt_InitialCondition.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_StateManager.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Thyra_AdaptiveSolutionManager.hpp"

namespace AAdapt {

namespace rc {
class Manager;
}

class AdaptiveSolutionManager : public Thyra::AdaptiveSolutionManager
{
 public:
  AdaptiveSolutionManager(
      const Teuchos::RCP<Teuchos::ParameterList>& appParams,
      Teuchos::RCP<Thyra_Vector const> const&     initial_guess,
      const Teuchos::RCP<ParamLib>&               param_lib,
      const Albany::StateManager&                 StateMgr,
      const Teuchos::RCP<rc::Manager>&            rc_mgr,
      const Teuchos::RCP<Teuchos_Comm const>&     comm);

  //! Method called by the solver implementation to determine if the mesh needs
  //! adapting
  // A return type of true means that the mesh should be adapted
  virtual bool
  queryAdaptationCriteria()
  {
    return adapter_->queryAdaptationCriteria(iter_);
  }

  //! Method called by solver implementation to actually adapt the mesh
  //! Apply adaptation method to mesh and problem. Returns true if adaptation is
  //! performed successfully.
  virtual bool
  adaptProblem();

  //! Remap "old" solution into new data structures
  virtual void
  projectCurrentSolution();

  Teuchos::RCP<const Thyra_MultiVector>
  getInitialSolution() const
  {
    return current_soln;
  }

  Teuchos::RCP<Thyra_MultiVector>
  getOverlappedSolution()
  {
    return overlapped_soln;
  }
  Teuchos::RCP<const Thyra_MultiVector>
  getOverlappedSolution() const
  {
    return overlapped_soln;
  }

  Teuchos::RCP<Thyra_Vector const>
  updateAndReturnOverlapSolution(Thyra_Vector const& solution /*not overlapped*/);
  Teuchos::RCP<Thyra_Vector const>
  updateAndReturnOverlapSolutionDot(Thyra_Vector const& solution_dot /*not overlapped*/);
  Teuchos::RCP<Thyra_Vector const>
  updateAndReturnOverlapSolutionDotDot(Thyra_Vector const& solution_dotdot /*not overlapped*/);
  Teuchos::RCP<const Thyra_MultiVector>
  updateAndReturnOverlapSolutionMV(const Thyra_MultiVector& solution /*not overlapped*/);

  Teuchos::RCP<Thyra_Vector>
  get_overlapped_f() const
  {
    return overlapped_f;
  }
  Teuchos::RCP<Thyra_LinearOp>
  get_overlapped_jac() const
  {
    return overlapped_jac;
  }

  Teuchos::RCP<const Albany::CombineAndScatterManager>
  get_cas_manager() const
  {
    return cas_manager;
  }

  Teuchos::RCP<Thyra_MultiVector>
  getCurrentSolution()
  {
    return current_soln;
  }

  void
  scatterX(
      Thyra_Vector const&                    x,
      const Teuchos::Ptr<Thyra_Vector const> x_dot,
      const Teuchos::Ptr<Thyra_Vector const> x_dotdot);

  void
  scatterX(const Thyra_MultiVector& soln);

  bool
  isAdaptive()
  {
    return adapter_ != Teuchos::null;
  }

 private:
  Teuchos::RCP<const Albany::CombineAndScatterManager> cas_manager;

  Teuchos::RCP<Thyra_Vector>   overlapped_f;
  Teuchos::RCP<Thyra_LinearOp> overlapped_jac;

  // The solution directly from the discretization class
  Teuchos::RCP<Thyra_MultiVector> current_soln;
  Teuchos::RCP<Thyra_MultiVector> overlapped_soln;

  // Number of time derivative vectors that we need to support
  int const num_time_deriv;

  const Teuchos::RCP<Teuchos::ParameterList>         appParams_;
  const Teuchos::RCP<Albany::AbstractDiscretization> disc_;
  const Teuchos::RCP<ParamLib>&                      paramLib_;
  const Albany::StateManager&                        stateMgr_;
  const Teuchos::RCP<Teuchos_Comm const>             comm_;

  //! Output stream, defaults to printing just Proc 0
  Teuchos::RCP<Teuchos::FancyOStream> out;

  Teuchos::RCP<AbstractAdapter> adapter_;

  void
  buildAdapter(const Teuchos::RCP<rc::Manager>& rc_mgr);

  void
  resizeMeshDataArrays(const Teuchos::RCP<const Albany::AbstractDiscretization>& disc);
};

}  // namespace AAdapt

#endif  // AADAPT_ADAPTIVE_SOLUTION_MANAGER_HPP
