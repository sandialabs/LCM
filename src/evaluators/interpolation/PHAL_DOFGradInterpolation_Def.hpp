// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if defined(ALBANY_TIMER)
#include <chrono>
#endif

#include "Albany_Macros.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_DOFGradInterpolation.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

//*****
template <typename EvalT, typename Traits, typename ScalarT>
DOFGradInterpolationBase<EvalT, Traits, ScalarT>::DOFGradInterpolationBase(
    Teuchos::ParameterList const&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : val_node(p.get<std::string>("Variable Name"), dl->node_scalar),
      GradBF(p.get<std::string>("Gradient BF Name"), dl->node_qp_gradient),
      grad_val_qp(p.get<std::string>("Gradient Variable Name"), dl->qp_gradient)
{
  this->addDependentField(val_node.fieldTag());
  this->addDependentField(GradBF.fieldTag());
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolationBase" + PHX::print<EvalT>());

  // std::vector<PHX::DataLayout::size_type> dims;
  std::vector<PHX::DataLayout::size_type> dims;
  GradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
}

//*****
template <typename EvalT, typename Traits, typename ScalarT>
void
DOFGradInterpolationBase<EvalT, Traits, ScalarT>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node, fm);
  this->utils.setFieldData(GradBF, fm);
  this->utils.setFieldData(grad_val_qp, fm);

  d.fill_field_dependencies(this->dependentFields(), this->evaluatedFields());
}

// *********************************************************************
// Kokkos functor Residual
#if defined(KOKKOS_OPTIMIZED)
template <typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION void
DOFGradInterpolationBase<EvalT, Traits, ScalarT>::operator()(
    const team_member& thread) const
{
  int const thread_idx = thread.league_rank() * threads_per_team;
  int const end_loop   = thread_idx + threads_per_team > (numCells * numQPs) ?
                           (numCells * numQPs) :
                           (thread_idx + threads_per_team);
  ScalarT gradVal_tmp;

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(thread, thread_idx, end_loop), [=](int& indx) {
        int const cell = indx / numCells;
        int const qp = indx    = indx / numCells;
        int const vector_range = numNodes - 1;
        for (int dim = 0; dim < numDims; dim++) {
          grad_val_qp(cell, qp, dim) =
              val_node(cell, 0) * GradBF(cell, 0, qp, dim);

          /* Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(thread,
          vector_range),
                              [&](int const& lk, ScalarT& gradVal){
                          int const node=1+lk;
                          gradVal += val_node(cell, node) * GradBF(cell, node,
          qp, dim);
               }, gradVal_tmp);
           Kokkos::single(Kokkos::PerThread(thread),[&](){
              grad_val_qp(cell,qp,dim)=val_node(cell, 0) * GradBF(cell, 0, qp,
          dim)+gradVal_tmp;
          });
          */
          for (int node = 1; node < numNodes; ++node) {
            grad_val_qp(cell, qp, dim) +=
                val_node(cell, node) * GradBF(cell, node, qp, dim);
          }
        }
      });
}

#else
template <typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION void
DOFGradInterpolationBase<EvalT, Traits, ScalarT>::operator()(
    const DOFGradInterpolationBase_Residual_Tag& tag,
    int const&                                   cell) const
{
  for (int qp = 0; qp < numQPs; ++qp) {
    for (int dim = 0; dim < numDims; dim++) {
      grad_val_qp(cell, qp, dim) = val_node(cell, 0) * GradBF(cell, 0, qp, dim);
      for (int node = 1; node < numNodes; ++node) {
        grad_val_qp(cell, qp, dim) +=
            val_node(cell, node) * GradBF(cell, node, qp, dim);
      }
    }
  }
}
#endif
// ***************************************************************************************
template <typename EvalT, typename Traits, typename ScalarT>
void
DOFGradInterpolationBase<EvalT, Traits, ScalarT>::evaluateFields(
    typename Traits::EvalData workset)
{
  // Intrepid2 Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node,
  // GradBF);

#if defined(ALBANY_TIMER)
  PHX::Device::fence();
  auto start = std::chrono::high_resolution_clock::now();
#endif
  //  Kokkos::deep_copy(grad_val_qp.get_kokkos_view(), 0.0);

#if defined(KOKKOS_OPTIMIZED)

  threads_per_team = work_size;
  numTeams =
      (workset.numCells * numQPs + threads_per_team - 1) / threads_per_team;
  numCells = workset.numCells;

  const team_policy policy(numTeams, 1, 16);

  Kokkos::parallel_for(policy, *this);

#else
  Kokkos::parallel_for(
      DOFGradInterpolationBase_Residual_Policy(0, workset.numCells), *this);
#endif

#if defined(ALBANY_TIMER)
  PHX::Device::fence();
  auto      elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec =
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout << "DOFGradInterpolationBase Residual time = " << millisec << "  "
            << microseconds << std::endl;
#endif
}
// Specialization for Jacobian evaluation taking advantage of the sparsity of
// the derivatives
// *********************************************************************
#ifndef ALBANY_MESH_DEPENDS_ON_SOLUTION

// Kokkos kernel for Jacobian
template <typename Traits>
KOKKOS_INLINE_FUNCTION void
FastSolutionGradInterpolationBase<
    PHAL::AlbanyTraits::Jacobian,
    Traits,
    typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
operator()(
    const FastSolutionGradInterpolationBase_Jacobian_Tag& tag,
    int const&                                            cell) const
{
  for (int qp = 0; qp < this->numQPs; ++qp) {
    for (int dim = 0; dim < this->numDims; dim++) {
      this->grad_val_qp(cell, qp, dim) = ScalarT(
          num_dof,
          this->val_node(cell, 0).val() * this->GradBF(cell, 0, qp, dim));
      (this->grad_val_qp(cell, qp, dim)).fastAccessDx(offset) =
          this->val_node(cell, 0).fastAccessDx(offset) *
          this->GradBF(cell, 0, qp, dim);
      for (int node = 1; node < this->numNodes; ++node) {
        (this->grad_val_qp(cell, qp, dim)).val() +=
            this->val_node(cell, node).val() *
            this->GradBF(cell, node, qp, dim);
        (this->grad_val_qp(cell, qp, dim)).fastAccessDx(neq * node + offset) +=
            this->val_node(cell, node).fastAccessDx(neq * node + offset) *
            this->GradBF(cell, node, qp, dim);
      }
    }
  }
}
//*****

template <typename Traits>
void
FastSolutionGradInterpolationBase<
    PHAL::AlbanyTraits::Jacobian,
    Traits,
    typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
    evaluateFields(typename Traits::EvalData workset)
{
  // Intrepid2 Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node,
  // GradBF);

#if defined(ALBANY_TIMER)
  PHX::Device::fence();
  auto start = std::chrono::high_resolution_clock::now();
#endif

  num_dof = this->val_node(0, 0).size();
  neq     = workset.wsElNodeEqID.extent(2);

  Kokkos::parallel_for(
      FastSolutionGradInterpolationBase_Jacobian_Policy(0, workset.numCells),
      *this);

#if defined(ALBANY_TIMER)
  PHX::Device::fence();
  auto      elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec =
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout << "DOFGradInterpolationBase Jacobian time = " << millisec << "  "
            << microseconds << std::endl;
#endif
}
#endif  // ALBANY_MESH_DEPENDS_ON_SOLUTION

//*****

}  // Namespace PHAL
