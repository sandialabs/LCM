// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_Macros.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Utilities.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM {

//*****
template <typename EvalT, typename Traits>
ACETempStandAloneResid<EvalT, Traits>::ACETempStandAloneResid(Teuchos::ParameterList const& p)
    : wbf_(p.get<std::string>("Weighted BF Name"), p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout")),
      tdot_(
          p.get<std::string>("QP Time Derivative Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      wgradbf_(
          p.get<std::string>("Weighted Gradient BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout")),
      tgrad_(
          p.get<std::string>("Gradient QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout")),
      residual_(p.get<std::string>("Residual Name"), p.get<Teuchos::RCP<PHX::DataLayout>>("Node Scalar Data Layout")),
      thermal_conductivity_(
          p.get<std::string>("ACE Thermal Conductivity QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      thermal_inertia_(
          p.get<std::string>("ACE Thermal Inertia QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      thermal_cond_grad_at_qps_(p.get<std::string>("ACE Thermal Conductivity Gradient QP Variable Name"), 
			        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout")), 
      tau_(p.get<std::string>("Tau Name"), 
           p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      fos_(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  this->addDependentField(wbf_);
  this->addDependentField(tdot_);
  this->addDependentField(tgrad_);
  this->addDependentField(wgradbf_);
  this->addDependentField(thermal_conductivity_);
  this->addDependentField(thermal_inertia_);
  this->addDependentField(thermal_cond_grad_at_qps_);
  this->addDependentField(tau_);
  this->addEvaluatedField(residual_);

  use_stab_ = p.get<bool>("Use Stabilization"); 
  Teuchos::RCP<PHX::DataLayout> vector_dl = p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  workset_size_ = dims[0];
  num_nodes_    = dims[1];
  num_qps_      = dims[2];
  num_dims_     = dims[3];
  this->setName("ACETempStandAloneResid");
}

//*****
template <typename EvalT, typename Traits>
void
ACETempStandAloneResid<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wbf_, fm);
  this->utils.setFieldData(tgrad_, fm);
  this->utils.setFieldData(wgradbf_, fm);
  this->utils.setFieldData(tdot_, fm);
  this->utils.setFieldData(residual_, fm);
  this->utils.setFieldData(thermal_conductivity_, fm);
  this->utils.setFieldData(thermal_inertia_, fm);
  this->utils.setFieldData(tau_, fm);
  this->utils.setFieldData(thermal_cond_grad_at_qps_, fm);
}

//*****
template <typename EvalT, typename Traits>
void
ACETempStandAloneResid<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  // We are solving the following PDE:
  // thermal_inertia_ * dT/dt - thermal_conductivity_ * \nabla T = 0 in 3D
  for (std::size_t cell = 0; cell < workset_size_; ++cell) {
    for (std::size_t node = 0; node < num_nodes_; ++node) {
      residual_(cell, node) = 0.0;
      for (std::size_t qp = 0; qp < num_qps_; ++qp) {
        // Time-derivative contribution to residual
        residual_(cell, node) += thermal_inertia_(cell, qp) * tdot_(cell, qp) * wbf_(cell, node, qp);
        // Diffusion part of residual
        for (std::size_t ndim = 0; ndim < num_dims_; ++ndim) {
          residual_(cell, node) +=
              thermal_conductivity_(cell, qp) * tgrad_(cell, qp, ndim) * wgradbf_(cell, node, qp, ndim);
        }
      }
    }
  }
  if (use_stab_) {
    //Here we use a SUPG-type stabilization which takes the form:
    //stab = -grad(kappa)*grad(w)*tau*(c*dT/dt - (grad(kappa)*grad(T))) 
    //for this problem, where tau is the stabilization parameter.
    for (std::size_t cell = 0; cell < workset_size_; ++cell) {
      for (std::size_t node = 0; node < num_nodes_; ++node) {
        for (std::size_t qp = 0; qp < num_qps_; ++qp) {
          for (std::size_t ndim = 0; ndim < num_dims_; ++ndim) {
            residual_(cell, node) -= thermal_cond_grad_at_qps_(cell, qp, ndim) * wgradbf_(cell, node, qp, ndim)
                                  * tau_(cell, qp) * (thermal_inertia_(cell, qp) * tdot_(cell, qp)  -
                                  thermal_cond_grad_at_qps_(cell, qp, ndim) * tgrad_(cell, qp, ndim));
          }
        }
      }
    }
  }
}

//*****
}  // namespace LCM
