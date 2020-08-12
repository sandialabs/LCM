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
ACETempStabilization<EvalT, Traits>::ACETempStabilization(Teuchos::ParameterList const& p)
    : thermal_cond_grad_at_qps_(
          p.get<std::string>("ACE_Thermal_Conductivity Gradient QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout")),
      jacobian_det_(
          p.get<std::string>("Jacobian Det Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      tau_(p.get<std::string>("Tau Name"), p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      fos_(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  this->addDependentField(thermal_cond_grad_at_qps_);
  this->addDependentField(jacobian_det_);
  this->addEvaluatedField(tau_);

  stab_value_                 = p.get<double>("Stabilization Parameter Value");
  std::string tau_type_string = p.get<std::string>("Tau Type");
  if (tau_type_string == "None") {
    tau_type_ = NONE;
  } else if (tau_type_string == "SUPG") {
    tau_type_ = SUPG;
  } else if (tau_type_string == "Proportional to Mesh Size") {
    tau_type_ = PROP_TO_H;
  } else {
    ALBANY_ABORT(
        "Invalid stabilization parameter value!  Valid values are 'None', 'SUPG' and 'Proportional to Mesh Size'.");
  }
  Teuchos::RCP<PHX::DataLayout> vector_dl = p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  workset_size_ = dims[0];
  num_nodes_    = dims[1];
  num_qps_      = dims[2];
  num_dims_     = dims[3];
  this->setName("ACETempStabilization");
}

//*****
template <typename EvalT, typename Traits>
void
ACETempStabilization<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(thermal_cond_grad_at_qps_, fm);
  this->utils.setFieldData(tau_, fm);
  this->utils.setFieldData(jacobian_det_, fm);
}

//*****
template <typename EvalT, typename Traits>
void
ACETempStabilization<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  // Here, we use tau = stab_value_ * pos(h, num_dims_)/2.0/|grad(kappa)|
  // as the stabilization parameter, following a common choice
  // in the literature, where  h = mesh size.
  for (std::size_t cell = 0; cell < workset_size_; ++cell) {
    for (std::size_t qp = 0; qp < num_qps_; ++qp) {
      ScalarT mesh_size      = 2.0 * std::pow(jacobian_det_(cell, qp), 1.0 / num_dims_);
      ScalarT h_pow_num_dims = 1.0;
      for (std::size_t ndim = 0; ndim < num_dims_; ++ndim) {
        h_pow_num_dims *= mesh_size;
      }
      if (tau_type_ == NONE) {
        tau_(cell, qp) = 0.0;
      } else if (tau_type_ == SUPG) {
        ScalarT norm_grad_kappa = 0.0;
        for (std::size_t ndim = 0; ndim < num_dims_; ++ndim) {
          norm_grad_kappa += thermal_cond_grad_at_qps_(cell, qp, ndim) * thermal_cond_grad_at_qps_(cell, qp, ndim);
        }
        if (std::abs(norm_grad_kappa) > 1.0e-12) {
          norm_grad_kappa = std::sqrt(norm_grad_kappa);
        } else {
          norm_grad_kappa = 0.0;
        }
        tau_(cell, qp) = stab_value_ * h_pow_num_dims / 2.0 / norm_grad_kappa;
      } else if (tau_type_ == PROP_TO_H) {
        tau_(cell, qp) = stab_value_ * h_pow_num_dims;
      }
    }
  }
}

//*****
}  // namespace LCM
