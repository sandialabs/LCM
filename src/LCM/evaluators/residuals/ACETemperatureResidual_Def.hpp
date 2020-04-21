// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_Macros.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Utilities.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM {
template <typename EvalT, typename Traits>
ACETemperatureResidual<EvalT, Traits>::ACETemperatureResidual(
    Teuchos::ParameterList const&        p,
    Teuchos::RCP<Albany::Layouts> const& dl)
    : wbf_(  // dependent
          p.get<std::string>("Weighted BF Name"),
          dl->node_qp_scalar),
      wgradbf_(  // dependent
          p.get<std::string>("Weighted Gradient BF Name"),
          dl->node_qp_vector),
      tdot_(  // dependent
          p.get<std::string>("ACE Temperature Dot Name"),
          dl->qp_scalar),
      tgrad_(  // dependent
          p.get<std::string>("ACE Temperature Gradient Name"),
          dl->qp_vector),
      thermal_conductivity_(  // dependent
          p.get<std::string>("ACE Thermal Conductivity Name"),
          dl->qp_scalar),
      thermal_inertia_(  // dependent
          p.get<std::string>("ACE Thermal Inertia Name"),
          dl->qp_scalar),
      residual_(  // evaluated
          p.get<std::string>("ACE Residual Name"),
          dl->node_scalar),
      scale_residual_factor(p.get<double>("ACE Residual Scale Factor"))
{
  // List dependent fields
  this->addDependentField(wbf_);
  this->addDependentField(wgradbf_);
  this->addDependentField(tdot_);
  this->addDependentField(tgrad_);

  this->addDependentField(thermal_conductivity_);
  this->addDependentField(thermal_inertia_);

  // List evaluated field
  this->addEvaluatedField(residual_);

  Teuchos::RCP<PHX::DataLayout> vector_dl = dl->node_qp_vector;

  std::vector<PHX::DataLayout::size_type> dims;

  vector_dl->dimensions(dims);
  workset_size_ = dims[0];
  num_nodes_    = dims[1];
  num_qp_       = dims[2];
  num_dims_     = dims[3];

  this->setName("ACE Temperature Residual" + PHX::print<EvalT>());
}

template <typename EvalT, typename Traits>
void
ACETemperatureResidual<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  // List all fields
  this->utils.setFieldData(wbf_, fm);
  this->utils.setFieldData(wgradbf_, fm);
  this->utils.setFieldData(tdot_, fm);
  this->utils.setFieldData(tgrad_, fm);

  this->utils.setFieldData(thermal_conductivity_, fm);
  this->utils.setFieldData(thermal_inertia_, fm);

  this->utils.setFieldData(residual_, fm);
}

template <typename EvalT, typename Traits>
void ACETemperatureResidual<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData)
{
  for (std::size_t cell = 0; cell < workset_size_; ++cell) {
    for (std::size_t node = 0; node < num_nodes_; ++node) {
      residual_(cell, node) = 0.0;
      for (std::size_t qp = 0; qp < num_qp_; ++qp) {
        // Time-derivative contribution to residual
        residual_(cell, node) +=
            thermal_inertia_(cell, qp) * tdot_(cell, qp) * wbf_(cell, node, qp);
        // Diffusion part of residual
        for (std::size_t i = 0; i < num_dims_; ++i) {
          residual_(cell, node) += thermal_conductivity_(cell, qp) *
                                   tgrad_(cell, qp, i) *
                                   wgradbf_(cell, node, qp, i);
        }
      }
      residual_(cell, node) *= scale_residual_factor;
    }
  }
}

}  // namespace LCM
