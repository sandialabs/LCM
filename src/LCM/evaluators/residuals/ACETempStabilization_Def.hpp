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
    : 
      tdot_(
          p.get<std::string>("QP Time Derivative Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      wgradbf_(
          p.get<std::string>("Weighted Gradient BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout")),
      tgrad_(
          p.get<std::string>("Gradient QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout")),
      stab_(p.get<std::string>("Stabilization Name"), 
            p.get<Teuchos::RCP<PHX::DataLayout>>("Node Scalar Data Layout")),
      thermal_conductivity_(
          p.get<std::string>("ACE Thermal Conductivity QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      thermal_inertia_(
          p.get<std::string>("ACE Thermal Inertia QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  this->addDependentField(tdot_);
  this->addDependentField(tgrad_);
  this->addDependentField(wgradbf_);
  this->addDependentField(thermal_conductivity_);
  this->addDependentField(thermal_inertia_);
  this->addEvaluatedField(stab_);

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
ACETempStabilization<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(tgrad_, fm);
  this->utils.setFieldData(wgradbf_, fm);
  this->utils.setFieldData(tdot_, fm);
  this->utils.setFieldData(stab_, fm);
  this->utils.setFieldData(thermal_conductivity_, fm);
  this->utils.setFieldData(thermal_inertia_, fm);
}

//*****
template <typename EvalT, typename Traits>
void
ACETempStabilization<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  //IKT FIXME - FILL IN!
  for (std::size_t cell = 0; cell < workset_size_; ++cell) {
    for (std::size_t node = 0; node < num_nodes_; ++node) {
      stab_(cell, node) = 0.0;
    }
  }
}

//*****
}  // namespace LCM
