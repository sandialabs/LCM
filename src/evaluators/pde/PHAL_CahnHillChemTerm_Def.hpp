// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_Macros.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Phalanx_DataLayout.hpp"

template <typename ScalarT>
inline ScalarT
Sqr(ScalarT const& num)
{
  return num * num;
}

namespace PHAL {

//*****
template <typename EvalT, typename Traits>
CahnHillChemTerm<EvalT, Traits>::CahnHillChemTerm(
    Teuchos::ParameterList const& p)
    : rho(p.get<std::string>("Rho QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      w(p.get<std::string>("W QP Variable Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      chemTerm(
          p.get<std::string>("Chemical Energy Term"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))

{
  b = p.get<double>("b Value");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(rho.fieldTag());
  this->addDependentField(w.fieldTag());

  this->addEvaluatedField(chemTerm);

  this->setName("CahnHillChemTerm");
}

//*****
template <typename EvalT, typename Traits>
void
CahnHillChemTerm<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(rho, fm);
  this->utils.setFieldData(w, fm);

  this->utils.setFieldData(chemTerm, fm);
}

//*****
template <typename EvalT, typename Traits>
void
CahnHillChemTerm<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // Equations 1.1 and 2.2 in Garcke, Rumpf, and Weikard
  // psi(rho) = 0.25 * (rho^2 - b^2)^2

  for (std::size_t cell = 0; cell < workset.numCells; ++cell)
    for (std::size_t qp = 0; qp < numQPs; ++qp)

      // chemTerm(cell, qp) = 0.25 * Sqr(Sqr(rho(cell, qp)) - Sqr(b)) - w(cell,
      // qp);
      chemTerm(cell, qp) =
          (Sqr<ScalarT>(rho(cell, qp)) - Sqr<ScalarT>(b)) * rho(cell, qp) -
          w(cell, qp);
}

template <typename EvalT, typename Traits>
typename CahnHillChemTerm<EvalT, Traits>::ScalarT&
CahnHillChemTerm<EvalT, Traits>::getValue(std::string const& n)
{
  if (n == "b")

    return b;

  else {
    ALBANY_ABORT(
        std::endl
        << "Error! Logic error in getting parameter " << n
        << " in CahnHillChemTerm::getValue()" << std::endl);
    return b;
  }
}

}  // namespace PHAL
