// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_Macros.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

//*****
template <typename EvalT, typename Traits>
LangevinNoiseTerm<EvalT, Traits>::LangevinNoiseTerm(
    Teuchos::ParameterList const& p)
    : rho(p.get<std::string>("Rho QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      noiseTerm(
          p.get<std::string>("Langevin Noise Term"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      duration(2),
      rng(seedgen())  // seed the rng

{
  using ScalarT     = typename EvalT::ScalarT;
  sd                = p.get<double>("SD Value");
  auto const sd_val = Sacado::Value<ScalarT>::eval(sd);
  duration          = p.get<Teuchos::Array<int>>("Langevin Noise Time Period");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  nd      = Teuchos::rcp(new boost::normal_distribution<double>(0.0, sd_val));
  var_nor = Teuchos::rcp(new boost::variate_generator<
                         boost::mt19937&,
                         boost::normal_distribution<double>>(rng, *nd));
  this->addDependentField(rho.fieldTag());
  this->addEvaluatedField(noiseTerm);
  this->setName("LangevinNoiseTerm");
}

//*****
template <typename EvalT, typename Traits>
void
LangevinNoiseTerm<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(rho, fm);

  this->utils.setFieldData(noiseTerm, fm);
}

//*****
template <typename EvalT, typename Traits>
void
LangevinNoiseTerm<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  if (duration[0] < 0 || (workset.current_time >= duration[0] &&
                          workset.current_time < duration[1])) {
    // Standard deviation as sd about a mean of zero. Perturb the rho solution
    // by this.

    for (std::size_t cell = 0; cell < workset.numCells; ++cell)
      for (std::size_t qp = 0; qp < numQPs; ++qp)

        noiseTerm(cell, qp) = rho(cell, qp) + (*var_nor)();
    //        noiseTerm(cell, qp) = rho(cell, qp) + 0.1;
    //        noiseTerm(cell, qp) = 0.0;
  }
}

template <typename EvalT, typename Traits>
typename LangevinNoiseTerm<EvalT, Traits>::ScalarT&
LangevinNoiseTerm<EvalT, Traits>::getValue(std::string const& n)
{
  if (n == "sd")

    return sd;

  else {
    ALBANY_ABORT(
        std::endl
        << "Error! Logic error in getting parameter " << n
        << " in LangevinNoiseTerm::getValue()" << std::endl);
    return sd;
  }
}

// Private convenience function
template <typename EvalT, typename Traits>
long
LangevinNoiseTerm<EvalT, Traits>::seedgen()
{
  long seconds, s, seed, pid;

  pid = getpid();
  s   = time(&seconds); /* get CPU seconds since 01/01/1970 */

  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}

}  // namespace PHAL
