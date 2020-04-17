// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_LANGEVINNOISETERM_HPP
#define PHAL_LANGEVINNOISETERM_HPP

#include "Albany_Types.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

// Random and Gaussian number distribution
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/
namespace PHAL {

template <typename EvalT, typename Traits>
class LangevinNoiseTerm : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;

  LangevinNoiseTerm(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

  ScalarT&
  getValue(std::string const& n);

 private:
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT const, Cell, QuadPoint> rho;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint> noiseTerm;

  unsigned int numQPs, numDims, numNodes;

  ScalarT             sd;
  Teuchos::Array<int> duration;

  boost::mt19937                                   rng;
  Teuchos::RCP<boost::normal_distribution<double>> nd;
  Teuchos::RCP<boost::variate_generator<
      boost::mt19937&,
      boost::normal_distribution<double>>>
      var_nor;
  //  Teuchos::RCP<boost::normal_distribution<ScalarT> > nd;
  //  Teuchos::RCP<boost::variate_generator<boost::mt19937&,
  //  boost::normal_distribution<ScalarT> > > var_nor;

  // generate seed convenience function
  long
  seedgen();
};
}  // namespace PHAL

#endif
