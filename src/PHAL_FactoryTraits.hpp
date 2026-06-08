// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#ifndef PHAL_FACTORY_TRAITS_HPP
#define PHAL_FACTORY_TRAITS_HPP

#include "Albany_config.h"
#include "LCM/evaluators/bc/ACEWavePressureBC.hpp"
#include "LCM/evaluators/bc/TimeTracBC.hpp"
#include "PHAL_GatherCoordinateVector.hpp"
#include "PHAL_GatherScalarNodalParameter.hpp"
#include "PHAL_GatherSolution.hpp"
#include "PHAL_LoadStateField.hpp"
#include "PHAL_Neumann.hpp"
#include "Sacado_mpl_placeholders.hpp"

// \cond  Have doxygern ignore this namespace
using namespace Sacado::mpl::placeholders;
// \endcond

namespace PHAL {
/*! \brief Struct to define Evaluator objects for the EvaluatorFactory.

    Preconditions:
    - You must provide a Sacado::mpl::vector named EvaluatorTypes that contain
    all Evaluator objects that you wish the factory to build.  Do not confuse
    evaluator types (concrete instances of evaluator objects) with evaluation
    types (types of evaluations to perform, i.e., Residual, Jacobian).

*/

// Only NeumannFactoryTraits is defined here. Dirichlet BCs are handled
// directly by Application::eliminateConstrainedDOFs +
// injectConstrainedDOFValues — no Phalanx factory for the Dirichlet side.

template <typename Traits>
struct NeumannFactoryTraits
{
  static int const id_neumann                    = 0;
  static int const id_neumann_aggregator         = 1;
  static int const id_gather_coord_vector        = 2;
  static int const id_gather_solution            = 3;
  static int const id_load_stateField            = 4;
  static int const id_GatherScalarNodalParameter = 5;
  static int const id_timedep_bc                 = 6;
  static int const id_acetimedep_bc              = 7;

  typedef Sacado::mpl::vector<
      PHAL::Neumann<_, Traits>,                     //  0
      PHAL::NeumannAggregator<_, Traits>,           //  1
      PHAL::GatherCoordinateVector<_, Traits>,      //  2
      PHAL::GatherSolution<_, Traits>,              //  3
      PHAL::LoadStateField<_, Traits>,              //  4
      PHAL::GatherScalarNodalParameter<_, Traits>,  //  5
      LCM::TimeTracBC<_, Traits>,                   //  6
      LCM::ACEWavePressureBC<_, Traits>             //  7
      >
      EvaluatorTypes;
};

}  // namespace PHAL

#endif
