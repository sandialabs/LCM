// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#ifndef PHAL_FACTORY_TRAITS_HPP
#define PHAL_FACTORY_TRAITS_HPP

#include "Albany_config.h"
#include "LCM/evaluators/Time.hpp"
#include "LCM/evaluators/bc/ACEWavePressureBC.hpp"
#include "LCM/evaluators/bc/EquilibriumConcentrationBC.hpp"
#include "LCM/evaluators/bc/KfieldBC.hpp"
#include "LCM/evaluators/bc/TimeTracBC.hpp"
#include "LCM/evaluators/bc/TorsionBC.hpp"
#include "PHAL_Dirichlet.hpp"
#include "PHAL_ExprEvalSDBC.hpp"
#include "PHAL_GatherCoordinateVector.hpp"
#include "PHAL_GatherScalarNodalParameter.hpp"
#include "PHAL_GatherSolution.hpp"
#include "PHAL_LoadStateField.hpp"
#include "PHAL_Neumann.hpp"
#include "PHAL_SDirichlet.hpp"
#include "PHAL_TimeDepDBC.hpp"
#include "PHAL_TimeDepSDBC.hpp"
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

template <typename Traits>
struct DirichletFactoryTraits
{
  static int const id_dirichlet            = 0;
  static int const id_dirichlet_aggregator = 1;
  static int const id_timedep_bc           = 2;
  static int const id_timedep_sdbc         = 3;
  static int const id_sdbc                 = 4;
  static int const id_expreval_sdbc        = 5;
  static int const id_kfield_bc            = 6;
  static int const id_eq_concentration_bc  = 7;
  static int const id_time                 = 8;
  static int const id_torsion_bc           = 9;

  // Schwarz and StrongSchwarz BC evaluators were retired: the DBC
  // DOF-elimination path in Application::eliminateConstrainedDOFs and
  // Application::injectConstrainedDOFValues drives DTK transfer and value
  // injection directly without going through a Phalanx evaluator.
  typedef Sacado::mpl::vector<
      PHAL::Dirichlet<_, Traits>,                  //  0
      PHAL::DirichletAggregator<_, Traits>,        //  1
      PHAL::TimeDepDBC<_, Traits>,                 //  2
      PHAL::TimeDepSDBC<_, Traits>,                //  3
      PHAL::SDirichlet<_, Traits>,                 //  4
      PHAL::ExprEvalSDBC<_, Traits>,               //  5
      LCM::KfieldBC<_, Traits>,                    //  6
      LCM::EquilibriumConcentrationBC<_, Traits>,  //  7
      LCM::Time<_, Traits>,                        //  8
      LCM::TorsionBC<_, Traits>                    //  9
      >
      EvaluatorTypes;
};

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
