// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#ifndef PHAL_FACTORY_TRAITS_HPP
#define PHAL_FACTORY_TRAITS_HPP

#include "Albany_config.h"
#include "LCM/evaluators/Time.hpp"
#include "LCM/evaluators/bc/EquilibriumConcentrationBC.hpp"
#include "LCM/evaluators/bc/KfieldBC.hpp"
#include "LCM/evaluators/bc/SchwarzBC.hpp"
#include "LCM/evaluators/bc/StrongSchwarzBC.hpp"
#include "LCM/evaluators/bc/TimeTracBC.hpp"
#include "LCM/evaluators/bc/TorsionBC.hpp"
#include "PHAL_Dirichlet.hpp"
#include "PHAL_DirichletCoordinateFunction.hpp"
#include "PHAL_DirichletField.hpp"
#include "PHAL_DirichletOffNodeSet.hpp"
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
  static int const id_dirichlet                     = 0;
  static int const id_dirichlet_aggregator          = 1;
  static int const id_dirichlet_coordinate_function = 2;
  static int const id_dirichlet_field               = 3;
  static int const id_dirichlet_off_nodeset         = 4;  // eqs on side set
  static int const id_timedep_bc                    = 5;
  static int const id_timedep_sdbc                  = 6;
  static int const id_sdbc                          = 7;
  static int const id_expreval_sdbc                 = 8;
  static int const id_kfield_bc                     = 9;
  static int const id_eq_concentration_bc           = 10;
  static int const id_time                          = 11;
  static int const id_torsion_bc                    = 12;
  static int const id_schwarz_bc                    = 13;
  static int const id_strong_schwarz_bc             = 14;

  typedef Sacado::mpl::vector<
      PHAL::Dirichlet<_, Traits>,                  //  0
      PHAL::DirichletAggregator<_, Traits>,        //  1
      PHAL::DirichletCoordFunction<_, Traits>,     //  2
      PHAL::DirichletField<_, Traits>,             //  3
      PHAL::DirichletOffNodeSet<_, Traits>,        //  4
      PHAL::TimeDepDBC<_, Traits>,                 //  5
      PHAL::TimeDepSDBC<_, Traits>,                //  6
      PHAL::SDirichlet<_, Traits>,                 //  7
      PHAL::ExprEvalSDBC<_, Traits>,               //  8
      LCM::KfieldBC<_, Traits>,                    //  9
      LCM::EquilibriumConcentrationBC<_, Traits>,  // 10
      LCM::Time<_, Traits>,                        // 11
      LCM::TorsionBC<_, Traits>,                   // 12
      LCM::SchwarzBC<_, Traits>,                   // 13
      LCM::StrongSchwarzBC<_, Traits>              // 14
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

  typedef Sacado::mpl::vector<
      PHAL::Neumann<_, Traits>,                     //  0
      PHAL::NeumannAggregator<_, Traits>,           //  1
      PHAL::GatherCoordinateVector<_, Traits>,      //  2
      PHAL::GatherSolution<_, Traits>,              //  3
      PHAL::LoadStateField<_, Traits>,              //  4
      PHAL::GatherScalarNodalParameter<_, Traits>,  //  5
      LCM::TimeTracBC<_, Traits>                    //  6
      >
      EvaluatorTypes;
};

}  // namespace PHAL

#endif
