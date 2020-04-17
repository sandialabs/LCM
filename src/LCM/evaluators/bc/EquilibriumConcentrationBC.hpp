// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_EquilibriumConcentrationBC_hpp)
#define LCM_EquilibriumConcentrationBC_hpp

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>
#include <Phalanx_config.hpp>
#include <vector>

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
/** \brief Equilibrium Concentration BC Dirichlet evaluator
 */

// Specialization of the DirichletBase class
template <typename EvalT, typename Traits>
class EquilibriumConcentrationBC;

template <typename EvalT, typename Traits>
class EquilibriumConcentrationBC_Base
    : public PHAL::DirichletBase<EvalT, Traits>
{
 public:
  using ScalarT = typename EvalT::ScalarT;
  EquilibriumConcentrationBC_Base(Teuchos::ParameterList& p);
  void
  computeBCs(ScalarT& pressure, ScalarT& Cval);

  RealType applied_conc_, pressure_fac_;

 protected:
  int const coffset_;
  int const poffset_;
};

// Residual
template <typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::Residual, Traits>
    : public EquilibriumConcentrationBC_Base<
          PHAL::AlbanyTraits::Residual,
          Traits>
{
 public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

// Jacobian
template <typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::Jacobian, Traits>
    : public EquilibriumConcentrationBC_Base<
          PHAL::AlbanyTraits::Jacobian,
          Traits>
{
 public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

}  // namespace LCM

#endif
