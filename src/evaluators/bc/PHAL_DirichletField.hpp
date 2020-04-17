// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_DIRICHLET_FIELD_HPP
#define PHAL_DIRICHLET_FIELD_HPP

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include "PHAL_IdentityCoordinateFunctionTraits.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_ParameterList.hpp"

namespace PHAL {
/** \brief BC Dirichlet evaluator for general coordinate functions

*/

// **************************************************************
// **************************************************************
// * Specialization of the DirichletBase class
// **************************************************************
// **************************************************************

template <typename EvalT, typename Traits>
class DirichletField;

template <typename EvalT, typename Traits>
class DirichletField_Base : public PHAL::DirichletBase<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;
  DirichletField_Base(Teuchos::ParameterList& p);

 protected:
  std::string field_name;
};

// **************************************************************
// Residual
// **************************************************************
template <typename Traits>
class DirichletField<PHAL::AlbanyTraits::Residual, Traits>
    : public DirichletField_Base<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  DirichletField(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template <typename Traits>
class DirichletField<PHAL::AlbanyTraits::Jacobian, Traits>
    : public DirichletField_Base<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  DirichletField(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

}  // namespace PHAL

#endif
