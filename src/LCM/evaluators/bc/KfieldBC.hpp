// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef KFIELDBC_HPP
#define KFIELDBC_HPP

#include <vector>

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
/** \brief KfieldBC Dirichlet evaluator

*/

// **************************************************************
// **************************************************************
// * Specialization of the DirichletBase class
// **************************************************************
// **************************************************************

template <typename EvalT, typename Traits>
class KfieldBC;

template <typename EvalT, typename Traits>
class KfieldBC_Base : public PHAL::DirichletBase<EvalT, Traits>
{
 public:
  using ScalarT = typename EvalT::ScalarT;
  KfieldBC_Base(Teuchos::ParameterList& p);
  ScalarT&
  getValue(std::string const& n);
  void
  computeBCs(double* coord, ScalarT& Xval, ScalarT& Yval, RealType time);

  RealType    mu, nu, KIval, KIIval;
  ScalarT     KI, KII;
  std::string KI_name, KII_name;

 protected:
  int const             offset;
  std::vector<RealType> timeValues;
  std::vector<RealType> KIValues;
  std::vector<RealType> KIIValues;
};

// **************************************************************
// Residual
// **************************************************************
template <typename Traits>
class KfieldBC<PHAL::AlbanyTraits::Residual, Traits>
    : public KfieldBC_Base<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template <typename Traits>
class KfieldBC<PHAL::AlbanyTraits::Jacobian, Traits>
    : public KfieldBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

}  // namespace LCM

#endif
