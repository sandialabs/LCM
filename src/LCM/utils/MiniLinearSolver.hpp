// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_MiniLinearSolver_hpp)
#define LCM_MiniLinearSolver_hpp

#include "MiniTensor_Solvers.h"
#include "PHAL_AlbanyTraits.hpp"

namespace LCM {

///
/// Mini Linear Solver Base class
///
template <typename EvalT, minitensor::Index N = minitensor::DYNAMIC>
class MiniLinearSolver_Base
{
 public:
  using ScalarT = typename EvalT::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  virtual ~MiniLinearSolver_Base() {}

  virtual void
  solve(
      minitensor::Tensor<ScalarT, N> const& A,
      minitensor::Vector<ScalarT, N> const& b,
      minitensor::Vector<ScalarT, N>&       x)
  {
  }
};

// Specializations
template <typename EvalT, minitensor::Index N = minitensor::DYNAMIC>
class MiniLinearSolver;

// Residual
template <minitensor::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::Residual, N>
    : public MiniLinearSolver_Base<PHAL::AlbanyTraits::Residual, N>
{
 public:
  using ScalarT = PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  void
  solve(
      minitensor::Tensor<ScalarT, N> const& A,
      minitensor::Vector<ScalarT, N> const& b,
      minitensor::Vector<ScalarT, N>&       x) override;
};

// Jacobian
template <minitensor::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, N>
    : public MiniLinearSolver_Base<PHAL::AlbanyTraits::Jacobian, N>
{
 public:
  using ScalarT = PHAL::AlbanyTraits::Jacobian::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  void
  solve(
      minitensor::Tensor<ScalarT, N> const& A,
      minitensor::Vector<ScalarT, N> const& b,
      minitensor::Vector<ScalarT, N>&       x) override;
};

}  // namespace LCM

#include "MiniLinearSolver.t.hpp"

#endif  // LCM_MiniLinearSolver_hpp
