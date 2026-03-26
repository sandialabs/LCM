// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include "MiniLinearSolver.hpp"
#include "MiniNonlinearSolver.hpp"
#include "MiniSolvers.hpp"

#include <cassert>
#include <iostream>

namespace {

bool verbose = false;

int num_tests  = 0;
int num_passed = 0;

void
test_AlbanyResidual_NewtonBanana()
{
  ++num_tests;
  Teuchos::oblackholestream bhs;
  std::ostream&             os = verbose ? std::cout : bhs;

  using EvalT   = PHAL::AlbanyTraits::Residual;
  using ScalarT = typename EvalT::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  constexpr minitensor::Index DIM{2};

  using MIN  = minitensor::Minimizer<ValueT, DIM>;
  using FN   = LCM::Banana<ValueT>;
  using STEP = minitensor::StepBase<FN, ValueT, DIM>;

  MIN minimizer;

  std::unique_ptr<STEP> pstep = minitensor::stepFactory<FN, ValueT, DIM>(minitensor::StepType::NEWTON);

  assert(pstep->name() != nullptr);

  STEP& step = *pstep;

  FN banana;

  minitensor::Vector<ScalarT, DIM> x;

  x(0) = 0.0;
  x(1) = 3.0;

  LCM::MiniSolver<MIN, STEP, FN, EvalT, DIM> mini_solver(minimizer, step, banana, x);

  minimizer.printReport(os);

  if (minimizer.converged) {
    ++num_passed;
    std::cout << "  PASS: AlbanyResidual_NewtonBanana\n";
  } else {
    std::cerr << "  FAIL: AlbanyResidual_NewtonBanana\n";
  }
}

void
test_AlbanyJacobian_NewtonBanana()
{
  ++num_tests;
  Teuchos::oblackholestream bhs;
  std::ostream&             os = verbose ? std::cout : bhs;

  using EvalT   = PHAL::AlbanyTraits::Jacobian;
  using ScalarT = typename EvalT::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  constexpr minitensor::Index DIM{2};

  using MIN  = minitensor::Minimizer<ValueT, DIM>;
  using FN   = LCM::Banana<ValueT>;
  using STEP = minitensor::NewtonStep<FN, ValueT, DIM>;

  MIN minimizer;

  STEP step;

  FN banana;

  minitensor::Vector<ScalarT, DIM> x;

  x(0) = 0.0;
  x(1) = 3.0;

  LCM::MiniSolver<MIN, STEP, FN, EvalT, DIM> mini_solver(minimizer, step, banana, x);

  minimizer.printReport(os);

  if (minimizer.converged) {
    ++num_passed;
    std::cout << "  PASS: AlbanyJacobian_NewtonBanana\n";
  } else {
    std::cerr << "  FAIL: AlbanyJacobian_NewtonBanana\n";
  }
}

}  // anonymous namespace

int
main(int ac, char* av[])
{
  Kokkos::initialize();

  verbose = (ac > 1);

  std::cout << "Running utMiniSolvers tests...\n";

  test_AlbanyResidual_NewtonBanana();
  test_AlbanyJacobian_NewtonBanana();

  std::cout << num_passed << "/" << num_tests << " tests passed.\n";

  Kokkos::finalize();

  return (num_passed == num_tests) ? 0 : 1;
}
