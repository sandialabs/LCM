// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#include "MiniNonlinearSolver.hpp"
#include "MiniSolvers.hpp"
#include "MiniTensor_FunctionSet.h"
#include "ROL_MiniTensor_MiniSolver.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {

bool verbose = false;

int num_tests  = 0;
int num_passed = 0;

void
check(bool passed, char const* name)
{
  ++num_tests;
  if (passed) {
    ++num_passed;
    std::cout << "  PASS: " << name << "\n";
  } else {
    std::cerr << "  FAIL: " << name << "\n";
  }
}

template <typename EvalT>
bool
bananaRosenbrock()
{
  Teuchos::oblackholestream bhs;
  std::ostream&             os = verbose ? std::cout : bhs;

  using ScalarT = typename EvalT::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  constexpr minitensor::Index DIM{2};

  using FN  = LCM::Banana_Traits<EvalT>;
  using MIN = ROL::MiniTensor_Minimizer<ValueT, DIM>;

  ValueT const a = 1.0;
  ValueT const b = 100.0;

  FN fn(a, b);

  MIN minimizer;

  minimizer.verbose = verbose;

  std::string const algoname{"Line Search"};

  Teuchos::ParameterList params;

  params.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type", "Newton-Krylov");

  params.sublist("Status Test").set("Gradient Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Step Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Iteration Limit", 128);

  minitensor::Vector<ScalarT, DIM> x;

  x(0) = 0.0;
  x(1) = 3.0;

  LCM::MiniSolverROL<MIN, FN, EvalT, DIM> mini_solver(minimizer, algoname, params, fn, x);

  minimizer.printReport(os);

  return minimizer.converged;
}

template <typename EvalT>
bool
paraboloid()
{
  Teuchos::oblackholestream bhs;
  std::ostream&             os = verbose ? std::cout : bhs;

  using ScalarT = typename EvalT::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  constexpr minitensor::Index DIM{2};

  using FN  = LCM::Paraboloid_Traits<EvalT>;
  using MIN = ROL::MiniTensor_Minimizer<ValueT, DIM>;

  ValueT const a = 0.0;
  ValueT const b = 0.0;

  FN fn(a, b);

  MIN minimizer;

  minimizer.verbose = verbose;

  std::string const algoname{"Line Search"};

  Teuchos::ParameterList params;

  params.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type", "Newton-Krylov");

  params.sublist("Status Test").set("Gradient Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Step Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Iteration Limit", 128);

  minitensor::Vector<ScalarT, DIM> x;

  x(0) = 10.0 * minitensor::random<ValueT>();
  x(1) = 10.0 * minitensor::random<ValueT>();

  LCM::MiniSolverROL<MIN, FN, EvalT, DIM> mini_solver(minimizer, algoname, params, fn, x);

  minimizer.printReport(os);

  return minimizer.converged;
}

template <typename EvalT>
bool
paraboloidBounds()
{
  Teuchos::oblackholestream bhs;
  std::ostream&             os = verbose ? std::cout : bhs;

  using ScalarT = typename EvalT::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  constexpr minitensor::Index DIM{2};

  using FN  = LCM::Paraboloid_Traits<EvalT>;
  using MIN = ROL::MiniTensor_Minimizer<ValueT, DIM>;
  using BC  = minitensor::Bounds<ValueT, DIM>;

  ValueT const a = 0.0;
  ValueT const b = 0.0;

  FN fn(a, b);

  MIN minimizer;

  minimizer.verbose = verbose;

  minitensor::Vector<ValueT, DIM> lo(1.0, ROL::ROL_NINF<ValueT>());
  minitensor::Vector<ValueT, DIM> hi(10.0, ROL::ROL_INF<ValueT>());

  BC bounds(lo, hi);

  std::string const algoname{"Line Search"};

  Teuchos::ParameterList params;

  params.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type", "Newton-Krylov");

  params.sublist("Status Test").set("Gradient Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Step Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Iteration Limit", 128);

  minitensor::Vector<ScalarT, DIM> x;

  x(0) = 10.0 * minitensor::random<ValueT>();
  x(1) = 10.0 * minitensor::random<ValueT>();

  LCM::MiniSolverBoundsROL<MIN, FN, BC, EvalT, DIM> mini_solver(minimizer, algoname, params, fn, bounds, x);

  minimizer.printReport(os);

  return minimizer.converged;
}

template <typename EvalT>
bool
paraboloidEquality()
{
  Teuchos::oblackholestream bhs;
  std::ostream&             os = verbose ? std::cout : bhs;

  using ScalarT = typename EvalT::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  constexpr minitensor::Index NUM_VAR{2};
  constexpr minitensor::Index NUM_CONSTR{1};

  using FN  = LCM::Paraboloid_Traits<EvalT>;
  using MIN = ROL::MiniTensor_Minimizer<ValueT, NUM_VAR>;
  using EIC = minitensor::Circumference<ValueT, NUM_CONSTR, NUM_VAR>;

  ValueT const a = 2.0;
  ValueT const b = 0.0;
  ValueT const r = 1.0;

  FN fn;

  MIN minimizer;

  minimizer.verbose = verbose;

  EIC eq_constr(r, a, b);

  std::string const algoname{"Composite Step"};

  Teuchos::ParameterList params;

  params.sublist("Step").sublist(algoname).sublist("Optimality System Solver").set("Nominal Relative Tolerance", 1.e-8);
  params.sublist("Step").sublist(algoname).sublist("Optimality System Solver").set("Fix Tolerance", true);
  params.sublist("Step").sublist(algoname).sublist("Tangential Subproblem Solver").set("Iteration Limit", 128);
  params.sublist("Step").sublist(algoname).sublist("Tangential Subproblem Solver").set("Relative Tolerance", 1e-6);
  params.sublist("Step").sublist(algoname).set("Output Level", 0);
  params.sublist("Status Test").set("Gradient Tolerance", 1.0e-12);
  params.sublist("Status Test").set("Constraint Tolerance", 1.0e-12);
  params.sublist("Status Test").set("Step Tolerance", 1.0e-18);
  params.sublist("Status Test").set("Iteration Limit", 128);

  minitensor::Vector<ScalarT, NUM_VAR>    x(minitensor::Filler::ONES);
  minitensor::Vector<ScalarT, NUM_CONSTR> c(minitensor::Filler::ZEROS);

  LCM::MiniSolverEqIneqROL<MIN, FN, EIC, EvalT, NUM_VAR, NUM_CONSTR> mini_solver(minimizer, algoname, params, fn, eq_constr, x, c);

  minimizer.printReport(os);

  return minimizer.converged;
}

bool
paraboloidEqualityConstraintPlain()
{
  Teuchos::oblackholestream bhs;
  std::ostream&             os = verbose ? std::cout : bhs;

  constexpr minitensor::Index NUM_VAR{2};
  constexpr minitensor::Index NUM_CONSTR{1};

  double const a = 2.0;
  double const b = 0.0;
  double const r = 1.0;

  minitensor::Paraboloid<double, NUM_VAR> fn;

  minitensor::Circumference<double, NUM_CONSTR, NUM_VAR> eq_constr(r, a, b);

  std::string const algoname{"Composite Step"};

  Teuchos::ParameterList params;

  params.sublist("Step").sublist(algoname).sublist("Optimality System Solver").set("Nominal Relative Tolerance", 1.e-8);
  params.sublist("Step").sublist(algoname).sublist("Optimality System Solver").set("Fix Tolerance", true);
  params.sublist("Step").sublist(algoname).sublist("Tangential Subproblem Solver").set("Iteration Limit", 128);
  params.sublist("Step").sublist(algoname).sublist("Tangential Subproblem Solver").set("Relative Tolerance", 1e-6);
  params.sublist("Step").sublist(algoname).set("Output Level", 0);
  params.sublist("Status Test").set("Gradient Tolerance", 1.0e-12);
  params.sublist("Status Test").set("Constraint Tolerance", 1.0e-12);
  params.sublist("Status Test").set("Step Tolerance", 1.0e-18);
  params.sublist("Status Test").set("Iteration Limit", 128);

  minitensor::Vector<double, NUM_VAR>    x(minitensor::Filler::ONES);
  minitensor::Vector<double, NUM_CONSTR> c(minitensor::Filler::ZEROS);

  ROL::MiniTensor_Minimizer<double, NUM_VAR> minimizer;

  minimizer.verbose = verbose;

  minimizer.solve(algoname, params, fn, eq_constr, x, c);

  minimizer.printReport(os);

  double const                     tol{1.0e-14};
  minitensor::Vector<double, NUM_VAR> soln(1.0, 0.0);
  double const                     error = minitensor::norm(soln - x);

  return error <= tol;
}

}  // anonymous namespace

int
main(int ac, char* av[])
{
  Kokkos::initialize();

  verbose = (ac > 1);

  std::cout << "Running utMiniSolversROL tests...\n";

  check(bananaRosenbrock<PHAL::AlbanyTraits::Residual>(), "Rosenbrock_AlbanyResidualROL");
  check(bananaRosenbrock<PHAL::AlbanyTraits::Jacobian>(), "Rosenbrock_AlbanyJacobianROL");
  check(paraboloid<PHAL::AlbanyTraits::Residual>(), "Paraboloid_PlainROLResidual");
  check(paraboloid<PHAL::AlbanyTraits::Jacobian>(), "Paraboloid_PlainROLJacobian");
  check(paraboloidBounds<PHAL::AlbanyTraits::Residual>(), "Paraboloid_BoundsROLResidual");
  check(paraboloidBounds<PHAL::AlbanyTraits::Jacobian>(), "Paraboloid_BoundsROLJacobian");
  check(paraboloidEquality<PHAL::AlbanyTraits::Residual>(), "Paraboloid_EqualityROLResidual");
  check(paraboloidEquality<PHAL::AlbanyTraits::Jacobian>(), "Paraboloid_EqualityROLJacobian");
  check(paraboloidEqualityConstraintPlain(), "Paraboloid_EqualityConstraintPlain");

  std::cout << num_passed << "/" << num_tests << " tests passed.\n";

  Kokkos::finalize();

  return (num_passed == num_tests) ? 0 : 1;
}
