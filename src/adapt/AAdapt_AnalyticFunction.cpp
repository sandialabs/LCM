// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "AAdapt_AnalyticFunction.hpp"

#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stk_expreval/Evaluator.hpp>

#include "Albany_Macros.hpp"

double const pi = 3.141592653589793;

// Factory method to build functions based on a string
Teuchos::RCP<AAdapt::AnalyticFunction>
AAdapt::createAnalyticFunction(std::string name, int neq, int numDim, Teuchos::Array<double> data)
{
  Teuchos::RCP<AAdapt::AnalyticFunction> F;

  if (name == "Constant")
    F = Teuchos::rcp(new AAdapt::ConstantFunction(neq, numDim, data));

  else if (name == "Linear Y")
    F = Teuchos::rcp(new AAdapt::LinearY(neq, numDim, data));

  else if (name == "Linear")
    F = Teuchos::rcp(new AAdapt::Linear(neq, numDim, data));

  else if (name == "About Z")
    F = Teuchos::rcp(new AAdapt::AboutZ(neq, numDim, data));

  else if (name == "About Linear Z")
    F = Teuchos::rcp(new AAdapt::AboutLinearZ(neq, numDim, data));

  else if (name == "Gaussian Z")
    F = Teuchos::rcp(new AAdapt::GaussianZ(neq, numDim, data));

  else if (name == "Circle")
    F = Teuchos::rcp(new AAdapt::Circle(neq, numDim, data));

  else if (name == "Sin Scalar")
    F = Teuchos::rcp(new AAdapt::SinScalar(neq, numDim, data));

  else
    ALBANY_PANIC(name != "Valid Initial Condition Function", "Unrecognized initial condition function name: " << name);

  return F;
}

AAdapt::ConstantFunction::ConstantFunction(int neq_, int numDim_, Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (data.size() != neq),
      "Error! Invalid specification of initial condition: incorrect length of "
      "Function Data for Constant Function; neq = "
          << neq << ", data.size() = " << data.size() << std::endl);
}
void
AAdapt::ConstantFunction::compute(double* x, double const* X)
{
  if (data.size() > 0)
    for (int i = 0; i < neq; i++) x[i] = data[i];
}






// Private convenience function
long
AAdapt::seedgen(int worksetID)
{
  long seconds, s, seed, pid;

  pid = getpid();
  s   = time(&seconds); /* get CPU seconds since 01/01/1970 */

  // Use worksetID to give more randomness between calls

  seed = std::abs(((s * 181) * ((pid - 83) * 359) * worksetID) % 104729);
  return seed;
}

AAdapt::ConstantFunctionPerturbed::ConstantFunctionPerturbed(
    int                    neq_,
    int                    numDim_,
    int                    worksetID,
    Teuchos::Array<double> data_,
    Teuchos::Array<double> pert_mag_)
    : numDim(numDim_), neq(neq_), data(data_), pert_mag(pert_mag_)
{
  ALBANY_PANIC(
      (data.size() != neq || pert_mag.size() != neq),
      "Error! Invalid specification of initial condition: incorrect length of "
          << "Function Data for Constant Function Perturbed; neq = " << neq << ", data.size() = " << data.size() << ", pert_mag.size() = " << pert_mag.size()
          << std::endl);

  //  srand( time(NULL) ); // seed the random number gen
  srand(seedgen(worksetID));  // seed the random number gen
}

void
AAdapt::ConstantFunctionPerturbed::compute(double* x, double const* X)
{
  for (int i = 0; i < neq; i++) x[i] = data[i] + udrand(-pert_mag[i], pert_mag[i]);
}

// Private convenience function
double
AAdapt::ConstantFunctionPerturbed::udrand(double lo, double hi)
{
  static double const base    = 1.0 / (RAND_MAX + 1.0);
  double              deviate = std::rand() * base;
  return lo + deviate * (hi - lo);
}

AAdapt::ConstantFunctionGaussianPerturbed::ConstantFunctionGaussianPerturbed(
    int                    neq_,
    int                    numDim_,
    int                    worksetID,
    Teuchos::Array<double> data_,
    Teuchos::Array<double> pert_mag_)
    : numDim(numDim_),
      neq(neq_),
      data(data_),
      pert_mag(pert_mag_),
      rng(seedgen(worksetID)),  // seed the rng
      nd(neq_)
{
  ALBANY_PANIC(
      (data.size() != neq || pert_mag.size() != neq),
      "Error! Invalid specification of initial condition: incorrect length of " << "Function Data for Constant Function Gaussian Perturbed; neq = " << neq
                                                                                << ", data.size() = " << data.size()
                                                                                << ", pert_mag.size() = " << pert_mag.size() << std::endl);

  if (data.size() > 0 && pert_mag.size() > 0)
    for (int i = 0; i < neq; i++)
      if (pert_mag[i] > std::numeric_limits<double>::epsilon()) {
        nd[i] = Teuchos::rcp(new std::normal_distribution<double>(data[i], pert_mag[i]));
      }
}

void
AAdapt::ConstantFunctionGaussianPerturbed::compute(double* x, double const* X)
{
  for (int i = 0; i < neq; i++)
    if (nd[i] != Teuchos::null)
      x[i] = (*nd[i])(rng);

    else
      x[i] = data[i];
}

AAdapt::LinearY::LinearY(int neq_, int numDim_, Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq < 2) || (numDim < 2) || (data.size() != 1), "Error! Invalid call of LinearY with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::LinearY::compute(double* x, double const* X)
{
  x[0] = 0.0;
  x[1] = data[0] * X[0];

  if (numDim > 2) x[2] = 0.0;
}

AAdapt::Linear::Linear(int neq_, int numDim_, Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC((data.size() != neq * numDim), "Error! Invalid call of Linear with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::Linear::compute(double* x, double const* X)
{
  for (auto eq = 0; eq < neq; ++eq) {
    double s{0.0};
    for (auto dim = 0; dim < numDim; ++dim) {
      s += data[eq * numDim + dim] * X[dim];
    }
    x[eq] = s;
  }
}

AAdapt::AboutZ::AboutZ(int neq_, int numDim_, Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq < 2) || (numDim < 2) || (data.size() != 1), "Error! Invalid call of AboutZ with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::AboutZ::compute(double* x, double const* X)
{
  x[0] = -data[0] * X[1];
  x[1] = data[0] * X[0];

  if (neq > 2) x[2] = 0.0;
}

AAdapt::AboutLinearZ::AboutLinearZ(int neq_, int numDim_, Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq < 3) || (numDim < 3) || (data.size() != 1), "Error! Invalid call of AboutLinearZ with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::AboutLinearZ::compute(double* x, double const* X)
{
  x[0] = -data[0] * X[1] * X[2];
  x[1] = data[0] * X[0] * X[2];
  x[2] = 0.0;
}

AAdapt::GaussianZ::GaussianZ(int neq_, int numDim_, Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq < 2) || (numDim < 2) || (data.size() != 3), "Error! Invalid call of GaussianZ with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::GaussianZ::compute(double* x, double const* X)
{
  double const a = data[0];
  double const b = data[1];
  double const c = data[2];
  double const d = X[2] - b;

  x[0] = 0.0;
  x[1] = 0.0;
  x[2] = a * std::exp(-d * d / c / c / 2.0);
}

AAdapt::Circle::Circle(int neq_, int numDim_, Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_)
{
  bool error = true;
  if (neq == 1 || neq == 3) error = false;
  ALBANY_PANIC(error || (numDim != 2), "Error! Invalid call of Circle with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::Circle::compute(double* x, double const* X)
{
  if (((X[0] - .5) * (X[0] - .5) + (X[1] - .5) * (X[1] - .5)) < 1.0 / 16.0)
    x[0] = 1.0;
  else
    x[0] = 0.0;

  // This would be the initial condition for the auxiliary variables, but it
  // should not be needed.
  /*if (neq == 3) {
    x[1] = 0.0;
    x[2] = 0.0;
  }*/
}

AAdapt::SinScalar::SinScalar(int neq_, int numDim_, Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      neq != 1 || numDim < 2 || data.size() != numDim, "Error! Invalid call of SinScalar with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::SinScalar::compute(double* x, double const* X)
{
  x[0] = 1.0;
  for (int dim{0}; dim < numDim; ++dim) {
    x[0] *= sin(pi / data[dim] * X[dim]);
  }
}

AAdapt::ExpressionParser::ExpressionParser(int neq_, int dim_, Teuchos::Array<std::string>& expr_) : dim(dim_), neq(neq_), expr(expr_)
{
  ALBANY_ASSERT(expr.size() == neq, "Must have the same number of equations (" << neq << ") and expressions (" << expr.size() << ").");

  // Parse once here rather than per node in compute(), and bind the coordinate
  // variables by reference to coord_, which compute() then just updates.
  static char const* const coord_str[3] = {"x", "y", "z"};
  evals_.reserve(neq);
  for (int eq = 0; eq < neq; ++eq) {
    auto eval = std::make_shared<stk::expreval::Eval>(expr[eq]);
    eval->parse();
    for (int i = 0; i < dim; ++i) {
      eval->bindVariable(coord_str[i], coord_[i]);
    }
    evals_.push_back(eval);
  }
}

void
AAdapt::ExpressionParser::compute(double* unknowns, double const* coords)
{
  for (int i = 0; i < dim; ++i) {
    coord_[i] = coords[i];
  }
  for (int eq = 0; eq < neq; ++eq) {
    unknowns[eq] = evals_[eq]->evaluate();
  }
}
