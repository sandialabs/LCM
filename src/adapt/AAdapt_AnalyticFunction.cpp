// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "AAdapt_AnalyticFunction.hpp"

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

  else if (name == "Step X")
    F = Teuchos::rcp(new AAdapt::StepX(neq, numDim, data));

  else if (name == "TemperatureStep")
    F = Teuchos::rcp(new AAdapt::TemperatureStep(neq, numDim, data));

  else if (name == "Displacement Constant TemperatureStep")
    F = Teuchos::rcp(new AAdapt::DispConstTemperatureStep(neq, numDim, data));

  else if (name == "Displacement Constant TemperatureLinear")
    F = Teuchos::rcp(new AAdapt::DispConstTemperatureLinear(neq, numDim, data));

  else if (name == "TemperatureLinear")
    F = Teuchos::rcp(new AAdapt::TemperatureLinear(neq, numDim, data));

  else if (name == "1D Gauss-Sin")
    F = Teuchos::rcp(new AAdapt::GaussSin(neq, numDim, data));

  else if (name == "1D Gauss-Cos")
    F = Teuchos::rcp(new AAdapt::GaussCos(neq, numDim, data));

  else if (name == "Linear Y")
    F = Teuchos::rcp(new AAdapt::LinearY(neq, numDim, data));

  else if (name == "Linear")
    F = Teuchos::rcp(new AAdapt::Linear(neq, numDim, data));

  else if (name == "Constant Box")
    F = Teuchos::rcp(new AAdapt::ConstantBox(neq, numDim, data));

  else if (name == "About Z")
    F = Teuchos::rcp(new AAdapt::AboutZ(neq, numDim, data));

  else if (name == "Radial Z")
    F = Teuchos::rcp(new AAdapt::RadialZ(neq, numDim, data));

  else if (name == "About Linear Z")
    F = Teuchos::rcp(new AAdapt::AboutLinearZ(neq, numDim, data));

  else if (name == "Gaussian Z")
    F = Teuchos::rcp(new AAdapt::GaussianZ(neq, numDim, data));

  else if (name == "Circle")
    F = Teuchos::rcp(new AAdapt::Circle(neq, numDim, data));

  else if (name == "Gaussian Pressure")
    F = Teuchos::rcp(new AAdapt::GaussianPress(neq, numDim, data));

  else if (name == "Sin-Cos")
    F = Teuchos::rcp(new AAdapt::SinCos(neq, numDim, data));

  else if (name == "Sin Scalar")
    F = Teuchos::rcp(new AAdapt::SinScalar(neq, numDim, data));

  else if (name == "Taylor-Green Vortex")
    F = Teuchos::rcp(new AAdapt::TaylorGreenVortex(neq, numDim, data));

  else if (name == "1D Acoustic Wave")
    F = Teuchos::rcp(new AAdapt::AcousticWave(neq, numDim, data));

  else
    ALBANY_PANIC(name != "Valid Initial Condition Function", "Unrecognized initial condition function name: " << name);

  return F;
}

AAdapt::ConstantFunction::ConstantFunction(int neq_, int numDim_, Teuchos::Array<double> data_)
    : numDim(numDim_), neq(neq_), data(data_)
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

AAdapt::StepX::StepX(int neq_, int numDim_, Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (data.size() != 5),
      "Error! Invalid specification of initial condition: incorrect length of "
      "Function Data for Step X; Length = "
          << 5 << ", data.size() = " << data.size() << std::endl);
}

void
AAdapt::StepX::compute(double* x, double const* X)
{
  // Temperature bottom
  double T0 = data[0];
  // Temperature top
  double T1 = data[1];
  // constant temperature
  double T = data[2];
  // bottom x-coordinate
  double X0 = data[3];
  // top x-coordinate
  double X1 = data[4];

  double const TOL = 1.0e-12;

  // bottom
  if (X[0] < (X0 + TOL)) {
    x[0] = T0;
  } else if (X[0] > (X1 - TOL)) {
    x[0] = T1;
  } else {
    x[0] = T;
  }
}

AAdapt::TemperatureStep::TemperatureStep(int neq_, int numDim_, Teuchos::Array<double> data_)
    : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (data.size() != 6),
      "Error! Invalid specification of initial condition: incorrect length of "
      "Function Data for TemperatureStep; Length = "
          << 6 << ", data.size() = " << data.size() << std::endl);
}

void
AAdapt::TemperatureStep::compute(double* x, double const* X)
{
  // Temperature bottom
  double T0 = data[0];
  // Temperature top
  double T1 = data[1];
  // constant temperature
  double T = data[2];
  // bottom coordinate
  double Z0 = data[3];
  // top coordinate
  double Z1 = data[4];
  // flag to specify which coordinate we want.
  // 0 == x-coordinate
  // 1 == y-coordinate
  // 2 == z-cordinate
  int coord = static_cast<int>(data[5]);

  // check that coordinate is valid
  if ((coord > 2) || (coord < 0)) {
    ALBANY_ABORT("Error! Coordinate not valid!" << std::endl);
  }

  double const TOL = 1.0e-12;

  // bottom
  if (X[coord] < (Z0 + TOL)) {
    x[0] = T0;
  } else if (X[coord] > (Z1 - TOL)) {
    x[0] = T1;
  } else {
    x[0] = T;
  }
}

AAdapt::DispConstTemperatureStep::DispConstTemperatureStep(int neq_, int numDim_, Teuchos::Array<double> data_)
    : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (data.size() != 9),
      "Error! Invalid specification of initial condition: incorrect length of "
      "Function Data for Displacement Constant TemperatureStep; Length = "
          << 9 << ", data.size() = " << data.size() << std::endl);
}

void
AAdapt::DispConstTemperatureStep::compute(double* x, double const* X)
{
  // Get displacement
  for (int i = 0; i < 3; i++) x[i] = data[i];
  // Temperature bottom
  double T0 = data[3];
  // Temperature top
  double T1 = data[4];
  // constant temperature
  double T = data[5];
  // bottom coordinate
  double Z0 = data[6];
  // top coordinate
  double Z1 = data[7];
  // flag to specify which coordinate we want.
  // 0 == x-coordinate
  // 1 == y-coordinate
  // 2 == z-cordinate
  int coord = static_cast<int>(data[8]);

  // check that coordinate is valid
  if ((coord > 2) || (coord < 0)) {
    ALBANY_ABORT("Error! Coordinate not valid!" << std::endl);
  }

  double const TOL = 1.0e-12;

  // bottom
  if (X[coord] < (Z0 + TOL)) {
    x[3] = T0;
  } else if (X[coord] > (Z1 - TOL)) {
    x[3] = T1;
  } else {
    x[3] = T;
  }
}

AAdapt::DispConstTemperatureLinear::DispConstTemperatureLinear(int neq_, int numDim_, Teuchos::Array<double> data_)
    : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (data.size() != 8),
      "Error! Invalid specification of initial condition: incorrect length of "
      "Function Data for Displacement Constant TemperatureLinear; Length = "
          << 8 << ", data.size() = " << data.size() << std::endl);
}

void
AAdapt::DispConstTemperatureLinear::compute(double* x, double const* X)
{
  // Get displacement
  for (int i = 0; i < 3; i++) x[i] = data[i];
  // Temperature bottom
  double T0 = data[3];
  // Temperature top
  double T1 = data[4];
  // bottom coordinate
  double Z0 = data[5];
  // top coordinate
  double Z1 = data[6];
  // flag to specify which coordinate we want.
  // 0 == x-coordinate
  // 1 == y-coordinate
  // 2 == z-cordinate
  int coord = static_cast<int>(data[7]);

  // check that coordinate is valid
  if ((coord > 2) || (coord < 0)) {
    ALBANY_ABORT("Error! Coordinate not valid!" << std::endl);
  }

  double const TOL = 1.0e-12;

  // check that temperatures are not equal
  if (std::abs(T0 - T1) <= TOL) {
    ALBANY_ABORT("Error! Temperature are equals!" << std::endl);
  }
  // check coordinates are not equal
  if (std::abs(Z0 - Z1) <= TOL) {
    ALBANY_ABORT("Error! Z-coordinates are the same!" << std::endl);
  }

  // We interpolate Temperature as a linear function of z-ccordinate: T = b +
  // m*z
  double b = (T1 * Z0 - T0 * Z1) / (Z0 - Z1);
  double m = (T0 - T1) / (Z0 - Z1);

  // assign temperature
  x[3] = b + m * X[coord];
}

AAdapt::TemperatureLinear::TemperatureLinear(int neq_, int numDim_, Teuchos::Array<double> data_)
    : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (data.size() != 5),
      "Error! Invalid specification of initial condition: incorrect length of "
      "Function Data for TemperatureLinear; Length = "
          << 5 << ", data.size() = " << data.size() << std::endl);
}

void
AAdapt::TemperatureLinear::compute(double* x, double const* X)
{
  // Temperature bottom
  double T0 = data[0];
  // Temperature top
  double T1 = data[1];
  // bottom coordinate
  double Z0 = data[2];
  // top coordinate
  double Z1 = data[3];
  // flag to specify which coordinate we want.
  // 0 == x-coordinate
  // 1 == y-coordinate
  // 2 == z-cordinate
  int coord = static_cast<int>(data[4]);

  // check that coordinate is valid
  if ((coord > 2) || (coord < 0)) {
    ALBANY_ABORT("Error! Coordinate not valid!" << std::endl);
  }

  double const TOL = 1.0e-12;

  // check that temperatures are not equal
  if (std::abs(T0 - T1) <= TOL) {
    ALBANY_ABORT("Error! Temperature are equals!" << std::endl);
  }
  // check coordinates are not equal
  if (std::abs(Z0 - Z1) <= TOL) {
    ALBANY_ABORT("Error! Z-coordinates are the same!" << std::endl);
  }

  // We interpolate Temperature as a linear function of z-ccordinate: T = b +
  // m*z
  double b = (T1 * Z0 - T0 * Z1) / (Z0 - Z1);
  double m = (T0 - T1) / (Z0 - Z1);

  // assign temperature
  x[0] = b + m * X[coord];
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
          << "Function Data for Constant Function Perturbed; neq = " << neq << ", data.size() = " << data.size()
          << ", pert_mag.size() = " << pert_mag.size() << std::endl);

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
      //      rng(boost::random::random_device()()), // seed the rng
      rng(seedgen(worksetID)),  // seed the rng
      nd(neq_),
      var_nor(neq_)
{
  ALBANY_PANIC(
      (data.size() != neq || pert_mag.size() != neq),
      "Error! Invalid specification of initial condition: incorrect length of "
          << "Function Data for Constant Function Gaussian Perturbed; neq = " << neq
          << ", data.size() = " << data.size() << ", pert_mag.size() = " << pert_mag.size() << std::endl);

  if (data.size() > 0 && pert_mag.size() > 0)
    for (int i = 0; i < neq; i++)
      if (pert_mag[i] > std::numeric_limits<double>::epsilon()) {
        nd[i]      = Teuchos::rcp(new boost::normal_distribution<double>(data[i], pert_mag[i]));
        var_nor[i] = Teuchos::rcp(
            new boost::variate_generator<boost::mt19937&, boost::normal_distribution<double>>(rng, *nd[i]));
      }
}

void
AAdapt::ConstantFunctionGaussianPerturbed::compute(double* x, double const* X)
{
  for (int i = 0; i < neq; i++)
    if (var_nor[i] != Teuchos::null)
      x[i] = (*var_nor[i])();

    else
      x[i] = data[i];
}

AAdapt::GaussSin::GaussSin(int neq_, int numDim_, Teuchos::Array<double> data_)
    : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq != 1) || (numDim != 1) || (data.size() != 1),
      "Error! Invalid call of GaussSin with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::GaussSin::compute(double* x, double const* X)
{
  x[0] = sin(pi * X[0]) + 0.5 * data[0] * X[0] * (1.0 - X[0]);
}

AAdapt::GaussCos::GaussCos(int neq_, int numDim_, Teuchos::Array<double> data_)
    : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq != 1) || (numDim != 1) || (data.size() != 1),
      "Error! Invalid call of GaussCos with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::GaussCos::compute(double* x, double const* X)
{
  x[0] = 1 + cos(2 * pi * X[0]) + 0.5 * data[0] * X[0] * (1.0 - X[0]);
}

AAdapt::LinearY::LinearY(int neq_, int numDim_, Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq < 2) || (numDim < 2) || (data.size() != 1),
      "Error! Invalid call of LinearY with " << neq << " " << numDim << "  " << data.size() << std::endl);
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
  ALBANY_PANIC(
      (data.size() != neq * numDim),
      "Error! Invalid call of Linear with " << neq << " " << numDim << "  " << data.size() << std::endl);
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

AAdapt::ConstantBox::ConstantBox(int neq_, int numDim_, Teuchos::Array<double> data_)
    : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (data.size() != 2 * numDim + neq),
      "Error! Invalid call of Linear with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::ConstantBox::compute(double* x, double const* X)
{
  bool in_box{true};
  for (auto dim = 0; dim < numDim; ++dim) {
    double const& lo = data[dim];
    double const& hi = data[dim + numDim];
    in_box           = in_box && lo <= X[dim] && X[dim] <= hi;
  }

  if (in_box == true) {
    for (auto eq = 0; eq < neq; ++eq) {
      x[eq] = data[2 * numDim + eq];
    }
  }
}

AAdapt::AboutZ::AboutZ(int neq_, int numDim_, Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq < 2) || (numDim < 2) || (data.size() != 1),
      "Error! Invalid call of AboutZ with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::AboutZ::compute(double* x, double const* X)
{
  x[0] = -data[0] * X[1];
  x[1] = data[0] * X[0];

  if (neq > 2) x[2] = 0.0;
}

AAdapt::RadialZ::RadialZ(int neq_, int numDim_, Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq < 2) || (numDim < 2) || (data.size() != 1),
      "Error! Invalid call of RadialZ with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::RadialZ::compute(double* x, double const* X)
{
  x[0] = data[0] * X[0];
  x[1] = data[0] * X[1];

  if (neq > 2) x[2] = 0.0;
}

AAdapt::AboutLinearZ::AboutLinearZ(int neq_, int numDim_, Teuchos::Array<double> data_)
    : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq < 3) || (numDim < 3) || (data.size() != 1),
      "Error! Invalid call of AboutLinearZ with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::AboutLinearZ::compute(double* x, double const* X)
{
  x[0] = -data[0] * X[1] * X[2];
  x[1] = data[0] * X[0] * X[2];
  x[2] = 0.0;
}

AAdapt::GaussianZ::GaussianZ(int neq_, int numDim_, Teuchos::Array<double> data_)
    : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq < 2) || (numDim < 2) || (data.size() != 3),
      "Error! Invalid call of GaussianZ with " << neq << " " << numDim << "  " << data.size() << std::endl);
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
  ALBANY_PANIC(
      error || (numDim != 2),
      "Error! Invalid call of Circle with " << neq << " " << numDim << "  " << data.size() << std::endl);
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

AAdapt::GaussianPress::GaussianPress(int neq_, int numDim_, Teuchos::Array<double> data_)
    : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq < 3) || (numDim < 2) || (data.size() != 4),
      "Error! Invalid call of GaussianPress with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::GaussianPress::compute(double* x, double const* X)
{
  for (int i = 0; i < neq - 1; i++) {
    x[i] = 0.0;
  }

  x[neq - 1] = data[0] * exp(-data[1] * ((X[0] - data[2]) * (X[0] - data[2]) + (X[1] - data[3]) * (X[1] - data[3])));
}

AAdapt::SinCos::SinCos(int neq_, int numDim_, Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq < 3) || (numDim < 2),
      "Error! Invalid call of SinCos with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::SinCos::compute(double* x, double const* X)
{
  x[0] = sin(2.0 * pi * X[0]) * cos(2.0 * pi * X[1]);
  x[1] = cos(2.0 * pi * X[0]) * sin(2.0 * pi * X[1]);
  x[2] = sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[1]);
}

AAdapt::SinScalar::SinScalar(int neq_, int numDim_, Teuchos::Array<double> data_)
    : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      neq != 1 || numDim < 2 || data.size() != numDim,
      "Error! Invalid call of SinScalar with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::SinScalar::compute(double* x, double const* X)
{
  x[0] = 1.0;
  for (int dim{0}; dim < numDim; ++dim) {
    x[0] *= sin(pi / data[dim] * X[dim]);
  }
}

AAdapt::TaylorGreenVortex::TaylorGreenVortex(int neq_, int numDim_, Teuchos::Array<double> data_)
    : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq < 3) || (numDim != 2),
      "Error! Invalid call of TaylorGreenVortex with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::TaylorGreenVortex::compute(double* x, double const* X)
{
  x[0] = 1.0;                                           // initial density
  x[1] = -cos(2.0 * pi * X[0]) * sin(2.0 * pi * X[1]);  // initial u-velocity
  x[2] = sin(2.0 * pi * X[0]) * cos(2.0 * pi * X[1]);   // initial v-velocity
  x[3] = cos(2.0 * pi * X[0]) + cos(2.0 * pi * X[1]);   // initial temperature
}

AAdapt::AcousticWave::AcousticWave(int neq_, int numDim_, Teuchos::Array<double> data_)
    : numDim(numDim_), neq(neq_), data(data_)
{
  ALBANY_PANIC(
      (neq > 3) || (numDim > 2) || (data.size() != 3),
      "Error! Invalid call of AcousticWave with " << neq << " " << numDim << "  " << data.size() << std::endl);
}
void
AAdapt::AcousticWave::compute(double* x, double const* X)
{
  double const U0 = data[0];
  double const n  = data[1];
  double const L  = data[2];
  x[0]            = U0 * cos(n * X[0] / L);

  for (int i = 1; i < numDim; i++) x[i] = 0.0;
}

AAdapt::ExpressionParser::ExpressionParser(int neq_, int dim_, Teuchos::Array<std::string>& expr_)
    : dim(dim_), neq(neq_), expr(expr_)
{
  ALBANY_ASSERT(
      expr.size() == neq,
      "Must have the same number of equations (" << neq << ") and expressions (" << expr.size() << ").");
}

void
AAdapt::ExpressionParser::compute(double* unknowns, double const* coords)
{
  std::vector<std::string> coord_str{"x", "y", "z"};
  double*                  X = const_cast<double*>(coords);
  for (auto eq = 0; eq < neq; ++eq) {
    auto const&         expr_str = expr[eq];
    stk::expreval::Eval expr_eval(expr_str);
    expr_eval.parse();
    for (auto i = 0; i < dim; ++i) {
      expr_eval.bindVariable(coord_str[i], X[i]);
    }
    unknowns[eq] = expr_eval.evaluate();
  }
}
