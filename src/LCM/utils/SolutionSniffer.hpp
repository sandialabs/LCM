// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LCM_SolutionSniffer_hpp)
#define LCM_SolutionSniffer_hpp

#include "Albany_DataTypes.hpp"
#include "NOX_Abstract_PrePostOperator.H"
#include "NOX_Abstract_Vector.H"
#include "NOX_StatusTest_ModelEvaluatorFlag.hpp"

namespace Albany {
class Application;
}

namespace LCM {

///
/// Observer that is called at various points in the NOX nonlinear solver
///
class SolutionSniffer : public NOX::Abstract::PrePostOperator
{
 public:
  SolutionSniffer();

  //! Attach the owning Application so runPreSolve can drive the
  //! linear-elastic warmstart predictor before NOX's first residual fill.
  //! Optional — when unset (or when the Application has no DBC descriptors)
  //! runPreSolve falls back to status-test reset only.
  void
  setApplication(Teuchos::RCP<Albany::Application> const& app);

  virtual ~SolutionSniffer();

  virtual void
  runPreIterate(NOX::Solver::Generic const& solver);

  virtual void
  runPostIterate(NOX::Solver::Generic const& solver);

  virtual void
  runPreSolve(NOX::Solver::Generic const& solver);

  virtual void
  runPostSolve(NOX::Solver::Generic const& solver);

  void
  setStatusTest(Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> status_test);

  ST
  getInitialNorm();

  ST
  getFinalNorm();

  ST
  getDifferenceNorm();

  Teuchos::RCP<NOX::Abstract::Vector>
  getLastSoln();

 private:
  Teuchos::RCP<Albany::Application>                 app_{Teuchos::null};
  Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> status_test_{Teuchos::null};
  Teuchos::RCP<NOX::Abstract::Vector>               soln_init_{Teuchos::null};
  Teuchos::RCP<NOX::Abstract::Vector>               last_soln_{Teuchos::null};
  ST                                                norm_init_{0.0};
  ST                                                norm_final_{0.0};
  ST                                                norm_diff_{0.0};
};

}  // namespace LCM

#endif  // LCM_SolutionSniffer_hpp
