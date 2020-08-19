// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_SOLVER_FACTORY_HPP
#define ALBANY_SOLVER_FACTORY_HPP

#include "Albany_Application.hpp"
#include "Piro_ObserverBase.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Thyra_ModelEvaluator.hpp"
#include "Thyra_ResponseOnlyModelEvaluatorBase.hpp"
#include "Thyra_VectorBase.hpp"

//! Albany driver code, problems, discretizations, and responses
namespace Albany {

/*!
 * \brief A factory class to instantiate AbstractSolver objects
 */
class SolverFactory
{
 public:
  //! Default constructor
  SolverFactory(std::string const& inputfile, const Teuchos::RCP<Teuchos_Comm const>& comm);

  SolverFactory(
      const Teuchos::RCP<Teuchos::ParameterList>& input_appParams,
      const Teuchos::RCP<Teuchos_Comm const>&     comm);

  //! Destructor
  virtual ~SolverFactory() = default;

  // Thyra version of above
  virtual Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>
  create(
      const Teuchos::RCP<Teuchos_Comm const>& appComm,
      const Teuchos::RCP<Teuchos_Comm const>& solverComm,
      Teuchos::RCP<Thyra_Vector const> const& initial_guess = Teuchos::null);

  // Thyra version of above
  Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>
  createAndGetAlbanyApp(
      Teuchos::RCP<Application>&              albanyApp,
      const Teuchos::RCP<Teuchos_Comm const>& appComm,
      const Teuchos::RCP<Teuchos_Comm const>& solverComm,
      Teuchos::RCP<Thyra_Vector const> const& initial_guess   = Teuchos::null,
      bool                                    createAlbanyApp = true);

  // Thyra version of above
  Teuchos::RCP<Thyra::ModelEvaluator<ST>>
  createAlbanyAppAndModel(
      Teuchos::RCP<Application>&              albanyApp,
      const Teuchos::RCP<Teuchos_Comm const>& appComm,
      Teuchos::RCP<Thyra_Vector const> const& initial_guess   = Teuchos::null,
      bool const                              createAlbanyApp = true);

  Teuchos::ParameterList&
  getAnalysisParameters() const
  {
    return appParams->sublist("Piro").sublist("Analysis");
  }

  Teuchos::ParameterList&
  getParameters() const
  {
    return *appParams;
  }

  Teuchos::RCP<Teuchos::ParameterList> const
  getParametersRCP() const
  {
    return appParams;
  }

  void
  setSchwarz(bool const schwarz)
  {
    is_schwarz_ = schwarz;
  }

 public:
  // Functions to generate reference parameter lists for validation
  //  EGN 9/2013: made these three functions public, as they pertain to valid
  //    parameter lists for Albany::Application objects, which may get created
  //    apart from Albany::SolverFactory.  It may be better to relocate these
  //    to the Application class, or as functions "related to"
  //    Albany::Application.
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidAppParameters() const;
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidDebugParameters() const;
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidScalingParameters() const;
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidParameterParameters() const;
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidResponseParameters() const;

 private:
  // Private functions to set default parameter values
  void
  setSolverParamDefaults(Teuchos::ParameterList* appParams, int myRank);

  Teuchos::RCP<Teuchos::ParameterList const>
  getValidRegressionResultsParameters() const;

 public:
  int
  checkSolveTestResults(
      int                                          response_index,
      int                                          parameter_index,
      Teuchos::RCP<Thyra_Vector const> const&      g,
      const Teuchos::RCP<const Thyra_MultiVector>& dgdp) const;

  /** \brief Function that does regression testing for Dakota runs. */
  int
  checkDakotaTestResults(int response_index, const Teuchos::SerialDenseVector<int, double>* drdv) const;

  /** \brief Function that does regression testing for Analysis runs. */
  int
  checkAnalysisTestResults(int response_index, const Teuchos::RCP<Thyra::VectorBase<double>>& tvec) const;

  Teuchos::RCP<Thyra::ModelEvaluator<ST>>
  returnModel() const
  {
    return model_;
  };

  Teuchos::RCP<Piro::ObserverBase<double>>
  returnObserver() const
  {
    return observer_;
  };

 private:
  /** \brief Testing utility that compares two numbers using two tolerances */
  bool
  scaledCompare(double x1, double x2, double relTol, double absTol, std::string const& name) const;

  Teuchos::ParameterList*
  getTestParameters(int response_index) const;

  void
  storeTestResults(Teuchos::ParameterList* testParams, int failures, int comparisons) const;

  Teuchos::RCP<Thyra::ModelEvaluator<ST>> model_;

  Teuchos::RCP<Piro::ObserverBase<double>> observer_;

 protected:
  //! Parameter list specifying what solver to create
  Teuchos::RCP<Teuchos::ParameterList> appParams;

  Teuchos::RCP<Teuchos::FancyOStream> out;

  bool is_schwarz_{false};
};

}  // namespace Albany

#endif  // ALBANY_SOLVER_FACTORY_HPP
