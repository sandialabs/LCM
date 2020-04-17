// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef AADAPT_ABSTRACT_ADAPTERT_HPP
#define AADAPT_ABSTRACT_ADAPTERT_HPP

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_VerboseObject.hpp>

#include "Albany_SacadoTypes.hpp"
#include "Albany_StateManager.hpp"

namespace AAdapt {

///
/// \brief Abstract interface for representing the set of adaptation tools
/// available.
///
class AbstractAdapter
{
 public:
  ///
  /// Only constructor
  ///
  AbstractAdapter(
      const Teuchos::RCP<Teuchos::ParameterList>& params_,
      const Teuchos::RCP<ParamLib>&               paramLib_,
      const Albany::StateManager&                 StateMgr_,
      const Teuchos::RCP<Teuchos_Comm const>&     comm_);

  ///
  /// Destructor
  ///
  virtual ~AbstractAdapter() = default;

  //! Method called by LOCA Solver to determine if the mesh needs adapting
  // A return type of true means that the mesh should be adapted
  virtual bool
  queryAdaptationCriteria(int iteration) = 0;

  //! Method called by LOCA Solver to actually adapt the mesh
  //! Apply adaptation method to mesh and problem. Returns true if adaptation is
  //! performed successfully.
  virtual bool
  adaptMesh() = 0;

  virtual void
  postAdapt()
  {
  }

  ///
  /// Each adapter must generate its list of valid parameters
  ///
  virtual Teuchos::RCP<Teuchos::ParameterList const>
  getValidAdapterParameters() const
  {
    return getGenericAdapterParams("Generic Adapter List");
  }

 protected:
  ///
  /// List of valid problem params common to all adapters, as
  /// a starting point for the specific  getValidAdapterParameters
  ///
  Teuchos::RCP<Teuchos::ParameterList>
  getGenericAdapterParams(std::string listname = "AdapterList") const;

  ///
  /// Configurable output stream, defaults to printing on proc=0
  ///
  Teuchos::RCP<Teuchos::FancyOStream> output_stream_;

  ///
  /// Adaptation parameters
  ///
  Teuchos::RCP<Teuchos::ParameterList> adapt_params_;

  ///
  /// Parameter library
  ///
  Teuchos::RCP<ParamLib> param_lib_;

  ///
  /// State Manager
  ///
  const Albany::StateManager& state_mgr_;

  ///
  /// Teuchos communicator
  ///
  Teuchos::RCP<Teuchos_Comm const> teuchos_comm_;
};

}  // namespace AAdapt

#endif  // AADAPT_ABSTRACT_ADAPTERT_HPP
