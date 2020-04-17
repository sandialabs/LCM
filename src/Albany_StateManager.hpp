// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_STATE_MANAGER_HPP
#define ALBANY_STATE_MANAGER_HPP

#include <map>
#include <string>
#include <vector>

#include "Adapt_NodalDataBase.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_EigendataInfoStructT.hpp"
#include "Albany_Macros.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_TpetraTypes.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {

/// Class to manage saved state data.
/* \brief The usage is to register state variables that will be saved
 * during problem construction, where they are described by a string
 * and a DataLayout. One time, the allocate method is called, which
 * creates the memory for a vector of worksets of these states, which
 * are stored as MDFields.
 */

class StateManager
{
 public:
  enum SAType
  {
    ELEM,
    NODE
  };

  StateManager();

  ~StateManager(){};

  typedef std::map<std::string, Teuchos::RCP<PHX::DataLayout>> RegisteredStates;

  /// Method to call multiple times (before allocate) to register which states
  /// will be saved.
  void
  registerStateVariable(
      std::string const&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      std::string const&                   ebName,
      std::string const&                   init_type           = "scalar",
      double const                         init_val            = 0.0,
      bool const                           registerOldState    = false,
      bool const                           outputToExodus      = true,
      std::string const&                   responseIDtoRequire = "",
      StateStruct::MeshFieldEntity const*  fieldEntity         = 0,
      std::string const&                   meshPartName        = "");

  void
  registerNodalVectorStateVariable(
      std::string const&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      std::string const&                   ebName,
      std::string const&                   init_type           = "scalar",
      double const                         init_val            = 0.0,
      bool const                           registerOldState    = false,
      bool const                           outputToExodus      = true,
      std::string const&                   responseIDtoRequire = "");

  /// Method to call multiple times (before allocate) to register which states
  /// will be saved.
  /// Returns param vector with all info to build a SaveStateField or
  /// LoadStateField evaluator
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(
      std::string const&                   name,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const Teuchos::RCP<PHX::DataLayout>& dummy,
      std::string const&                   ebName,
      std::string const&                   init_type        = "scalar",
      double const                         init_val         = 0.0,
      bool const                           registerOldState = false);

  // Field entity is known. Useful for NodalDataToElemNode field. Input dl is of
  // ElemNode type
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(
      std::string const&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      std::string const&                   ebName,
      bool const                           outputToExodus,
      StateStruct::MeshFieldEntity const*  fieldEntity,
      std::string const&                   meshPartName = "");

  /// If field name to save/load is different from state name
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(
      std::string const&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const Teuchos::RCP<PHX::DataLayout>& dummy,
      std::string const&                   ebName,
      std::string const&                   init_type,
      double const                         init_val,
      bool const                           registerOldState,
      std::string const&                   fieldName);

  /// If you want to give more control over whether or not to output to Exodus
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(
      std::string const&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const Teuchos::RCP<PHX::DataLayout>& dummy,
      std::string const&                   ebName,
      std::string const&                   init_type,
      double const                         init_val,
      bool const                           registerOldState,
      bool const                           outputToExodus);

  Teuchos::RCP<Teuchos::ParameterList>
  registerNodalVectorStateVariable(
      std::string const&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const Teuchos::RCP<PHX::DataLayout>& dummy,
      std::string const&                   ebName,
      std::string const&                   init_type,
      double const                         init_val,
      bool const                           registerOldState,
      bool const                           outputToExodus);

  /// Very basic
  void
  registerStateVariable(
      std::string const&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      std::string const&                   init_type);

  Teuchos::RCP<Teuchos::ParameterList>
  registerSideSetStateVariable(
      std::string const&                   sideSetName,
      std::string const&                   stateName,
      std::string const&                   fieldName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      std::string const&                   ebName,
      bool const                           outputToExodus,
      StateStruct::MeshFieldEntity const*  fieldEntity  = NULL,
      std::string const&                   meshPartName = "");

  Teuchos::RCP<Teuchos::ParameterList>
  registerSideSetStateVariable(
      std::string const&                   sideSetName,
      std::string const&                   stateName,
      std::string const&                   fieldName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      std::string const&                   ebName,
      std::string const&                   init_type,
      double const                         init_val,
      bool const                           registerOldState,
      bool const                           outputToExodus,
      std::string const&                   responseIDtoRequire,
      StateStruct::MeshFieldEntity const*  fieldEntity,
      std::string const&                   meshPartName = "");

  /// Method to re-initialize state variables, which can be called multiple
  /// times after allocating
  void
  importStateData(Albany::StateArrays& statesToCopyFrom);

  /// Method to get the Names of the state variables
  std::map<std::string, RegisteredStates> const&
  getRegisteredStates() const
  {
    return statesToStore;
  }

  /// Method to get the Names of the state variables
  std::map<std::string, std::map<std::string, RegisteredStates>> const&
  getRegisteredSideSetStates() const
  {
    return sideSetStatesToStore;
  }

  /// Method to get the ResponseIDs for states which have been registered and
  /// (should)
  ///  have a SaveStateField evaluator associated with them that evaluates the
  ///  responseID
  std::vector<std::string>
  getResidResponseIDsToRequire(std::string& elementBlockName);

  /// Method to make the current newState the oldState, and vice versa
  void
  updateStates();

  /// Method to get a StateInfoStruct of info needed by STK to output States as
  /// Fields
  Teuchos::RCP<Albany::StateInfoStruct>
  getStateInfoStruct() const;

  /// Equivalent of previous method for the sideSets states
  std::map<std::string, Teuchos::RCP<StateInfoStruct>> const&
  getSideSetStateInfoStruct() const;

  /// Method to set discretization object
  void
  setupStateArrays(const Teuchos::RCP<Albany::AbstractDiscretization>& discObj);

  /// Method to get discretization object
  Teuchos::RCP<Albany::AbstractDiscretization>
  getDiscretization() const;

  /// Method to get state information for a specific workset
  Albany::StateArray&
  getStateArray(SAType type, int ws) const;

  /// Method to get state information for all worksets
  Albany::StateArrays&
  getStateArrays() const;

  // Set the state array for all worksets.
  void
  setStateArrays(Albany::StateArrays& sa);

  Albany::StateArrays&
  getSideSetStateArrays(std::string const& sideSet);

  Teuchos::RCP<Adapt::NodalDataBase>
  getNodalDataBase()
  {
    return stateInfo->createNodalDataBase();
  }

  Teuchos::RCP<Adapt::NodalDataBase>
  getSideSetNodalDataBase(std::string const& sideSet)
  {
    return sideSetStateInfo.at(sideSet)->createNodalDataBase();
  }

  void
  printStates(std::string const& where = "") const;

  Teuchos::RCP<Tpetra_MultiVector>
  getAuxDataT();

  void
  setEigenDataT(const Teuchos::RCP<Albany::EigendataStructT>& eigdata);
  void
  setAuxDataT(const Teuchos::RCP<Tpetra_MultiVector>& aux_data);
  bool
  areStateVarsAllocated() const
  {
    return stateVarsAreAllocated;
  }

 private:
  /// Private to prohibit copying
  StateManager(const StateManager&);

  /// Private to prohibit copying
  StateManager&
  operator=(const StateManager&);

  /// Sets states arrays from a given StateInfoStruct into a given
  /// discretization
  void
  doSetStateArrays(
      const Teuchos::RCP<Albany::AbstractDiscretization>& disc,
      const Teuchos::RCP<StateInfoStruct>&                stateInfoPtr);

  /// boolean to enforce that allocate gets called once, and after registration
  /// and befor gets
  bool stateVarsAreAllocated;

  /// Container to hold the states that have been registered, by element block,
  /// to be allocated later
  std::map<std::string, RegisteredStates> statesToStore;
  std::map<std::string, std::map<std::string, RegisteredStates>>
      sideSetStatesToStore;

  /// Discretization object which allows StateManager to perform input/output
  Teuchos::RCP<Albany::AbstractDiscretization> disc;

  /// NEW WAY
  Teuchos::RCP<StateInfoStruct> stateInfo;
  std::map<std::string, Teuchos::RCP<StateInfoStruct>>
      sideSetStateInfo;  // A map sideSetName->stateInfoBd

  Teuchos::RCP<EigendataStructT>   eigenDataT;
  Teuchos::RCP<Tpetra_MultiVector> auxDataT;
};

}  // Namespace Albany

#endif  // ALBANY_STATE_MANAGER_HPP
