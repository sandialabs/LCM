// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_IOSS_STKMESHSTRUCT_HPP
#define ALBANY_IOSS_STKMESHSTRUCT_HPP

#include <Ionit_Initializer.h>

#include <stk_io/IossBridge.hpp>
#include <stk_io/StkMeshIoBroker.hpp>

#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_config.h"

namespace Albany {

class IossSTKMeshStruct : public GenericSTKMeshStruct
{
 public:
  IossSTKMeshStruct(
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<Teuchos::ParameterList>& adaptParams,
      const Teuchos::RCP<Teuchos_Comm const>&     commT);

  //! Borrowing constructor: builds an IossSTKMeshStruct that shares
  //! metaData/bulkData/mesh_data with `donor` instead of opening the
  //! Exodus file a second time. partVec/nsPartVec/ssPartVec/meshSpecs
  //! are copied from the donor (the underlying parts live on the shared
  //! metaData, so the copies are alias maps). The borrowed instance's
  //! setFieldAndBulkData will declare its own field container on the
  //! shared metaData but skip bulk-data setup/commit. Used by
  //! ACE_ThermoMechanical to share one mesh across thermal+mechanical
  //! Applications.
  IossSTKMeshStruct(
      const Teuchos::RCP<IossSTKMeshStruct>&      donor,
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<Teuchos::ParameterList>& adaptParams,
      const Teuchos::RCP<Teuchos_Comm const>&     commT);

  ~IossSTKMeshStruct();

  void
  setFieldAndBulkData(
      const Teuchos::RCP<Teuchos_Comm const>&                                          commT,
      const Teuchos::RCP<Teuchos::ParameterList>&                                      params,
      const unsigned int                                                               neq_,
      const AbstractFieldContainer::FieldContainerRequirements&                        req,
      const Teuchos::RCP<Albany::StateInfoStruct>&                                     sis,
      const unsigned int                                                               worksetSize,
      std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> const&              side_set_sis = {},
      const std::map<std::string, AbstractFieldContainer::FieldContainerRequirements>& side_set_req = {});

  //! Run the deferred post-declaration work on the donor instance:
  //! mesh_data->set_bulk_data, add_all_mesh_fields_as_input_fields,
  //! metaData->commit(), populate bulk data, and read restart. Called by
  //! the orchestrator once all participating apps have declared their
  //! fields. Borrowed instances must NOT call this (they share state
  //! with the donor).
  void
  commitAndPopulate(
      const Teuchos::RCP<Teuchos_Comm const>&      commT,
      const Teuchos::RCP<Teuchos::ParameterList>&  params,
      const Teuchos::RCP<Albany::StateInfoStruct>& sis) override;

  int
  getSolutionFieldHistoryDepth() const
  {
    return m_solutionFieldHistoryDepth;
  }
  double
  getSolutionFieldHistoryStamp(int step) const;
  void
  loadSolutionFieldHistory(int step);

  //! Flag if solution has a restart values -- used in Init Cond
  bool
  hasRestartSolution() const
  {
    return m_hasRestartSolution;
  }

  //! If restarting, convenience function to return restart data time
  double
  restartDataTime() const
  {
    return m_restartDataTime;
  }

 private:
  Ioss::Init::Initializer ioInit;

  void
  loadOrSetCoordinates3d(int index);

  Teuchos::RCP<Teuchos::ParameterList const>
  getValidDiscretizationParameters() const;

  Teuchos::RCP<Teuchos::FancyOStream>    out;
  bool                                   usePamgen;
  bool                                   useSerialMesh;
  bool                                   periodic;
  Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data;

  bool   m_hasRestartSolution;
  double m_restartDataTime;
  int    m_solutionFieldHistoryDepth;

  //! Parameters cached by setFieldAndBulkData for use by a deferred
  //! commitAndPopulate. Only meaningful when deferCommit is true and the
  //! orchestrator will invoke commitAndPopulate later.
  AbstractFieldContainer::FieldContainerRequirements                        cached_req_;
  std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>>              cached_side_set_sis_;
  std::map<std::string, AbstractFieldContainer::FieldContainerRequirements> cached_side_set_req_;
  unsigned int                                                              cached_worksetSize_{0};
};

}  // Namespace Albany

#endif  // ALBANY_IOSS_STKMESHSTRUCT_HPP
