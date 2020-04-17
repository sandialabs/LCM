// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_SIDE_SET_STK_MESH_STRUCT_HPP
#define ALBANY_SIDE_SET_STK_MESH_STRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"

namespace Albany {

class SideSetSTKMeshStruct : public GenericSTKMeshStruct
{
 public:
  SideSetSTKMeshStruct(
      const MeshSpecsStruct&                      inputMeshSpecs,
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<Teuchos_Comm const>&     commT);

  virtual ~SideSetSTKMeshStruct();

  void
  setFieldAndBulkData(
      const Teuchos::RCP<Teuchos_Comm const>&                   comm,
      const Teuchos::RCP<Teuchos::ParameterList>&               params,
      const unsigned int                                        neq_,
      const AbstractFieldContainer::FieldContainerRequirements& req,
      const Teuchos::RCP<Albany::StateInfoStruct>&              sis,
      const unsigned int                                        worksetSize,
      std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> const&
          side_set_sis = {},
      const std::
          map<std::string, AbstractFieldContainer::FieldContainerRequirements>&
              side_set_req = {});

  void
  setParentMeshInfo(
      const AbstractSTKMeshStruct& parentMeshStruct_,
      std::string const&           sideSetName);

  bool
  hasRestartSolution() const
  {
    return false;
  }
  double
  restartDataTime() const
  {
    return 0.;
  }

  Teuchos::RCP<Teuchos::ParameterList const>
  getValidDiscretizationParameters() const;

 private:
  Teuchos::RCP<const AbstractSTKMeshStruct> parentMeshStruct;  // Weak ptr
  std::string                               parentMeshSideSetName;
};

}  // Namespace Albany

#endif  // ALBANY_SIDE_SET_STK_MESH_STRUCT_HPP
