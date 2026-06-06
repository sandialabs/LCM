// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_SchwarzTransfer.hpp"

#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Albany_Application.hpp"
#include "Albany_Macros.hpp"
#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "Albany_STKDiscretization.hpp"
#include "DTK_MapOperatorFactory.hpp"
#include "Teuchos_RCPDecl.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Part.hpp"
#include "stk_mesh/base/Selector.hpp"

namespace Albany {

Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>
doDTKInterpolation(
    DataTransferKit::STKMeshManager&                    coupled_manager,
    DataTransferKit::STKMeshManager&                    this_manager,
    AbstractSTKFieldContainer::VectorFieldType*         coupled_field,
    AbstractSTKFieldContainer::VectorFieldType*         this_field,
    int const                                           neq,
    Teuchos::ParameterList&                             dtk_params)
{
  // Source-side Tpetra MultiVector backed by the coupled-mesh STK field.
  Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>> coupled_vector =
      coupled_manager.createFieldMultiVector<AbstractSTKFieldContainer::VectorFieldType>(Teuchos::ptr(coupled_field), neq);

  // Target-side MultiVector backed by the this-mesh STK field. After
  // map_op->apply the interpolated values land in both this MV and the
  // underlying STK field storage.
  Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>> this_vector =
      this_manager.createFieldMultiVector<AbstractSTKFieldContainer::VectorFieldType>(Teuchos::ptr(this_field), neq);

  DataTransferKit::MapOperatorFactory op_factory;

  Teuchos::RCP<DataTransferKit::MapOperator> map_op =
      op_factory.create(coupled_vector->getMap(), this_vector->getMap(), dtk_params);

  // Setup builds the interpolation linear operator (consistent interpolation
  // by default — see dtk_params["Map Type"]).
  map_op->setup(coupled_manager.functionSpace(), this_manager.functionSpace());

  // Apply does the mesh-to-mesh transfer.
  map_op->apply(*coupled_vector, *this_vector);

  return this_vector;
}

Teuchos::Array<Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>>
computeSchwarzTransferDTK(
    Application const& this_app,
    Application const& coupled_app,
    std::string const& nodeset_name)
{
  ALBANY_EXPECT(this_app.getNumEquations() == coupled_app.getNumEquations());
  int const neq = this_app.getNumEquations();

  // Source mesh + field array (one entry per time-derivative slot).
  auto* coupled_stk_disc = static_cast<STKDiscretization*>(coupled_app.getDiscretization().get());
  Teuchos::RCP<stk::mesh::MetaData const> const coupled_meta_data =
      Teuchos::rcpFromRef(coupled_stk_disc->getSTKMetaData());
  Teuchos::RCP<Teuchos::ParameterList const> coupled_app_params = coupled_app.getAppPL();
  Teuchos::ParameterList                     dtk_params         = coupled_app_params->sublist("DataTransferKit");
  // ParameterList::get(name, default) has a side effect of *adding* the
  // parameter with the default value if not present. DTK's MapOperatorFactory
  // expects "Map Type" to be in dtk_params before it can construct the
  // operator; the original StrongSchwarzBC::computeBCsDTK relied on this
  // side-effect call to seed it.
  (void)dtk_params.get("Map Type", std::string("Consistent Interpolation"));

  Teuchos::Array<AbstractSTKFieldContainer::VectorFieldType*> coupled_field_array =
      Teuchos::rcp_dynamic_cast<OrdinarySTKFieldContainer<true>>(coupled_stk_disc->getSTKMeshStruct()->getFieldContainer())
          ->getSolutionFieldArray();
  int const num_sol_vecs = coupled_field_array.length();
  ALBANY_ASSERT(num_sol_vecs > 0, "coupled_field_array must have at least 1 entry!");

  AbstractSTKFieldContainer::VectorFieldType* coupled_field        = coupled_field_array[0];
  stk::mesh::Selector                         coupled_stk_selector = stk::mesh::Selector(coupled_meta_data->universal_part());
  Teuchos::RCP<stk::mesh::BulkData>           coupled_bulk_data    = Teuchos::rcpFromRef(coupled_field->get_mesh());

  // Target mesh + DTK field array (paired with coupled_field_array slot-for-slot).
  auto* this_stk_disc = static_cast<STKDiscretization*>(this_app.getDiscretization().get());
  Teuchos::RCP<stk::mesh::MetaData const> this_meta_data = Teuchos::rcpFromRef(this_stk_disc->getSTKMetaData());

  Teuchos::Array<AbstractSTKFieldContainer::VectorFieldType*> this_field_array =
      Teuchos::rcp_dynamic_cast<OrdinarySTKFieldContainer<true>>(this_stk_disc->getSTKMeshStruct()->getFieldContainer())
          ->getSolutionFieldDTKArray();
  ALBANY_ASSERT(num_sol_vecs == this_field_array.length(),
                "coupled_field_array and this_field_array must have the same length!");

  AbstractSTKFieldContainer::VectorFieldType* this_field      = this_field_array[0];
  stk::mesh::Part*                            this_part       = this_meta_data->get_part(nodeset_name);
  Teuchos::RCP<stk::mesh::BulkData>           this_bulk_data  = Teuchos::rcpFromRef(this_field->get_mesh());

  DataTransferKit::STKMeshManager coupled_manager(coupled_bulk_data, coupled_stk_selector);
  stk::mesh::Selector             this_stk_selector(*this_part);
  DataTransferKit::STKMeshManager this_manager(this_bulk_data, this_stk_selector);

  Teuchos::Array<Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>> result(num_sol_vecs);
  result[0] = doDTKInterpolation(coupled_manager, this_manager, coupled_field, this_field, neq, dtk_params);

  if (num_sol_vecs > 1) {
    result[1] = doDTKInterpolation(coupled_manager, this_manager, coupled_field_array[1], this_field_array[1], neq, dtk_params);
  }
  if (num_sol_vecs > 2) {
    result[2] = doDTKInterpolation(coupled_manager, this_manager, coupled_field_array[2], this_field_array[2], neq, dtk_params);
  }

  return result;
}

}  // namespace Albany
