#ifndef ALBANY_TPETRA_THYRA_UTILS_HPP
#define ALBANY_TPETRA_THYRA_UTILS_HPP

#include "Albany_ThyraTypes.hpp"
#include "Albany_TpetraTypes.hpp"

namespace Albany {

// The wrappers in thyra throw if the input Thyra/Tpetra pointer is null
// These routines are here to handle that case, and simply return a
// Teuchos::null if the input RCP is null. They are just a convenience
// routine that performs the check before calling the Thyra converter.

// ============ Tpetra->Thyra conversion routines ============ //
Teuchos::RCP<const Thyra_SpmdVectorSpace>
createThyraVectorSpace(const Teuchos::RCP<const Tpetra_Map>& map);

Teuchos::RCP<Thyra_Vector>
createThyraVector(const Teuchos::RCP<Tpetra_Vector>& v);

Teuchos::RCP<Thyra_Vector const>
createConstThyraVector(const Teuchos::RCP<const Tpetra_Vector>& v);

Teuchos::RCP<Thyra_MultiVector>
createThyraMultiVector(const Teuchos::RCP<Tpetra_MultiVector>& mv);

Teuchos::RCP<const Thyra_MultiVector>
createConstThyraMultiVector(const Teuchos::RCP<const Tpetra_MultiVector>& mv);

Teuchos::RCP<Thyra_LinearOp>
createThyraLinearOp(const Teuchos::RCP<Tpetra_Operator>& op);

Teuchos::RCP<const Thyra_LinearOp>
createConstThyraLinearOp(const Teuchos::RCP<const Tpetra_Operator>& op);

// ============ Thyra->Tpetra conversion routines ============ //
Teuchos::RCP<const Tpetra_Map>
getTpetraMap(Teuchos::RCP<Thyra_VectorSpace const> const& vs, bool const throw_if_not_tpetra = true);

Teuchos::RCP<Tpetra_Vector>
getTpetraVector(Teuchos::RCP<Thyra_Vector> const& v, bool const throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_Vector>
getConstTpetraVector(Teuchos::RCP<Thyra_Vector const> const& v, bool const throw_if_not_tpetra = true);

Teuchos::RCP<Tpetra_MultiVector>
getTpetraMultiVector(Teuchos::RCP<Thyra_MultiVector> const& mv, bool const throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_MultiVector>
getConstTpetraMultiVector(const Teuchos::RCP<const Thyra_MultiVector>& mv, bool const throw_if_not_tpetra = true);

Teuchos::RCP<Tpetra_Operator>
getTpetraOperator(const Teuchos::RCP<Thyra_LinearOp>& lop, bool const throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_Operator>
getConstTpetraOperator(const Teuchos::RCP<const Thyra_LinearOp>& lop, bool const throw_if_not_tpetra = true);

Teuchos::RCP<Tpetra_CrsMatrix>
getTpetraMatrix(const Teuchos::RCP<Thyra_LinearOp>& lop, bool const throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_CrsMatrix>
getConstTpetraMatrix(const Teuchos::RCP<const Thyra_LinearOp>& lop, bool const throw_if_not_tpetra = true);

// --- Conversion from references rather than RCPs --- //

Teuchos::RCP<Tpetra_Vector>
getTpetraVector(Thyra_Vector& v, bool const throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_Vector>
getConstTpetraVector(Thyra_Vector const& v, bool const throw_if_not_tpetra = true);

Teuchos::RCP<Tpetra_MultiVector>
getTpetraMultiVector(Thyra_MultiVector& mv, bool const throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_MultiVector>
getConstTpetraMultiVector(const Thyra_MultiVector& mv, bool const throw_if_not_tpetra = true);

Teuchos::RCP<Tpetra_Operator>
getTpetraOperator(Thyra_LinearOp& lop, bool const throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_Operator>
getConstTpetraOperator(const Thyra_LinearOp& lop, bool const throw_if_not_tpetra = true);

Teuchos::RCP<Tpetra_CrsMatrix>
getTpetraMatrix(Thyra_LinearOp& lop, bool const throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_CrsMatrix>
getConstTpetraMatrix(const Thyra_LinearOp& lop, bool const throw_if_not_tpetra = true);

}  // namespace Albany

#endif  // ALBANY_TPETRA_THYRA_UTILS_HPP
