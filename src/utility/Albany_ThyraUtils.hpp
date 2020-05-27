#ifndef ALBANY_THYRA_UTILS_HPP
#define ALBANY_THYRA_UTILS_HPP

// Get basic Thyra types
#include "Albany_ThyraTypes.hpp"

// Get Teuchos_Comm type
#include "Albany_CommTypes.hpp"

// Get Kokkos types (for the 1d device view)
#include "Albany_KokkosTypes.hpp"

namespace Albany {

// ========= Vector Spaces utilities ========= //

Teuchos::RCP<Thyra_VectorSpace const>
createLocallyReplicatedVectorSpace(int const size, const Teuchos::RCP<Teuchos_Comm const> comm);

Teuchos::RCP<Teuchos_Comm const>
getComm(Teuchos::RCP<Thyra_VectorSpace const> const& vs);
Teuchos::Array<GO>
getGlobalElements(Teuchos::RCP<Thyra_VectorSpace const> const& vs, const Teuchos::ArrayView<const LO>& lids);
Teuchos::Array<LO>
getLocalElements(Teuchos::RCP<Thyra_VectorSpace const> const& vs, const Teuchos::ArrayView<const GO>& gids);
void
getGlobalElements(Teuchos::RCP<Thyra_VectorSpace const> const& vs, const Teuchos::ArrayView<GO>& gids);
LO
getLocalSubdim(Teuchos::RCP<Thyra_VectorSpace const> const& vs);

Teuchos::Array<GO>
getGlobalElements(Teuchos::RCP<Thyra_VectorSpace const> const& vs);

// Check if two vector spaces are indeed the same vector space
bool
sameAs(Teuchos::RCP<Thyra_VectorSpace const> const& vs1, Teuchos::RCP<Thyra_VectorSpace const> const& vs2);

// Remove a set of local components from a vector space
Teuchos::RCP<Thyra_VectorSpace const>
removeComponents(Teuchos::RCP<Thyra_VectorSpace const> const& vs, const Teuchos::ArrayView<const LO>& local_components);

// The complement of the above: the specified components are the ones to keep
Teuchos::RCP<Thyra_VectorSpace const>
createSubspace(
    Teuchos::RCP<Thyra_VectorSpace const> const& vs,
    const Teuchos::ArrayView<const LO>&          subspace_components);

// Create a vector space, given the ids of the space components
Teuchos::RCP<const Thyra_SpmdVectorSpace>
createVectorSpace(
    const Teuchos::RCP<Teuchos_Comm const>& comm,
    const Teuchos::ArrayView<const GO>&     gids,
    const GO                                globalDim = Teuchos::OrdinalTraits<GO>::invalid());

// Intersects vectors spaces
Teuchos::RCP<Thyra_VectorSpace const>
createVectorSpacesIntersection(
    Teuchos::RCP<Thyra_VectorSpace const> const& vs1,
    Teuchos::RCP<Thyra_VectorSpace const> const& vs2,
    const Teuchos::RCP<Teuchos_Comm const>&      comm);

// Complement of a vector spaces to another
// NOTE: elements in vs2 that are not in vs1 are ignored.
Teuchos::RCP<Thyra_VectorSpace const>
createVectorSpacesDifference(
    Teuchos::RCP<Thyra_VectorSpace const> const& vs1,
    Teuchos::RCP<Thyra_VectorSpace const> const& vs2,
    const Teuchos::RCP<Teuchos_Comm const>&      comm);

// ========= Thyra_LinearOp utilities ========= //

// These routines help to manipulate the a Thyra_LinearOp. They are needed
// so we can abstract from the concrete linear algebra package, and rely
// only on the Thyra interfaces.

Teuchos::RCP<Thyra_VectorSpace const>
getColumnSpace(const Teuchos::RCP<const Thyra_LinearOp>& lop);

Teuchos::RCP<Thyra_VectorSpace const>
getRowSpace(const Teuchos::RCP<const Thyra_LinearOp>& lop);

std::size_t
getNumEntriesInLocalRow(const Teuchos::RCP<const Thyra_LinearOp>& lop, const LO lrow);

// Fill related helpers
bool
isFillActive(const Teuchos::RCP<const Thyra_LinearOp>& lop);
void
resumeFill(const Teuchos::RCP<Thyra_LinearOp>& lop);
bool
isFillComplete(const Teuchos::RCP<const Thyra_LinearOp>& lop);
void
fillComplete(const Teuchos::RCP<Thyra_LinearOp>& lop);

// Entries manipulation helpers
void
assign(const Teuchos::RCP<Thyra_LinearOp>& lop, const ST value);
void
getDiagonalCopy(const Teuchos::RCP<const Thyra_LinearOp>& lop, Teuchos::RCP<Thyra_Vector>& diag);
void
getLocalRowValues(
    const Teuchos::RCP<Thyra_LinearOp>& lop,
    const LO                            lrow,
    Teuchos::Array<LO>&                 indices,
    Teuchos::Array<ST>&                 values);

void
setLocalRowValues(
    const Teuchos::RCP<Thyra_LinearOp>& lop,
    const LO                            lrow,
    const Teuchos::ArrayView<const LO>  indices,
    const Teuchos::ArrayView<const ST>  values);
void
setLocalRowValues(const Teuchos::RCP<Thyra_LinearOp>& lop, const LO lrow, const Teuchos::ArrayView<const ST> values);

int
addToLocalRowValues(
    const Teuchos::RCP<Thyra_LinearOp>& lop,
    const LO                            lrow,
    const Teuchos::ArrayView<const LO>  indices,
    const Teuchos::ArrayView<const ST>  values);

int
addToGlobalRowValues(
    const Teuchos::RCP<Thyra_LinearOp>& lop,
    const GO                            grow,
    const Teuchos::ArrayView<const GO>  indices,
    const Teuchos::ArrayView<const ST>  values);

void
replaceGlobalValues(
    const Teuchos::RCP<Thyra_LinearOp>& lop,
    const GO                            gid,
    const Teuchos::ArrayView<const GO>  indices,
    const Teuchos::ArrayView<const ST>  values);

void
insertGlobalValues(
    const Teuchos::RCP<Thyra_LinearOp>& lop,
    const GO                            grow,
    const Teuchos::ArrayView<const GO>  cols,
    const Teuchos::ArrayView<const ST>  values);

int
getGlobalMaxNumRowEntries(const Teuchos::RCP<const Thyra_LinearOp>& lop);

void
scale(const Teuchos::RCP<Thyra_LinearOp>& lop, const ST val);

Teuchos::RCP<Thyra_VectorSpace const>
createOneToOneVectorSpace(Teuchos::RCP<Thyra_VectorSpace const> const vs);

bool
isStaticGraph(const Teuchos::RCP<Thyra_LinearOp>& lop);
bool
isStaticGraph(const Teuchos::RCP<const Thyra_LinearOp>& lop);

Teuchos::RCP<const Thyra_LinearOp>
buildRestrictionOperator(
    Teuchos::RCP<Thyra_VectorSpace const> const& space,
    Teuchos::RCP<Thyra_VectorSpace const> const& subspace);

Teuchos::RCP<const Thyra_LinearOp>
buildProlongationOperator(
    Teuchos::RCP<Thyra_VectorSpace const> const& space,
    Teuchos::RCP<Thyra_VectorSpace const> const& subspace);

// Math properties helpers
double
computeConditionNumber(const Teuchos::RCP<const Thyra_LinearOp>& lop);

// Get a kokkos compatible object to view the content of the linear op on device
DeviceLocalMatrix<const ST>
getDeviceData(Teuchos::RCP<const Thyra_LinearOp>& lop);
DeviceLocalMatrix<ST>
getNonconstDeviceData(Teuchos::RCP<Thyra_LinearOp>& lop);

// ========= Thyra_(Multi)Vector utilities ========= //

// Const and nonconst version of a getter of local data in a Thyra
// vector/multivector
Teuchos::ArrayRCP<ST>
getNonconstLocalData(Teuchos::RCP<Thyra_Vector> const& v);
Teuchos::ArrayRCP<const ST>
getLocalData(Teuchos::RCP<Thyra_Vector const> const& v);
Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>
getNonconstLocalData(Teuchos::RCP<Thyra_MultiVector> const& mv);
Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>>
getLocalData(const Teuchos::RCP<const Thyra_MultiVector>& mv);

Teuchos::ArrayRCP<ST>
getNonconstLocalData(Thyra_Vector& v);
Teuchos::ArrayRCP<const ST>
getLocalData(Thyra_Vector const& v);
Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>
getNonconstLocalData(Thyra_MultiVector& mv);
Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>>
getLocalData(const Thyra_MultiVector& mv);

DeviceView1d<const ST>
getDeviceData(Teuchos::RCP<Thyra_Vector const> const& v);
DeviceView1d<ST>
getNonconstDeviceData(Teuchos::RCP<Thyra_Vector> const& v);

int
getNumVectors(const Teuchos::RCP<const Thyra_MultiVector>& mv);

// This is just a utility routine, that mildly extend the update method of
// Thyra_Vector, but does not have the complex signature of the
// linear_combination method of Thyra_Vector. In fact, the update method only
// allows to do y = y + alpha*x, while often one wants to do y = beta*y +
// alpha*x. The linear_combination method offers that capability, but the
// signature is more cumbersome. Thyra offers also a free function with a
// lighter signature for a linear_combination of 2 vectors, but its name is
// V_StVpStV, which can be a bit arcane. Here we simply wrap that function in
// one with a nicer name. Performs y = y_coeff*y + x_coeff*x;
void
scale_and_update(
    Teuchos::RCP<Thyra_Vector> const       y,
    const ST                               y_coeff,
    Teuchos::RCP<Thyra_Vector const> const x,
    const ST                               x_coeff);

// Thyra does not offer a 'mean' method in its (multi)vector interface.
// The method 'sum' in Thyra_VectorStdOps already does the sum,
// so here we simply scale by the vector (global) length.
ST
mean(Teuchos::RCP<Thyra_Vector const> const& v);
Teuchos::Array<ST>
means(const Teuchos::RCP<const Thyra_MultiVector>& mv);

// ======== I/O utilities ========= //

template <typename ThyraObjectType>
void
describe(
    const Teuchos::RCP<const ThyraObjectType>& obj,
    Teuchos::FancyOStream&                     out,
    const Teuchos::EVerbosityLevel             verbLevel);

// ========= Thyra_SpmdXYZ utilities ========== //

// These routines help to manipulate Thyra pointers, casting them to
// proper derived classes, and checking that the result is nonnull.

Teuchos::RCP<const Thyra_SpmdVectorSpace>
getSpmdVectorSpace(Teuchos::RCP<Thyra_VectorSpace const> const vs, bool const throw_on_failure = true);

// ========= Thyra_ProductXYZ utilities ========== //

// These routines help to manipulate Thyra pointers, casting them to
// proper derived classes, and checking that the result is nonnull.

Teuchos::RCP<const Thyra_ProductVectorSpace>
getProductVectorSpace(Teuchos::RCP<Thyra_VectorSpace const> const vs, bool const throw_on_failure = true);

Teuchos::RCP<Thyra_ProductVector>
getProductVector(Teuchos::RCP<Thyra_Vector> const v, bool const throw_on_failure = true);

Teuchos::RCP<const Thyra_ProductVector>
getConstProductVector(Teuchos::RCP<Thyra_Vector const> const v, bool const throw_on_failure = true);

Teuchos::RCP<Thyra_ProductMultiVector>
getProductMultiVector(Teuchos::RCP<Thyra_MultiVector> const mv, bool const throw_on_failure = true);

Teuchos::RCP<const Thyra_ProductMultiVector>
getConstProductMultiVector(const Teuchos::RCP<const Thyra_MultiVector> mv, bool const throw_on_failure = true);

}  // namespace Albany

#endif  // ALBANY_THYRA_UTILS_HPP
