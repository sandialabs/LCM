#include "Albany_ThyraUtils.hpp"

#include "Albany_CommUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_Macros.hpp"
#include "Albany_ThyraCrsMatrixFactory.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_Utils.hpp"
#include "Teuchos_CompilerCodeTweakMacros.hpp"
#include "Teuchos_RCP.hpp"
#include "Thyra_DefaultSpmdVector.hpp"
#include "Thyra_DefaultSpmdVectorSpace.hpp"
#include "Thyra_VectorStdOps.hpp"

namespace Albany {

// ========= Vector Spaces utilities ========= //

Teuchos::RCP<Thyra_VectorSpace const>
createLocallyReplicatedVectorSpace(int const size, const Teuchos::RCP<Teuchos_Comm const> comm)
{
  Teuchos::RCP<const Tpetra_Map> tmap(new Tpetra_Map(size, 0, comm, Tpetra::LocalGlobal::LocallyReplicated));
  return createThyraVectorSpace(tmap);
}

Teuchos::RCP<Teuchos_Comm const>
getComm(Teuchos::RCP<Thyra_VectorSpace const> const& vs)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs, false);
  if (!tmap.is_null()) {
    return tmap->getComm();
  }
  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in getComm! Could not cast Thyra_VectorSpace to any of the "
      "supported concrete types.\n");
}

GO
getMaxAllGlobalIndex(Teuchos::RCP<Thyra_VectorSpace const> const& vs)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs, false);
  if (!tmap.is_null()) {
    return tmap->getMaxAllGlobalIndex();
  }
  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in getMaxAllGlobalIndex! Could not cast Thyra_VectorSpace to any "
      "of the supported concrete types.\n");
}

Teuchos::Array<GO>
getGlobalElements(Teuchos::RCP<Thyra_VectorSpace const> const& vs, const Teuchos::ArrayView<const LO>& lids)
{
  auto               indexer = createGlobalLocalIndexer(vs);
  Teuchos::Array<GO> gids(lids.size());
  for (LO i = 0; i < lids.size(); ++i) {
    gids[i] = indexer->getGlobalElement(lids[i]);
  }
  return gids;
}

Teuchos::Array<LO>
getLocalElements(Teuchos::RCP<Thyra_VectorSpace const> const& vs, const Teuchos::ArrayView<const GO>& gids)
{
  auto               indexer = createGlobalLocalIndexer(vs);
  Teuchos::Array<LO> lids(gids.size());
  for (LO i = 0; i < gids.size(); ++i) {
    lids[i] = indexer->getLocalElement(gids[i]);
  }
  return lids;
}

void
getGlobalElements(Teuchos::RCP<Thyra_VectorSpace const> const& vs, const Teuchos::ArrayView<GO>& gids)
{
  auto     indexer  = createGlobalLocalIndexer(vs);
  const LO localDim = indexer->getNumLocalElements();
  ALBANY_PANIC(gids.size() != localDim, "Error! ArrayView for gids not properly dimensioned.\n");

  for (LO i = 0; i < localDim; ++i) {
    gids[i] = indexer->getGlobalElement(i);
  }
}

Teuchos::Array<GO>
getGlobalElements(Teuchos::RCP<Thyra_VectorSpace const> const& vs)
{
  Teuchos::Array<GO> gids(getLocalSubdim(vs));
  getGlobalElements(vs, gids());
  return gids;
}

LO
getLocalSubdim(Teuchos::RCP<Thyra_VectorSpace const> const& vs)
{
  auto spmd_vs = getSpmdVectorSpace(vs);
  return spmd_vs->localSubDim();
}

bool
sameAs(Teuchos::RCP<Thyra_VectorSpace const> const& vs1, Teuchos::RCP<Thyra_VectorSpace const> const& vs2)
{
  auto tmap1 = getTpetraMap(vs1, false);
  if (!tmap1.is_null()) {
    // We don't allow two vs with different linear algebra back ends
    auto tmap2 = getTpetraMap(vs2, true);
    return tmap1->isSameAs(*tmap2);
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in sameAs! Could not cast Thyra_VectorSpace to any of the "
      "supported concrete types.\n");

  // Silence compiler warning
  TEUCHOS_UNREACHABLE_RETURN(false);
}

Teuchos::RCP<Thyra_VectorSpace const>
removeComponents(Teuchos::RCP<Thyra_VectorSpace const> const& vs, const Teuchos::ArrayView<const LO>& local_components)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs, false);
  if (!tmap.is_null()) {
    const LO num_node_lids         = tmap->getNodeNumElements();
    const LO num_reduced_node_lids = num_node_lids - local_components.size();
    ALBANY_PANIC(
        num_reduced_node_lids < 0,
        "Error in removeComponents! Cannot remove more components than are "
        "actually present.\n");
    Teuchos::Array<Tpetra_GO> reduced_gids(num_reduced_node_lids);
    for (LO lid = 0, k = 0; lid < num_node_lids; ++lid) {
      if (std::find(local_components.begin(), local_components.end(), lid) == local_components.end()) {
        reduced_gids[k] = tmap->getGlobalElement(lid);
        ++k;
      }
    }

    Tpetra::global_size_t          inv_gs = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    Teuchos::RCP<const Tpetra_Map> reduced_map(
        new Tpetra_Map(inv_gs, reduced_gids().getConst(), tmap->getIndexBase(), tmap->getComm()));

    return createThyraVectorSpace(reduced_map);
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in removeComponents! Could not cast Thyra_VectorSpace to any of "
      "the supported concrete types.\n");

  // Silence compiler warning
  TEUCHOS_UNREACHABLE_RETURN(Teuchos::null);
}

Teuchos::RCP<Thyra_VectorSpace const>
createSubspace(Teuchos::RCP<Thyra_VectorSpace const> const& vs, const Teuchos::ArrayView<const LO>& subspace_components)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs, false);
  if (!tmap.is_null()) {
    Teuchos::Array<Tpetra_GO> subspace_gids(subspace_components.size());
    int                       k = 0;
    for (auto lid : subspace_components) {
      subspace_gids[k] = tmap->getGlobalElement(lid);
      ++k;
    }

    Tpetra::global_size_t          inv_gs = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    Teuchos::RCP<const Tpetra_Map> reduced_map(
        new Tpetra_Map(inv_gs, subspace_gids().getConst(), tmap->getIndexBase(), tmap->getComm()));

    return createThyraVectorSpace(reduced_map);
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in createSubspace! Could not cast Thyra_VectorSpace to any of the "
      "supported concrete types.\n");

  // Silence compiler warning
  TEUCHOS_UNREACHABLE_RETURN(Teuchos::null);
}

// Create a vector space, given the ids of the space components
Teuchos::RCP<const Thyra_SpmdVectorSpace>
createVectorSpace(
    const Teuchos::RCP<Teuchos_Comm const>& comm,
    const Teuchos::ArrayView<const GO>&     gids,
    const GO                                globalDim)
{
  const GO            invalid           = Teuchos::OrdinalTraits<GO>::invalid();
  auto                gsi               = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  const decltype(gsi) numGlobalElements = (globalDim == invalid) ? gsi : static_cast<Tpetra_GO>(globalDim);
  Teuchos::ArrayView<const Tpetra_GO> tgids(reinterpret_cast<const Tpetra_GO*>(gids.getRawPtr()), gids.size());
  Teuchos::RCP<const Tpetra_Map>      tmap = Teuchos::rcp(new Tpetra_Map(numGlobalElements, tgids, 0, comm));
  return createThyraVectorSpace(tmap);
}

Teuchos::RCP<Thyra_VectorSpace const>
createVectorSpacesIntersection(
    Teuchos::RCP<Thyra_VectorSpace const> const& vs1,
    Teuchos::RCP<Thyra_VectorSpace const> const& vs2,
    const Teuchos::RCP<Teuchos_Comm const>&      comm)
{
  auto gids1 = getGlobalElements(vs1);
  auto gids2 = getGlobalElements(vs2);
  std::sort(gids1.begin(), gids1.end());
  std::sort(gids2.begin(), gids2.end());

  const auto min_size = std::min(gids1.size(), gids2.size());

  Teuchos::Array<GO> gids(min_size);
  const auto         it = std::set_intersection(gids1.begin(), gids1.end(), gids2.begin(), gids2.end(), gids.begin());
  gids.resize(std::distance(gids.begin(), it));

  return createVectorSpace(comm, gids);
}

Teuchos::RCP<Thyra_VectorSpace const>
createVectorSpacesDifference(
    Teuchos::RCP<Thyra_VectorSpace const> const& vs1,
    Teuchos::RCP<Thyra_VectorSpace const> const& vs2,
    const Teuchos::RCP<Teuchos_Comm const>&      comm)
{
  auto gids1 = getGlobalElements(vs1);
  auto gids2 = getGlobalElements(vs2);
  std::sort(gids1.begin(), gids1.end());
  std::sort(gids2.begin(), gids2.end());

  Teuchos::Array<GO> gids;
  std::set_difference(gids1.begin(), gids1.end(), gids2.begin(), gids2.end(), std::back_inserter(gids));

  return createVectorSpace(comm, gids);
}

// ========= Thyra_LinearOp utilities ========= //

Teuchos::RCP<Thyra_VectorSpace const>
getColumnSpace(const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    return createThyraVectorSpace(tmat->getColMap());
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in getColumnSpace! Could not cast Thyra_LinearOp to any of the "
      "supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  return Teuchos::null;
}

Teuchos::RCP<Thyra_VectorSpace const>
getRowSpace(const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    return createThyraVectorSpace(tmat->getRowMap());
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in getRowSpace! Could not cast Thyra_LinearOp to any of the "
      "supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  return Teuchos::null;
}

std::size_t
getNumEntriesInLocalRow(const Teuchos::RCP<const Thyra_LinearOp>& lop, const LO lrow)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    return tmat->getNumEntriesInLocalRow(lrow);
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in getNumEntriesInLocalRow! Could not cast Thyra_LinearOp to any "
      "of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  return Teuchos::null;
}

bool
isFillActive(const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    return tmat->isFillActive();
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in isFillActive! Could not cast Thyra_LinearOp to any of the "
      "supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  return false;
}

bool
isFillComplete(const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    return tmat->isFillComplete();
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in isFillComplete! Could not cast Thyra_LinearOp to any of the "
      "supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  return false;
}

void
resumeFill(const Teuchos::RCP<Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    tmat->resumeFill();
    return;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in resumeFill! Could not cast Thyra_LinearOp to any of the "
      "supported concrete types.\n");
}

void
fillComplete(const Teuchos::RCP<Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    tmat->fillComplete();
    return;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in fillComplete! Could not cast Thyra_LinearOp to any of the "
      "supported concrete types.\n");
}

void
assign(const Teuchos::RCP<Thyra_LinearOp>& lop, const ST value)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    // Tpetra throws when trying to set scalars in an already filled matrix
    bool callFillComplete = false;
    if (!tmat->isFillActive()) {
      tmat->resumeFill();
      callFillComplete = true;
    }

    tmat->setAllToScalar(value);

    if (callFillComplete) {
      tmat->fillComplete();
    }

    return;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in assign! Could not cast Thyra_LinearOp to any of the supported "
      "concrete types.\n");
}

void
getDiagonalCopy(const Teuchos::RCP<const Thyra_LinearOp>& lop, Teuchos::RCP<Thyra_Vector>& diag)
{
  // Diagonal makes sense only for (globally) square operators.
  // From Thyra, we can't check the global ids of the range/domain vector
  // spaces, but at least we can check that they have the same (global)
  // dimension.
  ALBANY_PANIC(
      lop->range()->dim() != lop->domain()->dim(),
      "Error in getDiagonalCopy! Attempt to take the diagonal of a non-square "
      "operator.\n");

  // If diag is not created, do it.
  if (diag.is_null()) {
    diag = Thyra::createMember(lop->range());
  }

  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    tmat->getLocalDiagCopy(*Albany::getTpetraVector(diag, true));
    return;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in getDiagonalCopy! Could not cast Thyra_LinearOp to any of the "
      "supported concrete types.\n");
}

void
scale(const Teuchos::RCP<Thyra_LinearOp>& lop, const ST val)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    tmat->scale(val);
    return;
  }
  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in scale! Could not cast Thyra_LinearOp to any of the supported "
      "concrete types.\n");
}
void
getLocalRowValues(
    const Teuchos::RCP<Thyra_LinearOp>& lop,
    const LO                            lrow,
    Teuchos::Array<LO>&                 indices,
    Teuchos::Array<ST>&                 values)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    auto numEntries = tmat->getNumEntriesInLocalRow(lrow);
    indices.resize(numEntries);
    values.resize(numEntries);
    tmat->getLocalRowCopy(lrow, indices, values, numEntries);
    return;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in getLocalRowValues! Could not cast Thyra_LinearOp to any of the "
      "supported concrete types.\n");
}

int
addToLocalRowValues(
    const Teuchos::RCP<Thyra_LinearOp>& lop,
    const LO                            lrow,
    const Teuchos::ArrayView<const LO>  indices,
    const Teuchos::ArrayView<const ST>  values)
{
  // The following is an integer error code, to be returned by this
  // routine if something doesn't go right.  0 means success, 1 means failure
  int integer_error_code = 0;
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    auto returned_val = tmat->sumIntoLocalValues(lrow, indices, values);
    // std::cout << "IKT returned_val, indices size = " << returned_val << ", "
    // << indices.size() << std::endl;
    ALBANY_ASSERT(
        returned_val != -1,
        "Error: addToLocalRowValues returned -1, meaning linear op is not "
        "fillActive \n"
            << "or does not have an underlying non-null static graph!\n");
    // Tpetra's replaceLocalValues routine returns the number of indices for
    // which values were actually replaced; the number of "correct" indices.
    // This should be size of indices array.  Therefore if returned_val !=
    // indices.size() something went wrong
    if (returned_val != indices.size()) integer_error_code = 1;
    return integer_error_code;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in addToLocalRowValues! Could not cast Thyra_LinearOp to any of "
      "the supported concrete types.\n");
}

void
insertGlobalValues(
    const Teuchos::RCP<Thyra_LinearOp>& lop,
    const GO                            grow,
    const Teuchos::ArrayView<const GO>  cols,
    const Teuchos::ArrayView<const ST>  values)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    const Tpetra_GO                     tgrow = grow;
    Teuchos::ArrayView<const Tpetra_GO> tcols(reinterpret_cast<const Tpetra_GO*>(cols.getRawPtr()), cols.size());
    tmat->insertGlobalValues(tgrow, tcols, values);
    return;
  }
}

void
replaceGlobalValues(
    const Teuchos::RCP<Thyra_LinearOp>& lop,
    const GO                            gid,
    const Teuchos::ArrayView<const GO>  indices,
    const Teuchos::ArrayView<const ST>  values)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    const Tpetra_GO                     tgid = gid;
    Teuchos::ArrayView<const Tpetra_GO> tindices(
        reinterpret_cast<const Tpetra_GO*>(indices.getRawPtr()), indices.size());
    tmat->replaceGlobalValues(tgid, tindices, values);
    return;
  }
  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in replaceGlobalValues! Could not cast Thyra_LinearOp to any of "
      "the supported concrete types.\n");
}

int
addToGlobalRowValues(
    const Teuchos::RCP<Thyra_LinearOp>& lop,
    const GO                            grow,
    const Teuchos::ArrayView<const GO>  indices,
    const Teuchos::ArrayView<const ST>  values)
{
  // The following is an integer error code, to be returned by this
  // routine if something doesn't go right.  0 means success, 1 means failure
  int integer_error_code = 0;
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    const Tpetra_GO                     tgrow = grow;
    Teuchos::ArrayView<const Tpetra_GO> tindices(
        reinterpret_cast<const Tpetra_GO*>(indices.getRawPtr()), indices.size());
    auto returned_val = tmat->sumIntoGlobalValues(tgrow, tindices, values);
    // std::cout << "IKT returned_val, indices size = " << returned_val << ", "
    // << indices.size() << std::endl;
    ALBANY_ASSERT(
        returned_val != -1,
        "Error: addToGlobalRowValues returned -1, meaning linear op is not "
        "fillActive \n"
            << "or does not have an underlying non-null static graph!\n");
    // Tpetra's replaceGlobalValues routine returns the number of indices for
    // which values were actually replaced; the number of "correct" indices.
    // This should be size of indices array.  Therefore if returned_val !=
    // indices.size() something went wrong
    if (returned_val != indices.size()) integer_error_code = 1;
    return integer_error_code;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in addToGlobalRowValues! Could not cast Thyra_LinearOp to any of "
      "the supported concrete types.\n");
}

void
setLocalRowValues(
    const Teuchos::RCP<Thyra_LinearOp>& lop,
    const LO                            lrow,
    const Teuchos::ArrayView<const LO>  indices,
    const Teuchos::ArrayView<const ST>  values)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    tmat->replaceLocalValues(lrow, indices, values);
    return;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in setLocalRowValues! Could not cast Thyra_LinearOp to any of the "
      "supported concrete types.\n");
}

void
setLocalRowValues(const Teuchos::RCP<Thyra_LinearOp>& lop, const LO lrow, const Teuchos::ArrayView<const ST> values)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    Teuchos::ArrayView<const LO> indices;
    tmat->getGraph()->getLocalRowView(lrow, indices);
    ALBANY_PANIC(
        indices.size() != values.size(),
        "Error! This routine is meant for setting *all* values in a row, "
        "but the length of the input values array does not match the number of "
        "indices in the local row.\n");
    tmat->replaceLocalValues(lrow, indices, values);
    return;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in setLocalRowValues! Could not cast Thyra_LinearOp to any of the "
      "supported concrete types.\n");
}

int
getGlobalMaxNumRowEntries(const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    auto return_value = tmat->getGlobalMaxNumRowEntries();
    return return_value;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in getGlobalMaxNumRowEntries! Could not cast Thyra_LinearOp to "
      "any of the supported concrete types.\n");
}

bool
isStaticGraph(const Teuchos::RCP<Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    return tmat->isStaticGraph();
  }
  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in isStaticGraph! Could not cast Thyra_LinearOp to any of the "
      "supported concrete types.\n");
}

bool
isStaticGraph(const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    return tmat->isStaticGraph();
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in isStaticGraph! Could not cast Thyra_LinearOp to any of the "
      "supported concrete types.\n");
}

// The following routine creates a one-to-one version of the given Map where
// each GID lives on only one process. Therefore it is an owned (unique) map.
Teuchos::RCP<Thyra_VectorSpace const>
createOneToOneVectorSpace(Teuchos::RCP<Thyra_VectorSpace const> const vs)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs, false);
  if (!tmap.is_null()) {
    const Teuchos::RCP<const Tpetra_Map> map = Tpetra::createOneToOne(tmap);
    return createThyraVectorSpace(map);
  }
  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in createOneToOneVectorSpace! Could not cast Thyra_VectorSpace to "
      "any of the supported concrete types.\n");
}

Teuchos::RCP<const Thyra_LinearOp>
buildRestrictionOperator(
    Teuchos::RCP<Thyra_VectorSpace const> const& space,
    Teuchos::RCP<Thyra_VectorSpace const> const& subspace)
{
  // In the process, verify the that subspace is a subspace of space
  auto space_indexer    = createGlobalLocalIndexer(space);
  auto subspace_indexer = createGlobalLocalIndexer(subspace);

  ThyraCrsMatrixFactory factory(space, subspace);

  int const localSubDim = subspace_indexer->getNumLocalElements();
  for (LO lid = 0; lid < localSubDim; ++lid) {
    const GO gid = subspace_indexer->getGlobalElement(lid);
    ALBANY_PANIC(
        space_indexer->isLocallyOwnedElement(gid),
        "Error in buildRestrictionOperator! The input 'subspace' is not a "
        "subspace of the input 'space'.\n");
    factory.insertGlobalIndices(gid, Teuchos::arrayView(&gid, 1));
  }

  factory.fillComplete();
  Teuchos::RCP<Thyra_LinearOp> P = factory.createOp();
  assign(P, 1.0);

  return P;
}

Teuchos::RCP<const Thyra_LinearOp>
buildProlongationOperator(
    Teuchos::RCP<Thyra_VectorSpace const> const& space,
    Teuchos::RCP<Thyra_VectorSpace const> const& subspace)
{
  // In the process, verify the that subspace is a subspace of space
  auto space_indexer    = createGlobalLocalIndexer(space);
  auto subspace_indexer = createGlobalLocalIndexer(subspace);

  ThyraCrsMatrixFactory factory(subspace, space);

  int const localSubDim = subspace_indexer->getNumLocalElements();
  for (LO lid = 0; lid < localSubDim; ++lid) {
    const GO gid = subspace_indexer->getGlobalElement(lid);
    ALBANY_PANIC(
        space_indexer->isLocallyOwnedElement(gid),
        "Error in buildProlongationOperator! The input 'subspace' is not a "
        "subspace of the input 'space'.\n");
    factory.insertGlobalIndices(gid, Teuchos::arrayView(&gid, 1));
  }

  factory.fillComplete();
  Teuchos::RCP<Thyra_LinearOp> P = factory.createOp();
  assign(P, 1.0);

  return P;
}

double
computeConditionNumber(const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  double condest = std::numeric_limits<double>::quiet_NaN();
  // Dummy return value to silence compiler warning
  return condest;
}

DeviceLocalMatrix<const ST>
getDeviceData(Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    // Get the local matrix from tpetra.
    DeviceLocalMatrix<const ST> data = tmat->getLocalMatrix();
    return data;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in getDeviceData! Could not cast Thyra_Vector to any of the "
      "supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  DeviceLocalMatrix<const ST> dummy;
  return dummy;
}
template <int I>
struct ShowMeI
{
};

DeviceLocalMatrix<ST>
getNonconstDeviceData(Teuchos::RCP<Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop, false);
  if (!tmat.is_null()) {
    // Get the local matrix from tpetra.
    DeviceLocalMatrix<ST> data = tmat->getLocalMatrix();
    return data;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in getNonconstDeviceData! Could not cast Thyra_Vector to any of "
      "the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  DeviceLocalMatrix<ST> dummy;
  return dummy;
}

// ========= Thyra_Vector utilities ========== //

Teuchos::ArrayRCP<ST>
getNonconstLocalData(Teuchos::RCP<Thyra_Vector> const& v)
{
  Teuchos::ArrayRCP<ST> vals;

  // Allow failure, since we don't know what the underlying linear algebra is
  // Note: we do tpetra separately since it need to handle device/copy sync.
  //       everything else, we assume it inherits from SpmdVectorBase.
  auto tv = getTpetraVector(v, false);
  if (!tv.is_null()) {
    // Tpetra
    vals = tv->get1dViewNonConst();
  } else {
    // Thyra::SpmdVectorBase
    auto spmd_v = Teuchos::rcp_dynamic_cast<Thyra::SpmdVectorBase<ST>>(v);
    if (!spmd_v.is_null()) {
      spmd_v->getNonconstLocalData(Teuchos::outArg(vals));
    } else {
      // If all the tries above are unsuccessful, throw an error.
      ALBANY_ABORT(
          "Error in getNnconstLocalData! Could not cast Thyra_Vector to any of "
          "the supported concrete types.\n");
    }
  }

  return vals;
}

Teuchos::ArrayRCP<const ST>
getLocalData(Teuchos::RCP<Thyra_Vector const> const& v)
{
  Teuchos::ArrayRCP<const ST> vals;

  // Allow failure, since we don't know what the underlying linear algebra is
  // Note: we do tpetra separately since it need to handle device/copy sync.
  //       everything else, we assume it inherits from SpmdVectorBase.
  auto tv = getConstTpetraVector(v, false);
  if (!tv.is_null()) {
    // Tpetra
    vals = tv->get1dView();
  } else {
    // Thyra::SpmdVectorBase
    auto spmd_v = Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<ST>>(v);
    if (!spmd_v.is_null()) {
      spmd_v->getLocalData(Teuchos::outArg(vals));
    } else {
      // If all the tries above are unsuccessful, throw an error.
      ALBANY_ABORT(
          "Error in getLocalData! Could not cast Thyra_Vector to any of the "
          "supported concrete types.\n");
    }
  }

  return vals;
}

int
getNumVectors(const Teuchos::RCP<const Thyra_MultiVector>& mv)
{
  auto tv = getConstTpetraMultiVector(mv, false);
  if (!tv.is_null()) {
    return tv->getNumVectors();
  }
  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in getNumVectors! Could not cast Thyra_MultiVector to any of the "
      "supported concrete types.\n");
}

Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>
getNonconstLocalData(Teuchos::RCP<Thyra_MultiVector> const& mv)
{
  if (mv.is_null()) {
    return Teuchos::null;
  }

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> data(mv->domain()->dim());
  for (int i = 0; i < mv->domain()->dim(); ++i) {
    data[i] = getNonconstLocalData(mv->col(i));
  }
  return data;
}

Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>>
getLocalData(const Teuchos::RCP<const Thyra_MultiVector>& mv)
{
  if (mv.is_null()) {
    return Teuchos::null;
  }

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> data(mv->domain()->dim());
  for (int i = 0; i < mv->domain()->dim(); ++i) {
    data[i] = getLocalData(mv->col(i));
  }
  return data;
}

Teuchos::ArrayRCP<ST>
getNonconstLocalData(Thyra_Vector& v)
{
  Teuchos::ArrayRCP<ST> vals;

  // Allow failure, since we don't know what the underlying linear algebra is
  // Note: we do tpetra separately since it need to handle device/copy sync.
  //       everything else, we assume it inherits from SpmdVectorBase.
  auto tv = getTpetraVector(v, false);
  if (!tv.is_null()) {
    // Tpetra
    vals = tv->get1dViewNonConst();
  } else {
    // Thyra::SpmdVectorBase
    auto* spmd_v = dynamic_cast<Thyra::SpmdVectorBase<ST>*>(&v);
    if (spmd_v != nullptr) {
      spmd_v->getNonconstLocalData(Teuchos::outArg(vals));
    } else {
      // If all the tries above are unsuccessful, throw an error.
      ALBANY_ABORT(
          "Error in getNonconstLocalData! Could not cast Thyra_Vector to any "
          "of the supported concrete types.\n");
    }
  }

  return vals;
}

Teuchos::ArrayRCP<const ST>
getLocalData(Thyra_Vector const& v)
{
  Teuchos::ArrayRCP<const ST> vals;

  // Allow failure, since we don't know what the underlying linear algebra is
  // Note: we do tpetra separately since it need to handle device/copy sync.
  //       everything else, we assume it inherits from SpmdVectorBase.
  auto tv = getConstTpetraVector(v, false);
  if (!tv.is_null()) {
    // Tpetra
    vals = tv->get1dView();
  } else {
    // Thyra::SpmdVectorBase
    auto* spmd_v = dynamic_cast<const Thyra::SpmdVectorBase<ST>*>(&v);
    if (spmd_v != nullptr) {
      spmd_v->getLocalData(Teuchos::outArg(vals));
    } else {
      // If all the tries above are unsuccessful, throw an error.
      ALBANY_ABORT(
          "Error in getLocalData! Could not cast Thyra_Vector to any of the "
          "supported concrete types.\n");
    }
  }

  return vals;
}

Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>
getNonconstLocalData(Thyra_MultiVector& mv)
{
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> data(mv.domain()->dim());
  for (int i = 0; i < mv.domain()->dim(); ++i) {
    data[i] = getNonconstLocalData(mv.col(i));
  }
  return data;
}

Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>>
getLocalData(const Thyra_MultiVector& mv)
{
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> data(mv.domain()->dim());
  for (int i = 0; i < mv.domain()->dim(); ++i) {
    data[i] = getLocalData(mv.col(i));
  }
  return data;
}

DeviceView1d<const ST>
getDeviceData(Teuchos::RCP<Thyra_Vector const> const& v)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tv = getConstTpetraVector(v, false);
  if (!tv.is_null()) {
    auto                   data2d = tv->getLocalView<KokkosNode::execution_space>();
    DeviceView1d<const ST> data   = Kokkos::subview(data2d, Kokkos::ALL(), 0);
    return data;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in getDeviceData! Could not cast Thyra_Vector to any of the "
      "supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  DeviceView1d<const ST> dummy;
  return dummy;
}

DeviceView1d<ST>
getNonconstDeviceData(Teuchos::RCP<Thyra_Vector> const& v)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tv = getTpetraVector(v, false);
  if (!tv.is_null()) {
    auto             data2d = tv->getLocalView<KokkosNode::execution_space>();
    DeviceView1d<ST> data   = Kokkos::subview(data2d, Kokkos::ALL(), 0);
    return data;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in getNonconstDeviceData! Could not cast Thyra_Vector to any of "
      "the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  DeviceView1d<ST> dummy;
  return dummy;
}

void
scale_and_update(
    Teuchos::RCP<Thyra_Vector> const       y,
    const ST                               y_coeff,
    Teuchos::RCP<Thyra_Vector const> const x,
    const ST                               x_coeff)
{
  Thyra::V_StVpStV(y.ptr(), x_coeff, *x, y_coeff, *y);
}

ST
mean(Teuchos::RCP<Thyra_Vector const> const& v)
{
  return Thyra::sum(*v) / v->space()->dim();
}

Teuchos::Array<ST>
means(const Teuchos::RCP<const Thyra_MultiVector>& mv)
{
  int const          numVecs = mv->domain()->dim();
  Teuchos::Array<ST> vals(numVecs);
  for (int i = 0; i < numVecs; ++i) {
    vals[i] = mean(mv->col(i));
  }

  return vals;
}

// ======== I/O utilities ========= //

template <>
void
describe<Thyra_VectorSpace>(
    Teuchos::RCP<Thyra_VectorSpace const> const& vs,
    Teuchos::FancyOStream&                       out,
    const Teuchos::EVerbosityLevel               verbLevel)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tvs = getTpetraMap(vs, false);
  if (!tvs.is_null()) {
    tvs->describe(out, verbLevel);
    return;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in describe! Could not cast Thyra_Vector to any of the supported "
      "concrete types.\n");
}

template <>
void
describe<Thyra_Vector>(
    Teuchos::RCP<Thyra_Vector const> const& v,
    Teuchos::FancyOStream&                  out,
    const Teuchos::EVerbosityLevel          verbLevel)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tv = getConstTpetraVector(v, false);
  if (!tv.is_null()) {
    tv->describe(out, verbLevel);
    return;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in describe! Could not cast Thyra_Vector to any of the supported "
      "concrete types.\n");
}

template <>
void
describe<Thyra_LinearOp>(
    const Teuchos::RCP<const Thyra_LinearOp>& op,
    Teuchos::FancyOStream&                    out,
    const Teuchos::EVerbosityLevel            verbLevel)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto top = getConstTpetraOperator(op, false);
  if (!top.is_null()) {
    top->describe(out, verbLevel);
    return;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in describe! Could not cast Thyra_Vector to any of the supported "
      "concrete types.\n");
}

// ========= Matrix Market utilities ========== //

// These routines implement a specialization of the template functions declared
// in Albany_Utils.hpp

template <>
void
writeMatrixMarket<Thyra_Vector const>(
    Teuchos::RCP<Thyra_Vector const> const& v,
    std::string const&                      prefix,
    int const                               counter)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tv = getConstTpetraVector(v, false);
  if (!tv.is_null()) {
    writeMatrixMarket(tv, prefix, counter);
    return;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in writeMatrixMarket! Could not cast Thyra_Vector to any of the "
      "supported concrete types.\n");
}

template <>
void
writeMatrixMarket<Thyra_Vector>(Teuchos::RCP<Thyra_Vector> const& v, std::string const& prefix, int const counter)
{
  writeMatrixMarket(v.getConst(), prefix, counter);
}

template <>
void
writeMatrixMarket<const Thyra_MultiVector>(
    const Teuchos::RCP<const Thyra_MultiVector>& mv,
    std::string const&                           prefix,
    int const                                    counter)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmv = getConstTpetraMultiVector(mv, false);
  if (!tmv.is_null()) {
    writeMatrixMarket(tmv, prefix, counter);
    return;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in writeMatrixMarket! Could not cast Thyra_Vector to any of the "
      "supported concrete types.\n");
}

template <>
void
writeMatrixMarket<Thyra_MultiVector>(
    Teuchos::RCP<Thyra_MultiVector> const& mv,
    std::string const&                     prefix,
    int const                              counter)
{
  writeMatrixMarket(mv.getConst(), prefix, counter);
}

template <>
void
writeMatrixMarket<const Thyra_LinearOp>(
    const Teuchos::RCP<const Thyra_LinearOp>& A,
    std::string const&                        prefix,
    int const                                 counter)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tA = getConstTpetraMatrix(A, false);
  if (!tA.is_null()) {
    writeMatrixMarket(tA, prefix, counter);
    return;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in writeMatrixMarket! Could not cast Thyra_LinearOp to any of the "
      "supported concrete types.\n");
}

template <>
void
writeMatrixMarket<Thyra_LinearOp>(const Teuchos::RCP<Thyra_LinearOp>& A, std::string const& prefix, int const counter)
{
  writeMatrixMarket(A.getConst(), prefix, counter);
}

// These routines implement a specialization of the template functions declared
// in Albany_Utils.hpp
template <>
void
writeMatrixMarket<Thyra_VectorSpace const>(
    Teuchos::RCP<Thyra_VectorSpace const> const& vs,
    std::string const&                           prefix,
    int const                                    counter)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tm = getTpetraMap(vs, false);
  if (!tm.is_null()) {
    writeMatrixMarket(tm, prefix, counter);
    return;
  }

  // If all the tries above are unsuccessful, throw an error.
  ALBANY_ABORT(
      "Error in writeMatrixMarket! Could not cast Thyra_VectorSpace to any of "
      "the supported concrete types.\n");
}

template <>
void
writeMatrixMarket<Thyra_VectorSpace>(
    const Teuchos::RCP<Thyra_VectorSpace>& vs,
    std::string const&                     prefix,
    int const                              counter)
{
  writeMatrixMarket(vs.getConst(), prefix, counter);
}

// ========= Thyra_SpmdXYZ utilities ========== //

Teuchos::RCP<const Thyra_SpmdVectorSpace>
getSpmdVectorSpace(Teuchos::RCP<Thyra_VectorSpace const> const vs, bool const throw_on_failure)
{
  Teuchos::RCP<const Thyra_SpmdVectorSpace> spmd_vs;
  spmd_vs = Teuchos::rcp_dynamic_cast<const Thyra_SpmdVectorSpace>(vs, throw_on_failure);
  return spmd_vs;
}

// ========= Thyra_ProductXYZ utilities ========== //

Teuchos::RCP<const Thyra_ProductVectorSpace>
getProductVectorSpace(Teuchos::RCP<Thyra_VectorSpace const> const vs, bool const throw_on_failure)
{
  Teuchos::RCP<const Thyra_ProductVectorSpace> pvs;
  pvs = Teuchos::rcp_dynamic_cast<const Thyra_ProductVectorSpace>(vs, throw_on_failure);
  return pvs;
}

Teuchos::RCP<Thyra_ProductVector>
getProductVector(Teuchos::RCP<Thyra_Vector> const v, bool const throw_on_failure)
{
  Teuchos::RCP<Thyra_ProductVector> pv;
  pv = Teuchos::rcp_dynamic_cast<Thyra_ProductVector>(v, throw_on_failure);
  return pv;
}

Teuchos::RCP<const Thyra_ProductVector>
getConstProductVector(Teuchos::RCP<Thyra_Vector const> const v, bool const throw_on_failure)
{
  Teuchos::RCP<const Thyra_ProductVector> pv;
  pv = Teuchos::rcp_dynamic_cast<const Thyra_ProductVector>(v, throw_on_failure);
  return pv;
}

Teuchos::RCP<Thyra_ProductMultiVector>
getProductMultiVector(Teuchos::RCP<Thyra_MultiVector> const mv, bool const throw_on_failure)
{
  Teuchos::RCP<Thyra_ProductMultiVector> pmv;
  pmv = Teuchos::rcp_dynamic_cast<Thyra_ProductMultiVector>(mv, throw_on_failure);
  return pmv;
}

Teuchos::RCP<const Thyra_ProductMultiVector>
getConstProductMultiVector(const Teuchos::RCP<const Thyra_MultiVector> mv, bool const throw_on_failure)
{
  Teuchos::RCP<const Thyra_ProductMultiVector> pmv;
  pmv = Teuchos::rcp_dynamic_cast<const Thyra_ProductMultiVector>(mv, throw_on_failure);
  return pmv;
}

}  // namespace Albany
