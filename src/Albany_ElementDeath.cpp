// @HEADER
// *****************************************************************************
//                           Albany Package
//
// Copyright 2016 NTESS and the Albany contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Albany_ElementDeath.hpp"

#include <Teuchos_Assert.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <vector>

namespace Albany {

namespace {

using DoubleField = stk::mesh::Field<double>;

DoubleField*
getScratchField(const stk::mesh::MetaData& meta,
                stk::mesh::EntityRank rank,
                const std::string& name)
{
  auto* field = meta.get_field<double>(rank, name);
  TEUCHOS_TEST_FOR_EXCEPTION(
      field == nullptr, std::runtime_error,
      "applyElementDeath: scratch field '" << name
      << "' not registered. It should be declared in "
         "Albany_GenericSTKMeshStruct.cpp alongside activePart/deadCellsPart.");
  return field;
}

void
fieldFillZero(stk::mesh::BulkData& bulk, DoubleField& field)
{
  const auto& meta = bulk.mesh_meta_data();
  const auto& buckets = bulk.get_buckets(field.entity_rank(), meta.universal_part());
  for (const auto* bucket : buckets) {
    double* data = stk::mesh::field_data(field, *bucket);
    if (data != nullptr) {
      std::fill(data, data + bucket->size(), 0.0);
    }
  }
}

} // namespace

bool
applyElementDeath(
    stk::mesh::BulkData&            bulkData,
    stk::mesh::Part&                activePart,
    stk::mesh::Part&                deadCellsPart,
    const stk::mesh::EntityVector&  killed,
    const stk::mesh::PartVector&    sideSetParts,
    const stk::mesh::PartVector&    /*boundarySideSetParts*/)
{
  // Clone-before-disconnect element death (Adagio-style port).
  // Phases match doc/element_death_port.md.

  const auto& meta = bulkData.mesh_meta_data();
  const auto side_rank = meta.side_rank();

  auto* faceElemAttachCount =
      getScratchField(meta, side_rank, "deathFaceElemAttachCount");
  auto* faceExposureCount =
      getScratchField(meta, side_rank, "deathFaceExposureCount");
  auto* boundaryFaceMarker =
      getScratchField(meta, side_rank, "deathBoundaryFaceMarker");

  const auto& localPart = meta.locally_owned_part();
  const stk::mesh::Selector liveLocal =
      stk::mesh::Selector(localPart) &
      stk::mesh::Selector(activePart) &
      !stk::mesh::Selector(deadCellsPart);

  // Phase 2: count how many elements reference each face, then
  // parallel-sum so shared faces see the global total. The count is
  // taken with the dying cells STILL in the live selector (they
  // haven't been moved to deadCellsPart yet), so a face shared
  // between two dying cells on the same rank has count == 2 and is
  // treated as "shared" -- safe: it'll be cloned per dying side
  // (Phase 3) and then deleted when both exposure markers fire
  // (Phase 5).
  fieldFillZero(bulkData, *faceElemAttachCount);
  const auto& liveBuckets =
      bulkData.get_buckets(stk::topology::ELEM_RANK, liveLocal);
  for (const auto* bucket : liveBuckets) {
    for (size_t i = 0; i < bucket->size(); ++i) {
      stk::mesh::Entity elem = (*bucket)[i];
      const stk::mesh::Entity* faces = bulkData.begin(elem, side_rank);
      const unsigned nf = bulkData.num_connectivity(elem, side_rank);
      for (unsigned j = 0; j < nf; ++j) {
        double* count = stk::mesh::field_data(*faceElemAttachCount, faces[j]);
        if (count != nullptr) {
          *count += 1.0;
        }
      }
    }
  }
  {
    std::vector<const stk::mesh::FieldBase*> fields{faceElemAttachCount};
    stk::mesh::parallel_sum(bulkData, fields);
  }

  // Phase 3: clone-and-disconnect inside a single modification block.
  // For each dying cell's shared face: detach the dying cell, declare
  // a new locally-owned face on its side, bump the original face's
  // exposure marker, AND set the boundary marker so Phase 5 can add
  // the original face to sideSetParts on its owner rank (STK then
  // propagates the part change to the live-side ghost, keeping the
  // -erodible node-set consistent on both ranks -- otherwise DBCs on
  // the eroding surface land on the dying-rank's nodes but not the
  // live-rank's copies, and the solution drifts after a few steps).
  // Unshared exterior faces are left alone. Then move the dying
  // cells into deadCellsPart (Step B1 of the LCM convention).
  fieldFillZero(bulkData, *faceExposureCount);
  fieldFillZero(bulkData, *boundaryFaceMarker);

  bulkData.modification_begin();
  for (stk::mesh::Entity elem : killed) {
    // Snapshot the relation list before we mutate it.
    const stk::mesh::Entity* faces_ptr = bulkData.begin(elem, side_rank);
    const stk::mesh::ConnectivityOrdinal* ords_ptr =
        bulkData.begin_ordinals(elem, side_rank);
    const unsigned nf = bulkData.num_connectivity(elem, side_rank);
    std::vector<stk::mesh::Entity> elem_faces(faces_ptr, faces_ptr + nf);
    std::vector<stk::mesh::ConnectivityOrdinal> elem_face_ords(
        ords_ptr, ords_ptr + nf);

    for (unsigned j = 0; j < nf; ++j) {
      const stk::mesh::Entity face = elem_faces[j];
      const unsigned side_ord = static_cast<unsigned>(elem_face_ords[j]);
      const double* count =
          stk::mesh::field_data(*faceElemAttachCount, face);
      const bool isShared = count != nullptr && (*count) > 1.01;

      if (isShared) {
        bulkData.destroy_relation(elem, face, side_ord);
        bulkData.declare_element_side(elem, side_ord, sideSetParts);
        double* exposure =
            stk::mesh::field_data(*faceExposureCount, face);
        if (exposure != nullptr) {
          *exposure += 1.0;
        }
        double* boundaryMark =
            stk::mesh::field_data(*boundaryFaceMarker, face);
        if (boundaryMark != nullptr) {
          *boundaryMark = 1.0;
        }
      }
      // Unshared (boundary) faces of the dying cell are left attached
      // and unmodified. LCM keeps dying cells in the mesh
      // (deadCellsPart), so destroying their exclusive faces would
      // lose data. The original process_killed_elements only iterated
      // the dying cell's element-graph neighbors (shared interior
      // faces), so it never touched these unshared exterior faces
      // either -- there is nothing to mirror for them.
    }
  }
  for (stk::mesh::Entity elem : killed) {
    bulkData.change_entity_parts(
        elem,
        stk::mesh::PartVector{&deadCellsPart},
        stk::mesh::PartVector{});
  }
  bulkData.modification_end();

  // Phase 4: parallel-assemble the exposure and boundary markers so
  // every rank sees the global totals for shared faces.
  {
    std::vector<const stk::mesh::FieldBase*> fields{
        faceExposureCount, boundaryFaceMarker};
    stk::mesh::parallel_sum(bulkData, fields);
  }

  // Phase 5: collect owned faces that need part-membership updates
  // (boundary marker > 0) or destruction (exposure > 1.01), then do
  // both inside a single modification block.
  //
  // A face whose exposure marker hit 2 has had BOTH sides drop their
  // back-reference (dying-cell relations on each rank), so it now
  // has zero live relations on any rank. destroy_entity here doesn't
  // trip the multi-rank STK harmonization bug the recon avoids.
  //
  // A face whose boundary marker is > 0 but exposure is 1 has its
  // dying-side relation gone but a live-side cell still uses it. Add
  // it to sideSetParts so the live-side ghost (which STK propagates
  // the part change to via the owner) sees it as a death-boundary
  // face for DBC purposes.
  //
  // Only the owner rank acts; ghost copies are kept consistent by
  // STK's standard modification_end propagation.
  std::vector<stk::mesh::Entity> toAddToSideSetParts;
  std::vector<stk::mesh::Entity> toDelete;
  {
    const auto& faceBuckets =
        bulkData.get_buckets(side_rank, meta.universal_part());
    for (const auto* bucket : faceBuckets) {
      if (!bucket->owned()) continue;
      for (size_t i = 0; i < bucket->size(); ++i) {
        stk::mesh::Entity face = (*bucket)[i];
        const double* exposure =
            stk::mesh::field_data(*faceExposureCount, face);
        const double* boundaryMark =
            stk::mesh::field_data(*boundaryFaceMarker, face);
        const bool isExposed =
            exposure != nullptr && (*exposure) > 1.01;
        const bool isBoundary =
            boundaryMark != nullptr && (*boundaryMark) > 0.5;
        if (isExposed) {
          toDelete.push_back(face);
        } else if (isBoundary) {
          toAddToSideSetParts.push_back(face);
        }
      }
    }
  }

  bulkData.modification_begin();
  for (stk::mesh::Entity face : toAddToSideSetParts) {
    bulkData.change_entity_parts(
        face, sideSetParts, stk::mesh::PartVector{});
  }
  for (stk::mesh::Entity face : toDelete) {
    bulkData.destroy_entity(face);
  }
  bulkData.modification_end();

  return true;
}

} // namespace Albany
