//---------------------------------------------------------------------------//
/*
  Copyright (c) 2012, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the University of Wisconsin - Madison nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//---------------------------------------------------------------------------//
/*!
 * \brief DTK_IntrepidCellLocalMap.cpp
 * \author Stuart R. Slattery
 * \brief Helper functions for implementing the local map interface with
 * intrepid.
 */
//---------------------------------------------------------------------------//

#include "DTK_IntrepidCellLocalMap.hpp"
#include "DTK_DBC.hpp"
#include "DTK_IntrepidCell.hpp"
#include "DTK_ProjectionPrimitives.hpp"

#include <Kokkos_DynRankView.hpp>

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
// Return the entity measure with respect to the parameteric dimension (volume
// for a 3D entity, area for 2D, and length for 1D).
double IntrepidCellLocalMap::measure(
    const shards::CellTopology &entity_topo,
    const Kokkos::DynRankView<double, Kokkos::HostSpace> &entity_coords )
{
    // Get the Intrepid cell corresponding to the entity topology.
    IntrepidCell entity_cell( entity_topo, 1 );

    // Update thet state of the cell.
    IntrepidCell::updateState( entity_cell, entity_coords );

    // Compute the measure of the cell.
    Kokkos::DynRankView<double, Kokkos::HostSpace> measure( "measure", 1 );
    entity_cell.getCellMeasures( measure );
    return measure( 0 );
}

//---------------------------------------------------------------------------//
// Return the centroid of the entity.
void IntrepidCellLocalMap::centroid(
    const shards::CellTopology &entity_topo,
    const Kokkos::DynRankView<double, Kokkos::HostSpace> &entity_coords,
    const Teuchos::ArrayView<double> &centroid )
{
    // Get the Intrepid cell corresponding to the entity topology.
    IntrepidCell entity_cell( entity_topo, 1 );
    entity_cell.setCellNodeCoordinates( entity_coords );

    // Get the reference center of the cell.
    int sdim = entity_coords.extent_int( 2 );
    Kokkos::DynRankView<double, Kokkos::HostSpace> ref_center(
        "ref_center", 1, sdim );
    ProjectionPrimitives::referenceCellCenter( entity_topo, ref_center );

    // Map the cell center to the physical frame.
    Kokkos::DynRankView<double, Kokkos::HostSpace> phys_center(
        "phys_center", 1, 1, sdim );
    entity_cell.mapToCellPhysicalFrame( ref_center, phys_center );

    // Extract the centroid coordinates.
    for ( int d = 0; d < sdim; ++d )
    {
        centroid[d] = phys_center( 0, 0, d );
    }
}

//---------------------------------------------------------------------------//
// Map a point to the reference space of an entity. Return the parameterized
// point.
bool IntrepidCellLocalMap::mapToReferenceFrame(
    const shards::CellTopology &entity_topo,
    const Kokkos::DynRankView<double, Kokkos::HostSpace> &entity_coords,
    const Teuchos::ArrayView<const double> &point,
    const Teuchos::ArrayView<double> &reference_point )
{
    // Get the Intrepid cell corresponding to the entity topology.
    IntrepidCell entity_cell( entity_topo, 1 );
    entity_cell.setCellNodeCoordinates( entity_coords );

    // Map the point to the reference frame of the cell.
    int sdim = entity_coords.extent_int( 2 );
    Kokkos::DynRankView<double, Kokkos::HostSpace> point_container(
        "point_container", 1, sdim );
    for ( int d = 0; d < sdim; ++d )
    {
        point_container( 0, d ) = point[d];
    }
    Kokkos::DynRankView<double, Kokkos::HostSpace> ref_point_container(
        "ref_point_container", 1, sdim );
    entity_cell.mapToCellReferenceFrame( point_container, ref_point_container );
    for ( int d = 0; d < sdim; ++d )
    {
        reference_point[d] = ref_point_container( 0, d );
    }

    // Return true to indicate successful mapping. Catching Intrepid errors
    // and returning false is a possibility here.
    return true;
}

//---------------------------------------------------------------------------//
// Determine if a reference point is in the parameterized space of an entity.
bool IntrepidCellLocalMap::checkPointInclusion(
    const shards::CellTopology &entity_topo,
    const Teuchos::ArrayView<const double> &reference_point,
    const double tolerance )
{
    // Get the Intrepid cell corresponding to the entity topology.
    IntrepidCell entity_cell( entity_topo, 1 );

    // Check point inclusion.
    int sdim = reference_point.size();
    Kokkos::DynRankView<double, Kokkos::HostSpace> ref_point_container(
        "ref_point_container", 1, sdim );
    for ( int d = 0; d < sdim; ++d )
    {
        ref_point_container( 0, d ) = reference_point[d];
    }
    return entity_cell.pointInReferenceCell( ref_point_container, tolerance );
}

//---------------------------------------------------------------------------//
// Map a reference point to the physical space of an entity.
void IntrepidCellLocalMap::mapToPhysicalFrame(
    const shards::CellTopology &entity_topo,
    const Kokkos::DynRankView<double, Kokkos::HostSpace> &entity_coords,
    const Teuchos::ArrayView<const double> &reference_point,
    const Teuchos::ArrayView<double> &point )
{
    // Get the Intrepid cell corresponding to the entity topology.
    IntrepidCell entity_cell( entity_topo, 1 );
    entity_cell.setCellNodeCoordinates( entity_coords );

    // Map the reference point to the physical frame of the cell.
    int sdim = entity_coords.extent_int( 2 );
    Kokkos::DynRankView<double, Kokkos::HostSpace> ref_point_container(
        "ref_point_container", 1, sdim );
    for ( int d = 0; d < sdim; ++d )
    {
        ref_point_container( 0, d ) = reference_point[d];
    }
    Kokkos::DynRankView<double, Kokkos::HostSpace> point_container(
        "point_container", 1, 1, sdim );
    entity_cell.mapToCellPhysicalFrame( ref_point_container, point_container );
    for ( int d = 0; d < sdim; ++d )
    {
        point[d] = point_container( 0, 0, d );
    }
}

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//
// end DTK_IntrepidCellLocalMap.cpp
//---------------------------------------------------------------------------//
