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
 * \file   DTK_IntrepidCell.cpp
 * \author Stuart Slattery
 * \brief  Manager for intrepid cell-level operations.
 */
//---------------------------------------------------------------------------//

#include "DTK_DBC.hpp"

#include "DTK_IntrepidCell.hpp"

#include <Teuchos_as.hpp>

#include <Intrepid2_CellTools.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Intrepid2_FunctionSpaceTools.hpp>
#include <Intrepid2_Types.hpp>

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
IntrepidCell::IntrepidCell( const shards::CellTopology &cell_topology,
                            const unsigned degree )
    : d_topology( cell_topology )
    , d_cub_points( "d_cub_points" )
    , d_cub_weights( "d_cub_weights" )
    , d_jacobian( "d_jacobian" )
    , d_jacobian_det( "d_jacobian_det" )
    , d_weighted_measures( "d_weighted_measures" )
    , d_physical_ip_coordinates( "d_physical_ip_coordinates" )
{
    Intrepid2::DefaultCubatureFactory cub_factory;
    d_cubature = cub_factory.create<Kokkos::HostSpace, double, double>( d_topology, degree );

    unsigned num_cub_points = d_cubature->getNumPoints();
    unsigned cub_dim = cell_topology.getDimension();

    Kokkos::resize(d_cub_points, num_cub_points, cub_dim);
    Kokkos::resize(d_cub_weights, num_cub_points);
    d_cubature->getCubature( d_cub_points, d_cub_weights );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Destructor.
 */
IntrepidCell::~IntrepidCell() { /* ... */}

//---------------------------------------------------------------------------//
/*!
 * \brief Given physical coordinates for the cell nodes (Cell,Node,Dim),
 * assign them to the cell without allocating internal data.
 */
void IntrepidCell::setCellNodeCoordinates( const MDArray &cell_node_coords )
{
    d_cell_node_coords = cell_node_coords;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Given physical coordinates for the cell nodes (Cell,Node,Dim),
 * allocate the state of the cell object.
 */
void IntrepidCell::allocateCellState( const MDArray &cell_node_coords )
{
    // Store the cell node coords as the current state.
    setCellNodeCoordinates( cell_node_coords );

    // Get required dimensions.
    int num_cells = d_cell_node_coords.extent_int( 0 );
    int num_ip = d_cub_points.extent_int( 0 );
    int space_dim = d_cub_points.extent_int( 1 );

    // Resize arrays.
    Kokkos::resize(d_jacobian, num_cells, num_ip, space_dim, space_dim);
    Kokkos::resize(d_jacobian_det, num_cells, num_ip);
    Kokkos::resize(d_weighted_measures, num_cells, num_ip);
    Kokkos::resize(d_physical_ip_coordinates, num_cells, num_ip, space_dim);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Update the state of the cell object for the current cell node
 * coordinates.
 */
void IntrepidCell::updateCellState()
{
    // Compute the Jacobian.
    Intrepid2::CellTools<Kokkos::HostSpace>::setJacobian( d_jacobian, d_cub_points,
                                              d_cell_node_coords, d_topology );

    // Compute the determinant of the Jacobian.
    Intrepid2::CellTools<Kokkos::HostSpace>::setJacobianDet( d_jacobian_det, d_jacobian );

    // Compute the cell measures.
    Intrepid2::FunctionSpaceTools<Kokkos::HostSpace>::computeCellMeasure(
        d_weighted_measures, d_jacobian_det, d_cub_weights );

    // Compute physical frame integration point coordinates.
    Intrepid2::CellTools<Kokkos::HostSpace>::mapToPhysicalFrame(
        d_physical_ip_coordinates, d_cub_points, d_cell_node_coords,
        d_topology );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Free function for updating the cell state for a new set of
 * physical cells in a single call.
 */
void IntrepidCell::updateState( IntrepidCell &intrepid_cell,
                                const MDArray &cell_node_coords )
{
    intrepid_cell.allocateCellState( cell_node_coords );
    intrepid_cell.updateCellState();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Given a set of coordinates in the physical frame of the cell, map
 * them to the reference frame of the cell.
 */
void IntrepidCell::mapToCellReferenceFrame( const MDArray &physical_coords,
                                            MDArray &reference_coords )
{
    DTK_REQUIRE( 2 == physical_coords.rank() );
    DTK_REQUIRE( 2 == reference_coords.rank() );
    // Intrepid2 CellTools::mapToReferenceFrame expects rank-3 (C,P,D) inputs.
    // Wrap rank-2 (P,D) views with a leading cell dimension of 1.
    int num_points = physical_coords.extent_int(0);
    int space_dim = physical_coords.extent_int(1);
    MDArray phys_rank3("phys_rank3", 1, num_points, space_dim);
    MDArray ref_rank3("ref_rank3", 1, num_points, space_dim);
    for (int p = 0; p < num_points; ++p)
        for (int d = 0; d < space_dim; ++d)
            phys_rank3(0, p, d) = physical_coords(p, d);
    Intrepid2::CellTools<Kokkos::HostSpace>::mapToReferenceFrame(
        ref_rank3, phys_rank3, d_cell_node_coords, d_topology );
    for (int p = 0; p < num_points; ++p)
        for (int d = 0; d < space_dim; ++d)
            reference_coords(p, d) = ref_rank3(0, p, d);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Given a set of coordinates in the reference frame of the cell, map
 * them to the physical frame.
 */
void IntrepidCell::mapToCellPhysicalFrame( const MDArray &parametric_coords,
                                           MDArray &physical_coords )
{
    DTK_REQUIRE( 2 == parametric_coords.rank() );
    DTK_REQUIRE( 3 == physical_coords.rank() );
    DTK_REQUIRE( parametric_coords.extent_int( 1 ) ==
                 Teuchos::as<int>( d_topology.getDimension() ) );
    DTK_REQUIRE( physical_coords.extent_int( 0 ) ==
                 d_cell_node_coords.extent_int( 0 ) );
    DTK_REQUIRE( physical_coords.extent_int( 1 ) ==
                 parametric_coords.extent_int( 0 ) );
    DTK_REQUIRE( physical_coords.extent_int( 2 ) ==
                 Teuchos::as<int>( d_topology.getDimension() ) );

    Intrepid2::CellTools<Kokkos::HostSpace>::mapToPhysicalFrame(
        physical_coords, parametric_coords, d_cell_node_coords, d_topology );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Determine if a point given in parametric coordinates is inside of the
 * reference cell.
 */
bool IntrepidCell::pointInReferenceCell( const MDArray &reference_point,
                                         const double tolerance )
{
    // Intrepid2 checkPointInclusion expects a rank-1 point view.
    int sdim = reference_point.extent_int( 1 );
    Kokkos::DynRankView<Scalar, Kokkos::HostSpace> point( "point", sdim );
    for ( int d = 0; d < sdim; ++d )
    {
        point( d ) = reference_point( 0, d );
    }
    return Intrepid2::CellTools<Kokkos::HostSpace>::checkPointInclusion(
        point, d_topology, tolerance );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Determine if a point given in physical coordinates is inside of the
 * phyiscal cell.
 */
bool IntrepidCell::pointInPhysicalCell( const MDArray &point,
                                        const double tolerance )
{
    MDArray reference_point("reference_point", point.extent_int(0), point.extent_int(1));
    Kokkos::deep_copy(reference_point, 0.0);
    mapToCellReferenceFrame( point, reference_point );
    return pointInReferenceCell( reference_point, tolerance );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the number of cells in the current state.
 */
int IntrepidCell::getNumCells() const
{
    return d_weighted_measures.extent_int( 0 );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the number of cell points.
 */
int IntrepidCell::getNumIntegrationPoints() const
{
    return d_cub_points.extent_int( 0 );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the spatial dimension.
 */
int IntrepidCell::getSpatialDimension() const
{
    return d_cub_points.extent_int( 1 );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the cell measures (Cell). cell_measures must all ready be
 * allocated.
 */
void IntrepidCell::getCellMeasures( MDArray &cell_measures ) const
{
    DTK_REQUIRE( 1 == cell_measures.rank() );
    DTK_REQUIRE( cell_measures.extent_int( 0 ) ==
                 d_weighted_measures.extent_int( 0 ) );

    MDArray dofs( "dofs", d_cell_node_coords.extent_int( 0 ),
                  d_cub_weights.extent_int( 0 ) );
    Kokkos::deep_copy(dofs, 1.0);
    integrate( dofs, cell_measures );
}

//---------------------------------------------------------------------------//
// Get the physical cell point coordinates in each cell
// (Cell,IP,Dim).
void IntrepidCell::getPhysicalIntegrationCoordinates(
    MDArray &physical_ip_coordinates ) const
{
    physical_ip_coordinates = d_physical_ip_coordinates;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Given DOFs at the quadrature points {(Cell,Node) for scalar fields,
 * (Cell,Node,VecDim) for vector fields, and (Cell,Node,TensDim1,TensDim2) for
 * tensor fields.} perform the numerical integration in each cell by
 * contracting them with the weighted measures.
 */
void IntrepidCell::integrate( const MDArray &dofs, MDArray &integrals ) const
{
    Intrepid2::FunctionSpaceTools<Kokkos::HostSpace>::integrate(
        integrals, dofs, d_weighted_measures );
}

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//
// end DTK_IntrepidCell.cpp
//---------------------------------------------------------------------------//
