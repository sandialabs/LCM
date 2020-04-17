// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_NODALGRAPHUTILS_HPP
#define ALBANY_NODALGRAPHUTILS_HPP

namespace Albany {

/*!
 * \brief Various utilities for the construction of an STK nodal graph
 *
 */

std::size_t const hex_table[] = {1, 3, 4, 0, 2, 5, 1, 3, 6, 0, 2, 7,
                                 0, 5, 7, 1, 4, 6, 2, 5, 7, 3, 4, 6};

std::size_t const hex_nconnect = 3;

std::size_t const tet_table[] = {1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2};

std::size_t const tet_nconnect = 3;

std::size_t const quad_table[] = {1, 3, 0, 2, 1, 3, 0, 2};

std::size_t const quad_nconnect = 2;

std::size_t const tri_table[] = {1, 2, 0, 2, 0, 1};

std::size_t const tri_nconnect = 2;

}  // namespace Albany

#endif  // ALBANY_NODALGRAPHUTILS_HPP
