/*--------------------------------------------------------------------*/
/*    Copyright 2001, 2002 National Technology & Engineering Solutions of
 * Sandia, LLC (NTESS)                        */
/*    Under the terms of Contract DE-AC04-94AL85000, there is a       */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

// Copyright 2001, 2002 National Technology & Engineering Solutions of Sandia,
// LLC (NTESS)

#include <percept/stk_rebalance/Partition.hpp>
#include <stdexcept>

using namespace stk;
namespace stk {
using namespace rebalance;
}

Partition::Partition(ParallelMachine comm) : comm_(comm) {}
Partition::~Partition() {}
