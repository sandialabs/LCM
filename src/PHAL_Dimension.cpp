// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "PHAL_Dimension.hpp"

char const*
Dim::name() const
{
  static char const n[] = "Dim";
  return n;
}
Dim const&
Dim::tag()
{
  static Dim const myself;
  return myself;
}

char const*
VecDim::name() const
{
  static char const n[] = "VecDim";
  return n;
}
VecDim const&
VecDim::tag()
{
  static VecDim const myself;
  return myself;
}

char const*
LayerDim::name() const
{
  static char const n[] = "LayerDim";
  return n;
}
LayerDim const&
LayerDim::tag()
{
  static LayerDim const myself;
  return myself;
}

char const*
QuadPoint::name() const
{
  static char const n[] = "QuadPoint";
  return n;
}
QuadPoint const&
QuadPoint::tag()
{
  static QuadPoint const myself;
  return myself;
}

char const*
Node::name() const
{
  static char const n[] = "Node";
  return n;
}
const Node&
Node::tag()
{
  static const Node myself;
  return myself;
}

char const*
Vertex::name() const
{
  static char const n[] = "Vertex";
  return n;
}
const Vertex&
Vertex::tag()
{
  static const Vertex myself;
  return myself;
}

char const*
Point::name() const
{
  static char const n[] = "Point";
  return n;
}
const Point&
Point::tag()
{
  static const Point myself;
  return myself;
}

char const*
Cell::name() const
{
  static char const n[] = "Cell";
  return n;
}
const Cell&
Cell::tag()
{
  static const Cell myself;
  return myself;
}

char const*
Side::name() const
{
  static char const n[] = "Side";
  return n;
}
const Side&
Side::tag()
{
  static const Side myself;
  return myself;
}

char const*
Dummy::name() const
{
  static char const n[] = "Dummy";
  return n;
}
const Dummy&
Dummy::tag()
{
  static const Dummy myself;
  return myself;
}
