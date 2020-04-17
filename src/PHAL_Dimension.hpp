// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_DIMENSION_HPP
#define PHAL_DIMENSION_HPP

#include "Phalanx_ExtentTraits.hpp"
#include "Shards_Array.hpp"

struct Dim : public shards::ArrayDimTag
{
  Dim(){};
  char const*
  name() const;
  static Dim const&
  tag();
};

struct VecDim : public shards::ArrayDimTag
{
  VecDim(){};
  char const*
  name() const;
  static VecDim const&
  tag();
};

struct LayerDim : public shards::ArrayDimTag
{
  LayerDim(){};
  char const*
  name() const;
  static LayerDim const&
  tag();
};

struct QuadPoint : public shards::ArrayDimTag
{
  QuadPoint(){};
  char const*
  name() const;
  static QuadPoint const&
  tag();
};

struct Node : public shards::ArrayDimTag
{
  Node(){};
  char const*
  name() const;
  static Node const&
  tag();
};

struct Vertex : public shards::ArrayDimTag
{
  Vertex(){};
  char const*
  name() const;
  static Vertex const&
  tag();
};

struct Point : public shards::ArrayDimTag
{
  Point(){};
  char const*
  name() const;
  static Point const&
  tag();
};

struct Cell : public shards::ArrayDimTag
{
  Cell(){};
  char const*
  name() const;
  static Cell const&
  tag();
};

struct Side : public shards::ArrayDimTag
{
  Side(){};
  char const*
  name() const;
  static Side const&
  tag();
};

struct Dummy : public shards::ArrayDimTag
{
  Dummy(){};
  char const*
  name() const;
  static Dummy const&
  tag();
};

namespace PHX {
template <>
struct is_extent<Dim> : std::true_type
{
};
template <>
struct is_extent<LayerDim> : std::true_type
{
};
template <>
struct is_extent<VecDim> : std::true_type
{
};
template <>
struct is_extent<QuadPoint> : std::true_type
{
};
template <>
struct is_extent<Node> : std::true_type
{
};
template <>
struct is_extent<Vertex> : std::true_type
{
};
template <>
struct is_extent<Point> : std::true_type
{
};
template <>
struct is_extent<Cell> : std::true_type
{
};
template <>
struct is_extent<Side> : std::true_type
{
};
template <>
struct is_extent<Dummy> : std::true_type
{
};
}  // namespace PHX

#endif
