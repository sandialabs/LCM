// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_ALBANYTRAITS_HPP
#define PHAL_ALBANYTRAITS_HPP

#include "Sacado_mpl_find.hpp"
#include "Sacado_mpl_vector.hpp"

// traits Base Class
#include "Phalanx_Traits.hpp"

// Include User Data Types
#include "Albany_SacadoTypes.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Setup.hpp"
#include "Phalanx_KokkosDeviceTypes.hpp"
#include "Phalanx_Print.hpp"
#include "Phalanx_config.hpp"

//! PHalanx-ALbany Code base: templated evaluators for Sacado AD
namespace PHAL {

typedef PHX::Device::size_type size_type;

// Forward declaration since Workset needs AlbanyTraits
struct Workset;

// From a ScalarT, determine the ScalarRefT.
template <typename T>
struct Ref
{
  typedef T& type;
};
template <typename T>
struct RefKokkos
{
  typedef typename Kokkos::View<T*, PHX::Device>::reference_type type;
};
template <>
struct Ref<FadType> : RefKokkos<FadType>
{
};
#if defined(ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE)
template <>
struct Ref<TanFadType> : RefKokkos<TanFadType>
{
};
#endif

struct AlbanyTraits : public PHX::TraitsBase
{
  // ******************************************************************
  // *** Evaluation Types
  //   * ScalarT is for quantities that depend on solution/params
  //   * MeshScalarT is for quantities that depend on mesh coords only
  // ******************************************************************
  template <typename ScalarT_, typename MeshScalarT_, typename ParamScalarT_>
  struct EvaluationType
  {
    typedef ScalarT_      ScalarT;
    typedef MeshScalarT_  MeshScalarT;
    typedef ParamScalarT_ ParamScalarT;
  };

  struct Residual : EvaluationType<RealType, RealType, RealType>
  {
  };
#if defined(ALBANY_MESH_DEPENDS_ON_SOLUTION) && \
    defined(ALBANY_PARAMETERS_DEPEND_ON_SOLUTION)
  struct Jacobian : EvaluationType<FadType, FadType, FadType>
  {
  };
#elif defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
  struct Jacobian : EvaluationType<FadType, FadType, FadType>
  {
  };
#elif defined(ALBANY_PARAMETERS_DEPEND_ON_SOLUTION)
  struct Jacobian : EvaluationType<FadType, RealType, FadType>
  {
  };
#else
  struct Jacobian : EvaluationType<FadType, RealType, RealType>
  {
  };
#endif

  using EvalTypes  = Sacado::mpl::vector<Residual, Jacobian>;
  using BEvalTypes = Sacado::mpl::vector<Residual, Jacobian>;

  // ******************************************************************
  // *** Allocator Type
  // ******************************************************************
  //   typedef PHX::NewAllocator Allocator;
  // typedef PHX::ContiguousAllocator<RealType> Allocator;

  // ******************************************************************
  // *** User Defined Object Passed in for Evaluation Method
  // ******************************************************************
  typedef Setup&   SetupData;
  typedef Workset& EvalData;
  typedef Workset& PreEvalData;
  typedef Workset& PostEvalData;
};
}  // namespace PHAL

namespace PHX {
// Evaluation Types
template <>
inline std::string
print<PHAL::AlbanyTraits::Residual>()
{
  return "<Residual>";
}

template <>
inline std::string
print<PHAL::AlbanyTraits::Jacobian>()
{
  return "<Jacobian>";
}

// ******************************************************************
// *** Data Types
// ******************************************************************

// Create the data types for each evaluation type

#define DECLARE_EVAL_SCALAR_TYPES(EvalType, Type1, Type2) \
  template <>                                             \
  struct eval_scalar_types<PHAL::AlbanyTraits::EvalType>  \
  {                                                       \
    typedef Sacado::mpl::vector<Type1, Type2> type;       \
  };

template <>
struct eval_scalar_types<PHAL::AlbanyTraits::Residual>
{
  typedef Sacado::mpl::vector<RealType> type;
};
DECLARE_EVAL_SCALAR_TYPES(Jacobian, FadType, RealType)

#undef DECLARE_EVAL_SCALAR_TYPES
}  // namespace PHX

// Define macros for explicit template instantiation

// 1. Basic cases: depend only on EvalT and Traits
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits>;

// 2. Versatile cases: after EvalT and Traits, accept any number of args
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_RESIDUAL(name, ...) \
  template class name<                                                      \
      PHAL::AlbanyTraits::Residual,                                         \
      PHAL::AlbanyTraits,                                                   \
      __VA_ARGS__>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_JACOBIAN(name, ...) \
  template class name<                                                      \
      PHAL::AlbanyTraits::Jacobian,                                         \
      PHAL::AlbanyTraits,                                                   \
      __VA_ARGS__>;

// 3. Scalar dependent cases: after EvalT and Traits, accept one or two scalar
// types
//    NOTE: *always* allow RealType for the scalar type(s)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_RESIDUAL(name) \
  template class name<                                                      \
      PHAL::AlbanyTraits::Residual,                                         \
      PHAL::AlbanyTraits,                                                   \
      RealType>;

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_JACOBIAN(name) \
  template class name<                                                      \
      PHAL::AlbanyTraits::Jacobian,                                         \
      PHAL::AlbanyTraits,                                                   \
      FadType>;                                                             \
  template class name<                                                      \
      PHAL::AlbanyTraits::Jacobian,                                         \
      PHAL::AlbanyTraits,                                                   \
      RealType>;

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_RESIDUAL(name) \
  template class name<                                                       \
      PHAL::AlbanyTraits::Residual,                                          \
      PHAL::AlbanyTraits,                                                    \
      RealType,                                                              \
      RealType>;

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_JACOBIAN(name) \
  template class name<                                                       \
      PHAL::AlbanyTraits::Jacobian,                                          \
      PHAL::AlbanyTraits,                                                    \
      FadType,                                                               \
      RealType>;                                                             \
  template class name<                                                       \
      PHAL::AlbanyTraits::Jacobian,                                          \
      PHAL::AlbanyTraits,                                                    \
      RealType,                                                              \
      RealType>;                                                             \
  template class name<                                                       \
      PHAL::AlbanyTraits::Jacobian,                                          \
      PHAL::AlbanyTraits,                                                    \
      FadType,                                                               \
      FadType>;                                                              \
  template class name<                                                       \
      PHAL::AlbanyTraits::Jacobian,                                          \
      PHAL::AlbanyTraits,                                                    \
      RealType,                                                              \
      FadType>;

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_THREE_SCALAR_TYPES_RESIDUAL(name) \
  template class name<                                                         \
      PHAL::AlbanyTraits::Residual,                                            \
      PHAL::AlbanyTraits,                                                      \
      RealType,                                                                \
      RealType,                                                                \
      RealType>;

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_THREE_SCALAR_TYPES_JACOBIAN(name) \
  template class name<                                                         \
      PHAL::AlbanyTraits::Jacobian,                                            \
      PHAL::AlbanyTraits,                                                      \
      FadType,                                                                 \
      RealType,                                                                \
      RealType>;                                                               \
  template class name<                                                         \
      PHAL::AlbanyTraits::Jacobian,                                            \
      PHAL::AlbanyTraits,                                                      \
      RealType,                                                                \
      RealType,                                                                \
      RealType>;                                                               \
  template class name<                                                         \
      PHAL::AlbanyTraits::Jacobian,                                            \
      PHAL::AlbanyTraits,                                                      \
      FadType,                                                                 \
      FadType,                                                                 \
      RealType>;                                                               \
  template class name<                                                         \
      PHAL::AlbanyTraits::Jacobian,                                            \
      PHAL::AlbanyTraits,                                                      \
      RealType,                                                                \
      FadType,                                                                 \
      RealType>;                                                               \
  template class name<                                                         \
      PHAL::AlbanyTraits::Jacobian,                                            \
      PHAL::AlbanyTraits,                                                      \
      FadType,                                                                 \
      RealType,                                                                \
      FadType>;                                                                \
  template class name<                                                         \
      PHAL::AlbanyTraits::Jacobian,                                            \
      PHAL::AlbanyTraits,                                                      \
      RealType,                                                                \
      RealType,                                                                \
      FadType>;                                                                \
  template class name<                                                         \
      PHAL::AlbanyTraits::Jacobian,                                            \
      PHAL::AlbanyTraits,                                                      \
      FadType,                                                                 \
      FadType,                                                                 \
      FadType>;                                                                \
  template class name<                                                         \
      PHAL::AlbanyTraits::Jacobian,                                            \
      PHAL::AlbanyTraits,                                                      \
      RealType,                                                                \
      FadType,                                                                 \
      FadType>;

// 4. Input-output scalar type case: similar to the above one with two scalar
// types.
//    However, the output scalar type MUST be constructible from the input one,
//    so certain combinations are not allowed.
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_RESIDUAL(name) \
  template class name<                                                         \
      PHAL::AlbanyTraits::Residual,                                            \
      PHAL::AlbanyTraits,                                                      \
      RealType,                                                                \
      RealType>;

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_JACOBIAN(name) \
  template class name<                                                         \
      PHAL::AlbanyTraits::Jacobian,                                            \
      PHAL::AlbanyTraits,                                                      \
      RealType,                                                                \
      RealType>;                                                               \
  template class name<                                                         \
      PHAL::AlbanyTraits::Jacobian,                                            \
      PHAL::AlbanyTraits,                                                      \
      RealType,                                                                \
      FadType>;                                                                \
  template class name<                                                         \
      PHAL::AlbanyTraits::Jacobian,                                            \
      PHAL::AlbanyTraits,                                                      \
      FadType,                                                                 \
      FadType>;

// 5. General macros: you should call these in your cpp files,
//    which in turn will call the ones above.
#define PHAL_INSTANTIATE_TEMPLATE_CLASS(name)    \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name) \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name)

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE(name)    \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_RESIDUAL(name) \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_JACOBIAN(name)

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES(name)    \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_RESIDUAL(name) \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_JACOBIAN(name)

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_THREE_SCALAR_TYPES(name)    \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_THREE_SCALAR_TYPES_RESIDUAL(name) \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_THREE_SCALAR_TYPES_JACOBIAN(name)

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES(name)    \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_RESIDUAL(name) \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_JACOBIAN(name)

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(name, ...)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_RESIDUAL(name, __VA_ARGS__) \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_JACOBIAN(name, __VA_ARGS__)

#include "PHAL_Workset.hpp"

#endif  // PHAL_ALBANYTRAITS_HPP
