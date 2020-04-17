// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "AAdapt_RC_Field.hpp"

#include "AAdapt_RC_DataTypes_impl.hpp"
#include "AAdapt_RC_Manager.hpp"
#include "Albany_Layouts.hpp"
#include "MiniTensor_Tensor.h"
#include "Phalanx_MDField.hpp"

namespace AAdapt {
namespace rc {

template <int rank>
Field<rank>::Field() : valid_(false)
{
}

template <int rank>
bool
Field<rank>::init(Teuchos::ParameterList const& p, std::string const& name)
{
  std::string const name_rc      = Manager::decorate(name),
                    name_rc_name = name_rc + " Name";
  valid_                         = p.isType<std::string>(name_rc_name);
  if (!valid_) return false;
  f_ = decltype(f_)(
      p.get<std::string>(name_rc_name),
      p.get<Teuchos::RCP<PHX::DataLayout>>(name_rc + " Data Layout"));
  return true;
}

template <int rank>
Field<rank>::operator bool() const
{
  return valid_;
}

template <typename T>
struct SizeType
{
  using T_noref      = typename std::remove_reference<T>::type;
  using T_noref_nocv = typename std::remove_cv<T_noref>::type;
  using type         = typename T_noref_nocv::size_type;
};

#define loop(f, i, dim) \
  for (typename SizeType<decltype(f)>::type i = 0; i < f.extent(dim); ++i)
#define loopf(i, dim) loop(f_, i, dim)

template <>
template <typename ad_type>
void
Field<0>::addTo(typename Tensor<ad_type, 0>::type& f_incr) const
{
  loopf(cell, 0) loopf(qp, 1) f_incr(cell, qp) += f_(cell, qp);
}
template <>
template <typename ad_type>
void
Field<1>::addTo(typename Tensor<ad_type, 1>::type& f_incr) const
{
  loopf(cell, 0) loopf(qp, 1) loopf(i0, 2) f_incr(cell, qp, i0) +=
      f_(cell, qp, i0);
}
template <>
template <typename ad_type>
void
Field<2>::addTo(typename Tensor<ad_type, 2>::type& f_incr) const
{
  loopf(cell, 0) loopf(qp, 1) loopf(i0, 2) loopf(i1, 3)
      f_incr(cell, qp, i0, i1) += f_(cell, qp, i0, i1);
}
template <>
template <typename ad_type>
void
Field<0>::addTo(
    typename Tensor<ad_type, 0>::type& f_incr,
    std::size_t const                  cell,
    std::size_t const                  qp) const
{
  f_incr(cell, qp) += f_(cell, qp);
}
template <>
template <typename ad_type>
void
Field<1>::addTo(
    typename Tensor<ad_type, 1>::type& f_incr,
    std::size_t const                  cell,
    std::size_t const                  qp) const
{
  loopf(i0, 2) f_incr(cell, qp, i0) += f_(cell, qp, i0);
}
template <>
template <typename ad_type>
void
Field<2>::addTo(
    typename Tensor<ad_type, 2>::type& f_incr,
    std::size_t const                  cell,
    std::size_t const                  qp) const
{
  loopf(i0, 2) loopf(i1, 3) f_incr(cell, qp, i0, i1) += f_(cell, qp, i0, i1);
}

namespace {
template <typename ad_type>
struct MultiplyWork
{
  minitensor::Tensor<ad_type>  f_incr_mt;
  minitensor::Tensor<RealType> f_accum_mt;
  MultiplyWork(std::size_t const dim) : f_incr_mt(dim), f_accum_mt(dim) {}
};

template <typename ad_type>
inline void
multiplyIntoImpl(
    const Tensor<const RealType, 2>::type& f_,
    typename Tensor<ad_type, 2>::type&     f_incr,
    std::size_t const                      cell,
    std::size_t const                      qp,
    MultiplyWork<ad_type>&                 w)
{
  loopf(i0, 2) loopf(i1, 3) w.f_incr_mt(i0, i1)  = f_incr(cell, qp, i0, i1);
  loopf(i0, 2) loopf(i1, 3) w.f_accum_mt(i0, i1) = f_(cell, qp, i0, i1);
  minitensor::Tensor<ad_type> C = minitensor::dot(w.f_incr_mt, w.f_accum_mt);
  loopf(i0, 2) loopf(i1, 3) f_incr(cell, qp, i0, i1) = C(i0, i1);
}
}  // namespace

template <>
template <typename ad_type>
void
Field<2>::multiplyInto(
    typename Tensor<ad_type, 2>::type& f_incr,
    std::size_t const                  cell,
    std::size_t const                  qp) const
{
  MultiplyWork<ad_type> w(f_.extent(2));
  multiplyIntoImpl(f_, f_incr, cell, qp, w);
}
template <>
template <typename ad_type>
void
Field<2>::multiplyInto(typename Tensor<ad_type, 2>::type& f_incr) const
{
  MultiplyWork<ad_type> w(f_.extent(2));
  loopf(cell, 0) loopf(qp, 1) multiplyIntoImpl(f_, f_incr, cell, qp, w);
}

#undef loopf
#undef loop

aadapt_rc_eti_class(Field)
#define eti(ad_type, rank) \
  template void Field<rank>::addTo<ad_type>(Tensor<ad_type, rank>::type&) const;
    aadapt_rc_apply_to_all_ad_types_all_ranks(eti)
#undef eti
#define eti(ad_type, rank)                                                \
  template void Field<rank>::addTo<ad_type>(                              \
      Tensor<ad_type, rank>::type&, std::size_t const, std::size_t const) \
      const;
        aadapt_rc_apply_to_all_ad_types_all_ranks(eti)
#undef eti
#define eti(ad_type, arg2)                                                 \
  template void Field<2>::multiplyInto<ad_type>(Tensor<ad_type, 2>::type&) \
      const;
            aadapt_rc_apply_to_all_ad_types(eti, )
#undef eti
#define eti(ad_type, arg2)                       \
  template void Field<2>::multiplyInto<ad_type>( \
      Tensor<ad_type, 2>::type&, std::size_t const, std::size_t const) const;
                aadapt_rc_apply_to_all_ad_types(eti, )
#undef eti

}  // namespace rc
}  // namespace AAdapt
