// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "PHAL_Utilities.hpp"

#include "Albany_Application.hpp"
#include "Albany_StateInfoStruct.hpp"

namespace PHAL {

template <>
int
getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
    Albany::Application const*     app,
    Albany::MeshSpecsStruct const* ms)
{
  Teuchos::RCP<Teuchos::ParameterList const> const pl = app->getProblemPL();
  if (Teuchos::nonnull(pl)) {
    bool const extrudedColumnCoupled =
        pl->isParameter("Extruded Column Coupled in 2D Response") ?
            pl->get<bool>("Extruded Column Coupled in 2D Response") :
            false;
    if (extrudedColumnCoupled) {  // all column is coupled
      int side_node_count = ms->ctd.side[3].topology->node_count;
      int node_count      = ms->ctd.node_count;
      int numLevels =
          app->getDiscretization()->getLayeredMeshNumbering()->numLayers + 1;
      return app->getNumEquations() *
             (node_count + side_node_count * numLevels);
    }
  }
  return app->getNumEquations() * ms->ctd.node_count;
}

template <>
int
getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
    Albany::Application const* app,
    int const                  ebi)
{
  Teuchos::RCP<Teuchos::ParameterList const> const pl = app->getProblemPL();
  if (Teuchos::nonnull(pl)) {
    std::string const problemName =
        pl->isType<std::string>("Name") ? pl->get<std::string>("Name") : "";
  }
  return getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
      app, app->getEnrichedMeshSpecs()[ebi].get());
}

namespace {
template <typename ScalarT>
struct A2V
{
  std::vector<ScalarT>& v;
  A2V(std::vector<ScalarT>& v) : v(v) {}
  void
  operator()(typename Ref<ScalarT const>::type a, int const i)
  {
    v[i] = a;
  }
};

template <typename ScalarT>
struct V2A
{
  std::vector<ScalarT> const& v;
  V2A(std::vector<ScalarT> const& v) : v(v) {}
  void
  operator()(typename Ref<ScalarT>::type a, int const i)
  {
    a = v[i];
  }
};

template <typename ScalarT>
void
copy(const PHX::MDField<ScalarT>& a, std::vector<ScalarT>& v)
{
  v.resize(a.size());
  A2V<ScalarT> a2v(v);
  loop(a2v, a);
}

template <typename ScalarT>
void
copy(std::vector<ScalarT> const& v, PHX::MDField<ScalarT>& a)
{
  V2A<ScalarT> v2a(v);
  loop(v2a, a);
}

template <typename ScalarT>
void
myReduceAll(
    Teuchos_Comm const&           comm,
    Teuchos::EReductionType const reduct_type,
    std::vector<ScalarT>&         v)
{
  typedef typename ScalarT::value_type ValueT;
  // Size of array to hold one Fad's derivatives.
  int const sz = v[0].size();
  // Pack into a vector of values.
  std::vector<ValueT> pack;
  for (int i = 0; i < v.size(); ++i) {
    pack.push_back(v[i].val());
    for (int j = 0; j < sz; ++j) pack.push_back(v[i].fastAccessDx(j));
  }
  // reduceAll the package.
  switch (reduct_type) {
    case Teuchos::REDUCE_SUM: {
      std::vector<ValueT> send(pack);
      Teuchos::reduceAll<int, ValueT>(
          comm, reduct_type, pack.size(), &send[0], &pack[0]);
    } break;
    default: ALBANY_ABORT("not impl'ed");
  }
  // Unpack.
  int slot = 0;
  for (int i = 0; i < v.size(); ++i) {
    v[i].val() = pack[slot++];
    for (int j = 0; j < sz; ++j) v[i].fastAccessDx(j) = pack[slot++];
  }
}

template <>
void
myReduceAll<RealType>(
    Teuchos_Comm const&           comm,
    Teuchos::EReductionType const reduct_type,
    std::vector<RealType>&        v)
{
  std::vector<RealType> send(v);
  Teuchos::reduceAll<int, RealType>(
      comm, reduct_type, v.size(), &send[0], &v[0]);
}

}  // namespace

template <typename ScalarT>
void
reduceAll(
    Teuchos_Comm const&           comm,
    Teuchos::EReductionType const reduct_type,
    PHX::MDField<ScalarT>&        a)
{
  std::vector<ScalarT> v;
  copy<ScalarT>(a, v);
  myReduceAll<ScalarT>(comm, reduct_type, v);
  copy<ScalarT>(v, a);
}

template <typename ScalarT>
void
reduceAll(
    Teuchos_Comm const&           comm,
    Teuchos::EReductionType const reduct_type,
    ScalarT&                      a)
{
  ScalarT b = a;
  Teuchos::reduceAll(comm, reduct_type, 1, &a, &b);
  a = b;
}

template <typename ScalarT>
void
broadcast(
    Teuchos_Comm const&    comm,
    int const              root_rank,
    PHX::MDField<ScalarT>& a)
{
  std::vector<ScalarT> v;
  copy<ScalarT>(a, v);
  Teuchos::broadcast<int, ScalarT>(comm, root_rank, v.size(), &v[0]);
  copy<ScalarT>(v, a);
}

#if defined(ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE)
#define apply_to_all_ad_types(macro) \
  macro(RealType) macro(FadType) macro(TanFadType)
#else
#define apply_to_all_ad_types(macro) macro(RealType) macro(FadType)
#endif

#define eti(T)                \
  template void reduceAll<T>( \
      Teuchos_Comm const&, Teuchos::EReductionType const, PHX::MDField<T>&);
apply_to_all_ad_types(eti)
#undef eti
#define eti(T)                \
  template void reduceAll<T>( \
      Teuchos_Comm const&, Teuchos::EReductionType const, T&);
    apply_to_all_ad_types(eti)
#undef eti
#define eti(T)                \
  template void broadcast<T>( \
      Teuchos_Comm const&, int const root_rank, PHX::MDField<T>&);
        apply_to_all_ad_types(eti)
#undef eti
#undef apply_to_all_ad_types

}  // namespace PHAL
