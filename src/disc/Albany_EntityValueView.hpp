// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_ENTITYVALUEVIEW_HPP
#define ALBANY_ENTITYVALUEVIEW_HPP

#include <cstddef>

namespace Albany {

// A layout-aware, non-owning view of one mesh entity's values in an STK Field.
//
// STK stores Field data either with an entity's components adjacent
// (stk::mesh::Layout::Right, the usual host layout) or with each component
// strided across entities (stk::mesh::Layout::Left, which a unified-memory
// build uses on the host too). The discretization used to cache a bare
// double* for each entity and index it [i], which reaches component i only
// under Layout::Right and silently reads a neighbouring entity's data under
// Layout::Left.
//
// This view carries the component stride STK reports for the layout actually
// in force, so v[i] is component i on either. Build it from the new Field data
// API, which is the only thing that knows the stride:
//
//   auto data   = field.data();
//   auto values = data.entity_values(entity);
//   EntityValueView v(values.pointer(), values.component_stride());
//
// Lifetime: this is a view, valid only while the Field data it was built from
// stays put. That is the same contract the raw pointer had, so caching one
// where a double* used to be cached is no worse; both must be rebuilt after a
// mesh modification.
template <typename T>
class EntityValueViewT
{
 public:
  EntityValueViewT() = default;
  EntityValueViewT(T* data, int component_stride) : data_(data), stride_(component_stride) {}

  // Component access, correct under any STK Field layout.
  T& operator[](int component) const { return data_[component * stride_]; }

  bool is_null() const { return data_ == nullptr; }
  T*   pointer() const { return data_; }
  int  component_stride() const { return stride_; }

  // Lets existing null checks written against the old double* keep working.
  explicit operator bool() const { return data_ != nullptr; }

 private:
  T*  data_   = nullptr;
  int stride_ = 1;
};

// Read-only by default: every cached use in the discretization reads.
using EntityValueView    = EntityValueViewT<double const>;
using MutEntityValueView = EntityValueViewT<double>;

}  // namespace Albany

#endif  // ALBANY_ENTITYVALUEVIEW_HPP
