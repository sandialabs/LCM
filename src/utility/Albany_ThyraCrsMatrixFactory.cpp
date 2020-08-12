#include "Albany_ThyraCrsMatrixFactory.hpp"

#include "Albany_Macros.hpp"
#include "Albany_TpetraTypes.hpp"
#include "Albany_Utils.hpp"

namespace Albany {

// The implementation of the graph
struct ThyraCrsMatrixFactory::Impl
{
  Impl() = default;

  Teuchos::RCP<Tpetra_CrsGraph> t_graph;
};

ThyraCrsMatrixFactory::ThyraCrsMatrixFactory(
    Teuchos::RCP<Thyra_VectorSpace const> const domain_vs,
    Teuchos::RCP<Thyra_VectorSpace const> const range_vs,
    int const /*nonzeros_per_row*/)
    : m_graph(new Impl()), m_domain_vs(domain_vs), m_range_vs(range_vs), m_filled(false)
{
  t_range = getTpetraMap(range_vs);
  t_local_graph.resize(t_range->getNodeNumElements());
}

ThyraCrsMatrixFactory::ThyraCrsMatrixFactory(
    Teuchos::RCP<Thyra_VectorSpace const> const     domain_vs,
    Teuchos::RCP<Thyra_VectorSpace const> const     range_vs,
    const Teuchos::RCP<const ThyraCrsMatrixFactory> overlap_src)
    : m_domain_vs(domain_vs), m_range_vs(range_vs)
{
  ALBANY_PANIC(
      !overlap_src->is_filled(),
      "Error! Can only build a graph from an overlapped source if source has "
      "been filled already.\n");
  m_graph = Teuchos::rcp(new Impl());

  auto t_range         = getTpetraMap(range_vs);
  auto t_overlap_range = getTpetraMap(overlap_src->m_range_vs);
  auto t_overlap_graph = overlap_src->m_graph->t_graph;

  // Creating an empty graph. The graph will be automatically resized when
  // exported.
  m_graph->t_graph = createCrsGraph(t_range);

  Tpetra_Export exporter(t_overlap_range, t_range);
  m_graph->t_graph->doExport(*t_overlap_graph, exporter, Tpetra::INSERT);

  auto t_domain = getTpetraMap(domain_vs);
  m_graph->t_graph->fillComplete(t_domain, t_range);
  m_filled = true;
}

void
ThyraCrsMatrixFactory::insertGlobalIndices(const GO row, const Teuchos::ArrayView<const GO>& indices)
{
  // Indices are inserted in a temporary local graph.
  // The actual graph is created and filled when fillComplete is called
  // Despite being both 64 bits, GO and Tpetra_GO *may* be different *types*.
  int lrow = t_range->getLocalElement(static_cast<Tpetra_GO>(row));

  // ignore indices that are not owned by the this processor
  if (lrow < 0) return;

  auto& row_indices = t_local_graph[lrow];
  for (int i = 0; i < indices.size(); ++i) {
    row_indices.emplace(static_cast<Tpetra_GO>(indices[i]));
  }
}

void
ThyraCrsMatrixFactory::fillComplete()
{
  // We created the CrsGraph,
  // insert indices from the temporary local graph,
  // and call fill complete.
  Teuchos::ArrayRCP<size_t> nonzeros_per_row_array(t_range->getNodeNumElements());

  for (int lrow = 0; lrow < nonzeros_per_row_array.size(); ++lrow) {
    nonzeros_per_row_array[lrow] = t_local_graph[lrow].size();
  }

  m_graph->t_graph = Teuchos::rcp(new Tpetra_CrsGraph(t_range, nonzeros_per_row_array()));

  for (int lrow = 0; lrow < nonzeros_per_row_array.size(); ++lrow) {
    auto& row_indices = t_local_graph[lrow];
    if (row_indices.size() > 0) {
      Teuchos::Array<Tpetra_GO> t_indices(row_indices.size());
      int                       i = 0;
      for (const auto& index : row_indices) t_indices[i++] = index;
      auto row = t_range->getGlobalElement(lrow);

      m_graph->t_graph->insertGlobalIndices(row, t_indices);
    }
  }

  t_local_graph.clear();
  auto t_domain = getTpetraMap(m_domain_vs);
  m_graph->t_graph->fillComplete(t_domain, t_range);
  t_range.reset();

  m_filled = true;
}

Teuchos::RCP<Thyra_LinearOp>
ThyraCrsMatrixFactory::createOp() const
{
  ALBANY_PANIC(!m_filled, "Error! Cannot create a linear operator if the graph is not filled.\n");

  Teuchos::RCP<Thyra_LinearOp>   op;
  Teuchos::RCP<Tpetra_CrsMatrix> mat  = Teuchos::rcp(new Tpetra_CrsMatrix(m_graph->t_graph));
  auto const                     zero = Teuchos::ScalarTraits<ST>::zero();
  mat->resumeFill();
  mat->setAllToScalar(zero);
  mat->fillComplete();
  op = createThyraLinearOp(Teuchos::rcp_implicit_cast<Tpetra_Operator>(mat));
  return op;
}

}  // namespace Albany
