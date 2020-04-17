// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Adapt_NodalDataVector.hpp"

#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Teuchos_CommHelpers.hpp"

namespace Adapt {

NodalDataVector::NodalDataVector(
    const Teuchos::RCP<Albany::NodeFieldContainer>& nodeContainer_,
    NodeFieldSizeVector&                            nodeVectorLayout_,
    NodeFieldSizeMap&                               nodeVectorMap_,
    LO&                                             vectorsize_)
    : nodeContainer(nodeContainer_),
      nodeVectorLayout(nodeVectorLayout_),
      nodeVectorMap(nodeVectorMap_),
      vectorsize(vectorsize_),
      mapsHaveChanged(false),
      num_preeval_calls(0),
      num_posteval_calls(0)
{
  // Nothing to be done here
}

void
NodalDataVector::replaceOverlapVectorSpace(
    Teuchos::RCP<Thyra_VectorSpace const> const& vs)
{
  overlap_node_vs = vs;

  // Build the vector
  overlap_node_vec = Thyra::createMembers(overlap_node_vs, vectorsize);

  mapsHaveChanged = true;
}

void
NodalDataVector::replaceOverlapVectorSpace(
    const Teuchos::Array<GO>&               overlap_nodeGIDs,
    const Teuchos::RCP<Teuchos_Comm const>& comm_)
{
  auto vs = Albany::createVectorSpace(comm_, overlap_nodeGIDs());
  replaceOverlapVectorSpace(vs);
}

void
NodalDataVector::replaceOwnedVectorSpace(
    Teuchos::RCP<Thyra_VectorSpace const> const& vs)
{
  owned_node_vs = vs;

  // Build the vector
  owned_node_vec = Thyra::createMembers(owned_node_vs, vectorsize);

  mapsHaveChanged = true;
}

void
NodalDataVector::replaceOwnedVectorSpace(
    const Teuchos::Array<GO>&               owned_nodeGIDs,
    const Teuchos::RCP<Teuchos_Comm const>& comm_)
{
  auto vs = Albany::createVectorSpace(comm_, owned_nodeGIDs());
  replaceOwnedVectorSpace(vs);
}

Teuchos::RCP<const Albany::CombineAndScatterManager>
NodalDataVector::initializeCASManager()
{
  if (mapsHaveChanged) {
    cas_manager =
        Albany::createCombineAndScatterManager(owned_node_vs, overlap_node_vs);
    mapsHaveChanged = false;
  }
  return cas_manager;
}

void
NodalDataVector::exportAddNodalDataVector()
{
  cas_manager->scatter(
      *owned_node_vec, *overlap_node_vec, Albany::CombineMode::ADD);
}

void
NodalDataVector::getNDofsAndOffset(
    std::string const& stateName,
    int&               offset,
    int&               ndofs) const
{
  NodeFieldSizeMap::const_iterator it;
  it = nodeVectorMap.find(stateName);

  ALBANY_PANIC(
      (it == nodeVectorMap.end()),
      std::endl
          << "Error: cannot find state " << stateName << " in NodalDataVector"
          << std::endl);

  std::size_t value = it->second;

  offset = nodeVectorLayout[value].offset;
  ndofs  = nodeVectorLayout[value].ndofs;
}

void
NodalDataVector::saveNodalDataState() const
{
  // Save the nodal data arrays back to stk.
  for (auto it = nodeVectorLayout.begin(); it != nodeVectorLayout.end(); ++it) {
    (*nodeContainer)[it->name]->saveFieldVector(overlap_node_vec, it->offset);
  }
}

void
NodalDataVector::saveNodalDataState(
    const Teuchos::RCP<const Thyra_MultiVector>& mv,
    int const                                    start_col) const
{
  // Save the nodal data arrays back to stk.
  const size_t nv = mv->domain()->dim();
  for (auto it = nodeVectorLayout.begin(); it != nodeVectorLayout.end(); ++it) {
    if (it->offset < start_col ||
        static_cast<unsigned long>(it->offset) >= start_col + nv) {
      continue;
    }
    (*nodeContainer)[it->name]->saveFieldVector(mv, it->offset - start_col);
  }
}

void
NodalDataVector::saveNodalDataVector(
    std::string const&                           name,
    const Teuchos::RCP<const Thyra_MultiVector>& overlap_node_vector,
    int const                                    offset) const
{
  Albany::NodeFieldContainer::const_iterator it = nodeContainer->find(name);
  ALBANY_PANIC(
      it == nodeContainer->end(),
      "Error: Cannot locate nodal field " << name << " in NodalDataVector");
  (*nodeContainer)[name]->saveFieldVector(overlap_node_vector, offset);
}

void
NodalDataVector::initializeVectors(ST value)
{
  overlap_node_vec->assign(value);
  owned_node_vec->assign(value);
}

}  // namespace Adapt
