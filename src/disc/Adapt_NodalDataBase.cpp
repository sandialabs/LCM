// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Adapt_NodalDataBase.hpp"

#include "Adapt_NodalDataVector.hpp"

namespace Adapt {

NodalDataBase::NodalDataBase()
    : nodeContainer(Teuchos::rcp(new Albany::NodeFieldContainer)), vectorsize(0), initialized(false)
{
  // Nothing to be done here
}

void
NodalDataBase::updateNodalGraph(const Teuchos::RCP<const Albany::ThyraCrsMatrixFactory>& crsOpFactory)
{
  nodalOpFactory = crsOpFactory;
}

void
NodalDataBase::registerVectorState(std::string const& stateName, int ndofs)
{
  // Save the nodal data field names and lengths in order of allocation which
  // implies access order.
  auto it = nodeVectorMap.find(stateName);

  ALBANY_PANIC(
      (it != nodeVectorMap.end()),
      std::endl
          << "Error: found duplicate entry " << stateName << " in NodalDataVector");

  NodeFieldSize size;
  size.name   = stateName;
  size.offset = vectorsize;
  size.ndofs  = ndofs;

  nodeVectorMap[stateName] = nodeVectorLayout.size();
  nodeVectorLayout.push_back(size);

  vectorsize += ndofs;
}

void
NodalDataBase::initialize()
{
  if (initialized) {
    return;
  }

  if (vectorsize > 0) {
    nodal_data_vector = Teuchos::rcp(new NodalDataVector(nodeContainer, nodeVectorLayout, nodeVectorMap, vectorsize));
  }

  initialized = true;
}

void
NodalDataBase::replaceOverlapVectorSpace(Teuchos::RCP<Thyra_VectorSpace const> const& vs)
{
  initialize();

  if (Teuchos::nonnull(nodal_data_vector)) nodal_data_vector->replaceOverlapVectorSpace(vs);
}

void
NodalDataBase::replaceOwnedVectorSpace(Teuchos::RCP<Thyra_VectorSpace const> const& vs)
{
  initialize();

  if (Teuchos::nonnull(nodal_data_vector)) {
    nodal_data_vector->replaceOwnedVectorSpace(vs);
  }
}

void
NodalDataBase::replaceOverlapVectorSpace(
    const Teuchos::Array<GO>&               overlap_nodeGIDs,
    const Teuchos::RCP<Teuchos_Comm const>& comm_)
{
  initialize();

  if (Teuchos::nonnull(nodal_data_vector)) {
    nodal_data_vector->replaceOverlapVectorSpace(overlap_nodeGIDs, comm_);
  }
}

void
NodalDataBase::replaceOwnedVectorSpace(
    const Teuchos::Array<GO>&               local_nodeGIDs,
    const Teuchos::RCP<Teuchos_Comm const>& comm_)
{
  initialize();

  if (Teuchos::nonnull(nodal_data_vector)) {
    nodal_data_vector->replaceOwnedVectorSpace(local_nodeGIDs, comm_);
  }
}

void
NodalDataBase::registerManager(std::string const& key, const Teuchos::RCP<NodalDataBase::Manager>& manager)
{
  ALBANY_PANIC(isManagerRegistered(key), "A manager is already registered with key " << key);
  mgr_map[key] = manager;
}

bool
NodalDataBase::isManagerRegistered(std::string const& key) const
{
  return mgr_map.find(key) != mgr_map.end();
}

const Teuchos::RCP<NodalDataBase::Manager>&
NodalDataBase::getManager(std::string const& key) const
{
  ManagerMap::const_iterator it = mgr_map.find(key);
  ALBANY_PANIC(it == mgr_map.end(), "There is no manager with key " << key);
  return it->second;
}

}  // namespace Adapt
