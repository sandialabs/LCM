// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_SolutionCullingStrategy.hpp"

#include "Albany_Application.hpp"
#include "Albany_Gather.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Utils.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

class UniformSolutionCullingStrategy : public SolutionCullingStrategyBase
{
 public:
  explicit UniformSolutionCullingStrategy(int numValues) : numValues_(numValues)
  {
    // Nothing to be done here
  }
  Teuchos::Array<GO>
  selectedGIDs(Teuchos::RCP<Thyra_VectorSpace const> const& sourceVS) const;

 private:
  int numValues_;
};

Teuchos::Array<GO>
UniformSolutionCullingStrategy::selectedGIDs(Teuchos::RCP<Thyra_VectorSpace const> const& sourceVS) const
{
  auto source_indexer = createGlobalLocalIndexer(sourceVS);

  int const          localDim = source_indexer->getNumLocalElements();
  Teuchos::Array<GO> allGIDs(sourceVS->dim());
  Teuchos::Array<GO> myGIDs(localDim);

  for (LO lid = 0; lid < localDim; ++lid) {
    myGIDs[lid] = source_indexer->getGlobalElement(lid);
  }

  gatherAllV(source_indexer->getComm(), myGIDs(), allGIDs);
  std::sort(allGIDs.begin(), allGIDs.end());

  Teuchos::Array<GO> target_gids(numValues_);
  int const          stride = 1 + (allGIDs.size() - 1) / numValues_;
  for (int i = 0; i < numValues_; ++i) {
    target_gids[i] = allGIDs[i * stride];
  }
  return target_gids;
}

class NodeSetSolutionCullingStrategy : public SolutionCullingStrategyBase
{
 public:
  NodeSetSolutionCullingStrategy(std::string const& nodeSetLabel, const Teuchos::RCP<const Application>& app)
      : nodeSetLabel_(nodeSetLabel), app_(app), comm(app->getComm())
  {
    // Nothing to be done
  }

  Teuchos::Array<GO>
  selectedGIDs(Teuchos::RCP<Thyra_VectorSpace const> const& sourceVS) const;

  void
  setup()
  {
    disc_ = app_->getDiscretization();
    app_  = Teuchos::null;
  }

 private:
  std::string                      nodeSetLabel_;
  Teuchos::RCP<const Application>  app_;
  Teuchos::RCP<Teuchos_Comm const> comm;

  Teuchos::RCP<const AbstractDiscretization> disc_;
};

Teuchos::Array<GO>
NodeSetSolutionCullingStrategy::selectedGIDs(Teuchos::RCP<Thyra_VectorSpace const> const& sourceVS) const
{
  auto source_indexer = createGlobalLocalIndexer(sourceVS);

  // Gather gids on given nodeset on this rank
  Teuchos::Array<GO> mySelectedGIDs;

  NodeSetList const& nodeSets = disc_->getNodeSets();
  auto               it       = nodeSets.find(nodeSetLabel_);
  if (it != nodeSets.end()) {
    typedef NodeSetList::mapped_type NodeSetEntryList;
    NodeSetEntryList const&          sampleNodeEntries = it->second;

    for (NodeSetEntryList::const_iterator jt = sampleNodeEntries.begin(); jt != sampleNodeEntries.end(); ++jt) {
      typedef NodeSetEntryList::value_type NodeEntryList;
      NodeEntryList const&                 sampleEntries = *jt;
      for (NodeEntryList::const_iterator kt = sampleEntries.begin(); kt != sampleEntries.end(); ++kt) {
        mySelectedGIDs.push_back(source_indexer->getGlobalElement(*kt));
      }
    }
  }

  // Sum the number of selected gids across all ranks
  GO selectedGIDCount;
  GO mySelectedGIDCount = mySelectedGIDs.size();
  Teuchos::reduceAll<LO, GO>(*comm, Teuchos::REDUCE_SUM, 1, &mySelectedGIDCount, &selectedGIDCount);

  // Gather all selected gids
  Teuchos::Array<GO> target_gids;
  target_gids.resize(selectedGIDCount);

  gatherAllV(comm, mySelectedGIDs(), target_gids);
  std::sort(target_gids.begin(), target_gids.end());

  return target_gids;
}

class NodeGIDsSolutionCullingStrategy : public SolutionCullingStrategyBase
{
 public:
  NodeGIDsSolutionCullingStrategy(const Teuchos::Array<int>& nodeGIDs, const Teuchos::RCP<const Application>& app)
      : nodeGIDs_(nodeGIDs), app_(app), comm(app->getComm()), disc_(Teuchos::null)
  {
    // Nothing to be done
  }

  Teuchos::Array<GO>
  selectedGIDs(Teuchos::RCP<Thyra_VectorSpace const> const& sourceVS) const;
  void
  setup();

 private:
  Teuchos::Array<int>              nodeGIDs_;
  Teuchos::RCP<const Application>  app_;
  Teuchos::RCP<Teuchos_Comm const> comm;

  Teuchos::RCP<const AbstractDiscretization> disc_;
};

void
NodeGIDsSolutionCullingStrategy::setup()
{
  disc_ = app_->getDiscretization();
  // Once the discretization has been obtained, a handle to the application is
  // not required Release the resource to avoid possible circular references
  app_.reset();
}

Teuchos::Array<GO>
NodeGIDsSolutionCullingStrategy::selectedGIDs(Teuchos::RCP<Thyra_VectorSpace const> const& sourceVS) const
{
  Teuchos::Array<GO> mySelectedGIDs;

  // Subract 1 to convert exodus GIDs to our GIDs
  auto source_indexer = createGlobalLocalIndexer(sourceVS);
  for (int i = 0; i < nodeGIDs_.size(); ++i) {
    if (source_indexer->isLocallyOwnedElement(nodeGIDs_[i] - 1)) {
      mySelectedGIDs.push_back(nodeGIDs_[i] - 1);
    }
  }

  GO selectedGIDCount;
  GO mySelectedGIDCount = mySelectedGIDs.size();
  Teuchos::reduceAll<LO, GO>(*comm, Teuchos::REDUCE_SUM, 1, &mySelectedGIDCount, &selectedGIDCount);

  Teuchos::Array<GO> result(selectedGIDCount);

  gatherAllV(comm, mySelectedGIDs(), result);
  std::sort(result.begin(), result.end());

  return result;
}

Teuchos::RCP<SolutionCullingStrategyBase>
createSolutionCullingStrategy(const Teuchos::RCP<const Application>& app, Teuchos::ParameterList& params)
{
  std::string const cullingStrategyToken = params.get("Culling Strategy", "Uniform");

  if (cullingStrategyToken == "Uniform") {
    int const numValues = params.get("Num Values", 10);
    return Teuchos::rcp(new UniformSolutionCullingStrategy(numValues));
  } else if (cullingStrategyToken == "Node Set") {
    std::string const nodeSetLabel = params.get<std::string>("Node Set Label");
    return Teuchos::rcp(new NodeSetSolutionCullingStrategy(nodeSetLabel, app));
  } else if (cullingStrategyToken == "Node GIDs") {
    Teuchos::Array<int> nodeGIDs = params.get<Teuchos::Array<int>>("Node GID Array");
    return Teuchos::rcp(new NodeGIDsSolutionCullingStrategy(nodeGIDs, app));
  }

  bool const unsupportedCullingStrategy = true;
  ALBANY_PANIC(unsupportedCullingStrategy);
  return Teuchos::null;
}

}  // namespace Albany
