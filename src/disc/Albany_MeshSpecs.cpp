#include "Albany_MeshSpecs.hpp"

#include "Albany_Macros.hpp"
#include "Shards_BasicTopologies.hpp"
#include "Shards_CellTopologyTraits.hpp"

namespace Albany {

MeshSpecsStruct::MeshSpecsStruct()
{
  ctd.name       = "NULL";
  numDim         = -1;
  worksetSize    = -1;
  cubatureDegree = -1;
  ebName         = "";
}

MeshSpecsStruct::MeshSpecsStruct(
    const CellTopologyData&    ctd_,
    int                        numDim_,
    int                        cubatureDegree_,
    std::vector<std::string>   nsNames_,
    std::vector<std::string>   ssNames_,
    int                        worksetSize_,
    std::string const          ebName_,
    std::map<std::string, int> ebNameToIndex_,
    bool                       interleavedOrdering_,
    bool const                 sepEvalsByEB_,
    Intrepid2::EPolyType const cubatureRule_)
    : ctd(ctd_),
      numDim(numDim_),
      cubatureDegree(cubatureDegree_),
      nsNames(nsNames_),
      ssNames(ssNames_),
      worksetSize(worksetSize_),
      ebName(ebName_),
      ebNameToIndex(ebNameToIndex_),
      interleavedOrdering(interleavedOrdering_),
      sepEvalsByEB(sepEvalsByEB_),
      cubatureRule(cubatureRule_)
{
  ALBANY_PANIC(cubatureDegree < 0, "Error! Invalid cubature degree on element block '" << ebName << "'.\n");
}

}  // namespace Albany
