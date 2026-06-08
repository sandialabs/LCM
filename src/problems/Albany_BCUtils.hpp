// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#ifndef ALBANY_BCUTILS_HPP
#define ALBANY_BCUTILS_HPP

#include "Albany_DataTypes.hpp"
#include "Albany_MaterialDatabase.hpp"
#include "Albany_ProblemUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Phalanx_Evaluator_TemplateManager.hpp"

namespace Albany {

// Dirichlet-side BC utilities were retired alongside the dfm pipeline. The
// DBC DOF-elimination path in Application::eliminateConstrainedDOFs and
// injectConstrainedDOFValues now parses YAML DBC keys directly and owns
// all Dirichlet enforcement. Only Neumann-side BCs still flow through this
// utility.

struct NeumannTraits
{
  enum
  {
    type = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_neumann
  };
  enum
  {
    typeNa = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_neumann_aggregator
  };
  enum
  {
    typeGCV = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_gather_coord_vector
  };
  enum
  {
    typeGS = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_gather_solution
  };
  enum
  {
    typeTd = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_timedep_bc
  };
  enum
  {
    typeATd = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_acetimedep_bc
  };
  enum
  {
    typeSF = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_load_stateField
  };
  enum
  {
    typeSNP = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_GatherScalarNodalParameter
  };

  static std::string const bcParamsPl;

  typedef PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits> factory_type;

  static Teuchos::RCP<Teuchos::ParameterList const>
  getValidBCParameters(std::vector<std::string> const& sideSetIDs, std::vector<std::string> const& bcNames, std::vector<std::string> const& conditions);

  static std::string
  constructBCName(std::string const& ns, std::string const& dof, std::string const& condition);

  static std::string
  constructTimeDepBCName(std::string const& ns, std::string const& dof, std::string const& condition);

  static std::string
  constructACETimeDepBCName(std::string const& ns, std::string const& dof, std::string const& condition);
};

template <typename BCTraits>
class BCUtils
{
 public:
  BCUtils() {}

  //! Type of traits class being used
  typedef BCTraits traits_type;

  //! Function to check if the Neumann BC section of input file is present
  bool
  haveBCSpecified(const Teuchos::RCP<Teuchos::ParameterList>& params) const
  {
    return params->isSublist(traits_type::bcParamsPl);
  }

  Teuchos::Array<Teuchos::Array<int>>
  getOffsets() const
  {
    return offsets_;
  }

  std::vector<std::string>
  getNodeSetIDs() const
  {
    return nodeSetIDs_;
  }
  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
  constructBCEvaluators(
      const Teuchos::RCP<Albany::MeshSpecsStruct>&  meshSpecs,
      std::vector<std::string> const&               bcNames,
      const Teuchos::ArrayRCP<std::string>&         dof_names,
      bool                                          isVectorField,
      int                                           offsetToFirstDOF,
      std::vector<std::string> const&               conditions,
      const Teuchos::Array<Teuchos::Array<int>>&    offsets,
      const Teuchos::RCP<Albany::Layouts>&          dl,
      Teuchos::RCP<Teuchos::ParameterList>          params,
      Teuchos::RCP<ParamLib>                        paramLib,
      const Teuchos::RCP<Albany::MaterialDatabase>& materialDB = Teuchos::null);

  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
  constructBCEvaluators(
      const Teuchos::RCP<Albany::MeshSpecsStruct>&                         meshSpecs,
      std::vector<std::string> const&                                      bcNames,
      const Teuchos::ArrayRCP<std::string>&                                dof_names,
      bool                                                                 isVectorField,
      int                                                                  offsetToFirstDOF,
      std::vector<std::string> const&                                      conditions,
      const Teuchos::Array<Teuchos::Array<int>>&                           offsets,
      const Teuchos::RCP<Albany::Layouts>&                                 dl,
      Teuchos::RCP<Teuchos::ParameterList>                                 params,
      Teuchos::RCP<ParamLib>                                               paramLib,
      std::vector<Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits>>> const& extra_evaluators,
      const Teuchos::RCP<Albany::MaterialDatabase>&                        materialDB = Teuchos::null);

 protected:
  void
  buildEvaluatorsList(
      std::map<std::string, Teuchos::RCP<Teuchos::ParameterList>>& evaluators_to_build,
      const Teuchos::RCP<Albany::MeshSpecsStruct>&                 meshSpecs,
      std::vector<std::string> const&                              bcNames,
      const Teuchos::ArrayRCP<std::string>&                        dof_names,
      bool                                                         isVectorField,
      int                                                          offsetToFirstDOF,
      std::vector<std::string> const&                              conditions,
      const Teuchos::Array<Teuchos::Array<int>>&                   offsets,
      const Teuchos::RCP<Albany::Layouts>&                         dl,
      Teuchos::RCP<Teuchos::ParameterList>                         params,
      Teuchos::RCP<ParamLib>                                       paramLib,
      const Teuchos::RCP<Albany::MaterialDatabase>&                materialDB = Teuchos::null);

  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
  buildFieldManager(
      const Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator_TemplateManager<PHAL::AlbanyTraits>>>> evaluators,
      std::string&                                                                                     allBC,
      Teuchos::RCP<PHX::DataLayout>&                                                                   dummy);

  std::vector<std::string>            nodeSetIDs_;
  Teuchos::Array<Teuchos::Array<int>> offsets_;
};

// Forward declare the Neumann specialization of buildEvaluatorsList so the
// constructBCEvaluators<NeumannTraits> definition in _Def.hpp resolves to
// the specialization (defined later in the same file) rather than implicitly
// instantiating the primary template — the primary has no body.
template <>
void
BCUtils<NeumannTraits>::buildEvaluatorsList(
    std::map<std::string, Teuchos::RCP<Teuchos::ParameterList>>& evaluators_to_build,
    const Teuchos::RCP<Albany::MeshSpecsStruct>&                 meshSpecs,
    std::vector<std::string> const&                              bcNames,
    const Teuchos::ArrayRCP<std::string>&                        dof_names,
    bool                                                         isVectorField,
    int                                                          offsetToFirstDOF,
    std::vector<std::string> const&                              conditions,
    const Teuchos::Array<Teuchos::Array<int>>&                   offsets,
    const Teuchos::RCP<Albany::Layouts>&                         dl,
    Teuchos::RCP<Teuchos::ParameterList>                         params,
    Teuchos::RCP<ParamLib>                                       paramLib,
    const Teuchos::RCP<Albany::MaterialDatabase>&                materialDB);

}  // namespace Albany

#endif
