// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_BCUTILS_HPP
#define ALBANY_BCUTILS_HPP

#include <Phalanx_Evaluator_TemplateManager.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <string>
#include <vector>

#include "Albany_DataTypes.hpp"
#include "Albany_MaterialDatabase.hpp"
#include "Albany_MeshSpecs.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_FactoryTraits.hpp"

namespace Albany {

/*!
 * \brief Generic Functions to help define BC Field Manager
 */

//! Traits classes used for BCUtils
struct DirichletTraits
{
  enum
  {
    type = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_dirichlet
  };
  enum
  {
    typeTd = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_timedep_bc
  };
  enum
  {
    typeTs = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_timedep_sdbc
  };
  enum
  {
    typeKf = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_kfield_bc
  };
  enum
  {
    typeEq =
        PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_eq_concentration_bc
  };
  enum
  {
    typeTo = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_torsion_bc
  };
  enum
  {
    typeSt = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_sdbc
  };
  enum
  {
    typeEe = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_expreval_sdbc
  };
  enum
  {
    typeSw = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_schwarz_bc
  };
  enum
  {
    typeSsw =
        PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_strong_schwarz_bc
  };
  enum
  {
    typeDa = PHAL::DirichletFactoryTraits<
        PHAL::AlbanyTraits>::id_dirichlet_aggregator
  };
  enum
  {
    typeFb = PHAL::DirichletFactoryTraits<
        PHAL::AlbanyTraits>::id_dirichlet_coordinate_function
  };
  enum
  {
    typeF = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_dirichlet_field
  };
  enum
  {
    typeON = PHAL::DirichletFactoryTraits<
        PHAL::AlbanyTraits>::id_dirichlet_off_nodeset
  };

  static std::string const bcParamsPl;

  typedef PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits> factory_type;

  static Teuchos::RCP<Teuchos::ParameterList const>
  getValidBCParameters(
      std::vector<std::string> const& nodeSetIDs,
      std::vector<std::string> const& bcNames);

  static std::string
  constructBCName(std::string const& ns, std::string const& dof);

  static std::string
  constructSDBCName(std::string const& ns, std::string const& dof);

  static std::string
  constructExprEvalSDBCName(std::string const& ns, std::string const& dof);

  static std::string
  constructScaledSDBCName(std::string const& ns, std::string const& dof);

  static std::string
  constructBCNameField(std::string const& ns, std::string const& dof);

  static std::string
  constructSDBCNameField(std::string const& ns, std::string const& dof);

  static std::string
  constructExprEvalSDBCNameField(std::string const& ns, std::string const& dof);

  static std::string
  constructScaledSDBCNameField(std::string const& ns, std::string const& dof);

  static std::string
  constructTimeDepBCName(std::string const& ns, std::string const& dof);

  static std::string
  constructTimeDepSDBCName(std::string const& ns, std::string const& dof);

  static std::string
  constructBCNameOffNodeSet(std::string const& ns, std::string const& dof);

  static std::string
  constructPressureDepBCName(std::string const& ns, std::string const& dof);
};

struct NeumannTraits
{
  enum
  {
    type = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_neumann
  };
  enum
  {
    typeNa =
        PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_neumann_aggregator
  };
  enum
  {
    typeGCV =
        PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_gather_coord_vector
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
    typeSF = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_load_stateField
  };
  enum
  {
    typeSNP = PHAL::NeumannFactoryTraits<
        PHAL::AlbanyTraits>::id_GatherScalarNodalParameter
  };

  static std::string const bcParamsPl;

  typedef PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits> factory_type;

  static Teuchos::RCP<Teuchos::ParameterList const>
  getValidBCParameters(
      std::vector<std::string> const& sideSetIDs,
      std::vector<std::string> const& bcNames,
      std::vector<std::string> const& conditions);

  static std::string
  constructBCName(
      std::string const& ns,
      std::string const& dof,
      std::string const& condition);

  static std::string
  constructTimeDepBCName(
      std::string const& ns,
      std::string const& dof,
      std::string const& condition);
};

template <typename BCTraits>
class BCUtils
{
 public:
  BCUtils() {}

  //! Type of traits class being used
  typedef BCTraits traits_type;

  //! Function to check if the Neumann/Dirichlet BC section of input file is
  //! present
  bool
  haveBCSpecified(const Teuchos::RCP<Teuchos::ParameterList>& params) const
  {
    // If the BC sublist is not in the input file,
    // side/node sets can be contained in the Exodus file but are not defined in
    // the problem statement.
    // This is OK, just return

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

  bool
  useSDBCs() const
  {
    return use_sdbcs_;
  }

  //! Specific implementation for Dirichlet BC Evaluator below

  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
  constructBCEvaluators(
      std::vector<std::string> const&      nodeSetIDs,
      std::vector<std::string> const&      bcNames,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib>               paramLib,
      int const                            numEqn = 0);

  //! Specific implementation for Dirichlet BC Evaluator below

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
      const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs,
      std::vector<std::string> const&              bcNames,
      const Teuchos::ArrayRCP<std::string>&        dof_names,
      bool                                         isVectorField,
      int                                          offsetToFirstDOF,
      std::vector<std::string> const&              conditions,
      const Teuchos::Array<Teuchos::Array<int>>&   offsets,
      const Teuchos::RCP<Albany::Layouts>&         dl,
      Teuchos::RCP<Teuchos::ParameterList>         params,
      Teuchos::RCP<ParamLib>                       paramLib,
      std::vector<Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits>>> const&
                                                    extra_evaluators,
      const Teuchos::RCP<Albany::MaterialDatabase>& materialDB = Teuchos::null);

 private:
  //! Builds the list
  void
  buildEvaluatorsList(
      std::map<std::string, Teuchos::RCP<Teuchos::ParameterList>>&
                                           evaluatorss_to_build,
      std::vector<std::string> const&      nodeSetIDs,
      std::vector<std::string> const&      bcNames,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib>               paramLib,
      int const                            numEqn);

  //! Creates the list of evaluators (together with their parameter lists) to
  //! build
  void
  buildEvaluatorsList(
      std::map<std::string, Teuchos::RCP<Teuchos::ParameterList>>&
                                                    evaluators_to_build,
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

  //! Generic implementation of Field Manager construction function
  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
  buildFieldManager(
      const Teuchos::RCP<std::vector<Teuchos::RCP<
          PHX::Evaluator_TemplateManager<PHAL::AlbanyTraits>>>> evaluators,
      std::string&                                              allBC,
      Teuchos::RCP<PHX::DataLayout>&                            dummy);

 protected:
  Teuchos::Array<Teuchos::Array<int>> offsets_;
  std::vector<std::string>            nodeSetIDs_;
  bool                                use_sdbcs_{false};
  bool                                use_dbcs_{false};
};

//! Specific implementation for Dirichlet BC Evaluator

template <>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
BCUtils<DirichletTraits>::constructBCEvaluators(
    std::vector<std::string> const&      nodeSetIDs,
    std::vector<std::string> const&      bcNames,
    Teuchos::RCP<Teuchos::ParameterList> params,
    Teuchos::RCP<ParamLib>               paramLib,
    int const                            numEqn);

//! Specific implementation for Dirichlet BC Evaluator

template <>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
BCUtils<NeumannTraits>::constructBCEvaluators(
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
    const Teuchos::RCP<Albany::MaterialDatabase>& materialDB);

template <>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
BCUtils<NeumannTraits>::constructBCEvaluators(
    const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs,
    std::vector<std::string> const&              bcNames,
    const Teuchos::ArrayRCP<std::string>&        dof_names,
    bool                                         isVectorField,
    int                                          offsetToFirstDOF,
    std::vector<std::string> const&              conditions,
    const Teuchos::Array<Teuchos::Array<int>>&   offsets,
    const Teuchos::RCP<Albany::Layouts>&         dl,
    Teuchos::RCP<Teuchos::ParameterList>         params,
    Teuchos::RCP<ParamLib>                       paramLib,
    std::vector<Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits>>> const&
                                                  extra_evaluators,
    const Teuchos::RCP<Albany::MaterialDatabase>& materialDB);

template <>
void
BCUtils<DirichletTraits>::buildEvaluatorsList(
    std::map<std::string, Teuchos::RCP<Teuchos::ParameterList>>&
                                         evaluators_to_build,
    std::vector<std::string> const&      nodeSetIDs,
    std::vector<std::string> const&      bcNames,
    Teuchos::RCP<Teuchos::ParameterList> params,
    Teuchos::RCP<ParamLib>               paramLib,
    int                                  numEqn);

template <>
void
BCUtils<NeumannTraits>::buildEvaluatorsList(
    std::map<std::string, Teuchos::RCP<Teuchos::ParameterList>>&
                                                  evaluators_to_build,
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
    const Teuchos::RCP<Albany::MaterialDatabase>& materialDB);
}  // namespace Albany

// Define macro for explicit template instantiation
#define BCUTILS_INSTANTIATE_TEMPLATE_CLASS_DIRICHLET(name) \
  template class name<Albany::DirichletTraits>;
#define BCUTILS_INSTANTIATE_TEMPLATE_CLASS_NEUMANN(name) \
  template class name<Albany::NeumannTraits>;

#define BCUTILS_INSTANTIATE_TEMPLATE_CLASS(name)     \
  BCUTILS_INSTANTIATE_TEMPLATE_CLASS_DIRICHLET(name) \
  BCUTILS_INSTANTIATE_TEMPLATE_CLASS_NEUMANN(name)

#endif
