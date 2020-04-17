// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
#ifndef LAME_UTILS_HPP
#define LAME_UTILS_HPP

#include "Albany_config.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

// LAME material model base class
#if defined(ALBANY_LAME)
#include <models/Material.h>
typedef lame::MatProps  LameMatProps;
typedef lame::Material  LameMaterial;
typedef lame::matParams LameMatParams;
#endif
#if defined(ALBANY_LAMENT)
#include <models/Lament_Material.h>

#include "models/Lament_ElasticNew.h"
#include "models/Lament_Neohookean.h"
typedef lament::MatProps         LameMatProps;
typedef lament::Material<double> LameMaterial;
// typedef lament::Material<ADType> LameMaterial_AD;
typedef lament::matParams<double> LameMatParams;
#endif

//! Utility functions for interfacing to LAME material library
namespace LameUtils {

//! Convert a Teuchos::ParameterList into a lame(nt)::MatProps structure.
inline void
parameterListToMatProps(
    Teuchos::ParameterList const& lameMaterialParameters,
    LameMatProps&                 matProps)
{
  // load the material properties into the lame(nt)::MatProps container.
  // LAME material properties must be of type int, double, or string.

  for (Teuchos::ParameterList::ConstIterator it =
           lameMaterialParameters.begin();
       it != lameMaterialParameters.end();
       ++it) {
    std::string name = lameMaterialParameters.name(it);
    std::transform(
        name.begin(), name.end(), name.begin(), (int (*)(int))std::toupper);
    std::replace(name.begin(), name.end(), ' ', '_');

    const Teuchos::ParameterEntry entry = lameMaterialParameters.entry(it);
    if (entry.isType<int>()) {
      std::vector<int> propertyVector;
      propertyVector.push_back(Teuchos::getValue<int>(entry));
      matProps.insert(name, propertyVector);
    } else if (entry.isType<double>()) {
      std::vector<double> propertyVector;
      propertyVector.push_back(Teuchos::getValue<double>(entry));
      matProps.insert(name, propertyVector);
    } else if (entry.isType<std::string>()) {
      std::vector<std::string> propertyVector;
      propertyVector.push_back(Teuchos::getValue<std::string>(entry));
      matProps.insert(name, propertyVector);
    } else if (entry.isType<bool>()) {
      // Flag for reading from xml materials database is a bool -- not sent to
      // Lame
    } else {
      ALBANY_ABORT(
          " parameters for LAME material models must be of type double, int, "
          "or string.\n");
    }
  }
}

//! Instantiate a lame::MaterialModel given the model name and a set of material
//! parameters.
//  Teuchos::RCP<LameMaterial> constructLameMaterialModel(std::string const&
//  lameMaterialModelName,
Teuchos::RCP<LameMaterial>
constructLameMaterialModel(
    std::string const             lameMaterialModelName,
    Teuchos::ParameterList const& lameMaterialParameters);

#if defined(ALBANY_LAMENT)
//! Instantiate a lament::MaterialModel<ADType> given the model name and a set
//! of material parameters.
template <typename ScalarT>
inline Teuchos::RCP<lament::Material<ScalarT>>
constructLamentMaterialModel(
    std::string const&            lameMaterialModelName,
    Teuchos::ParameterList const& lameMaterialParameters)
{
  // Strings should be all upper case with spaces replaced with underscores
  std::string materialModelName = lameMaterialModelName;
  std::transform(
      materialModelName.begin(),
      materialModelName.end(),
      materialModelName.begin(),
      (int (*)(int))std::toupper);
  std::replace(materialModelName.begin(), materialModelName.end(), ' ', '_');

  LameMatProps props;
  parameterListToMatProps(lameMaterialParameters, props);

  Teuchos::RCP<lament::Material<ScalarT>> materialModel;

  if (materialModelName == "ELASTIC_NEW")
    materialModel = Teuchos::rcp(new lament::ElasticNew<ScalarT>(props));
  else if (materialModelName == "NEOHOOKEAN")
    materialModel = Teuchos::rcp(new lament::Neohookean<ScalarT>(props));
  else {
    if (materialModel.is_null())
      ALBANY_ABORT(
          " unsupported LAMENT material model: " + lameMaterialModelName +
          " (" + materialModelName + ")\n");
  }

  return materialModel;
}
#endif  // ALBANY_LAMENT

//! Return a vector containing the names of the state variables associated with
//! the given LAME material model and material parameters.
std::vector<std::string>
getStateVariableNames(
    std::string const&            lameMaterialModelName,
    Teuchos::ParameterList const& lameMaterialParameters);

//! Return a vector containing the initial values for the state variables
//! associated with the given LAME material model and material parameters.
std::vector<double>
getStateVariableInitialValues(
    std::string const&            lameMaterialModelName,
    Teuchos::ParameterList const& lameMaterialParameters);

}  // namespace LameUtils

#endif  // LAME_UTILS_HPP
