// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <fstream>

#include "Albany_Macros.hpp"
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
KCPermeability<EvalT, Traits>::KCPermeability(Teuchos::ParameterList& p)
    : kcPermeability(
          p.get<std::string>("Kozeny-Carman Permeability Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* elmd_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  std::string type =
      elmd_list->get("Kozeny-Carman Permeability Type", "Constant");
  if (type == "Constant") {
    is_constant    = true;
    constant_value = elmd_list->get(
        "Value", 1.0e-5);  // default value=1, identical to Terzaghi stress

    // Add Kozeny-Carman Permeability as a Sacado-ized parameter
    this->registerSacadoParameter("Kozeny-Carman Permeability", paramLib);
  } else {
    ALBANY_ABORT("Invalid Kozeny-Carman Permeability type " << type);
  }

  // Optional dependence on Temperature (E = E_ + dEdT * T)
  // Switched ON by sending Temperature field in p

  if (p.isType<std::string>("Porosity Name")) {
    Teuchos::RCP<PHX::DataLayout> scalar_dl =
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
    porosity =
        decltype(porosity)(p.get<std::string>("Porosity Name"), scalar_dl);
    this->addDependentField(porosity);
    isPoroElastic = true;

  } else {
    isPoroElastic = false;
  }

  this->addEvaluatedField(kcPermeability);
  this->setName("Kozeny-Carman Permeability" + PHX::print<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
KCPermeability<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(kcPermeability, fm);
  if (!is_constant) this->utils.setFieldData(coordVec, fm);
  if (isPoroElastic) this->utils.setFieldData(porosity, fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
KCPermeability<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  if (is_constant) {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        kcPermeability(cell, qp) = constant_value;
      }
    }
  }
  if (isPoroElastic) {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        // Cozeny Karman permeability equation
        kcPermeability(cell, qp) =
            constant_value * porosity(cell, qp) * porosity(cell, qp) *
            porosity(cell, qp) /
            ((1.0 - porosity(cell, qp)) * (1.0 - porosity(cell, qp)));
      }
    }
  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename KCPermeability<EvalT, Traits>::ScalarT&
KCPermeability<EvalT, Traits>::getValue(std::string const& n)
{
  if (n == "Kozeny-Carman Permeability") return constant_value;
  ALBANY_ABORT(
      std::endl
      << "Error! Logic error in getting paramter " << n
      << " in KCPermeability::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}  // namespace LCM
