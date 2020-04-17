// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(PHAL_IDENTITYCOORDINATEFUNCTIONTRAITS_HPP)
#define PHAL_IDENTITYCOORDINATEFUNCTIONTRAITS_HPP

#include <Sacado_ScalarParameterLibrary.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_VerboseObject.hpp>

#include "PHAL_AlbanyTraits.hpp"

namespace PHAL {

///
/// \brief Interface for representing a coordinate function to be applied as a
/// Dirichlet boundary condition
///

template <typename EvalT>
class IdentityCoordFunctionTraits
{
 public:
  typedef typename EvalT::ScalarT ScalarT;

  ///
  /// Only constructor
  ///
  IdentityCoordFunctionTraits(Teuchos::ParameterList& params_);

  ///
  /// Destructor
  ///
  ~IdentityCoordFunctionTraits(){};

  void
  computeBCs(double* coord, std::vector<ScalarT>& BCvals, const RealType time);

  int
  getNumComponents()
  {
    return numEqn;
  }

 protected:
  // Number of equations
  int numEqn;

  // Equation offset into residual vector (Always treated as zero for now)
  int eqnOffset;

 private:
  //! Private to prohibit default or copy constructor
  IdentityCoordFunctionTraits();
  IdentityCoordFunctionTraits(const IdentityCoordFunctionTraits&);

  //! Private to prohibit copying
  IdentityCoordFunctionTraits&
  operator=(const IdentityCoordFunctionTraits&);
};
}  // namespace PHAL

// Define macro for explicit template instantiation
#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::Residual>;
#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::Jacobian>;

#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS(name)    \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name) \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name)

#endif  // PHAL_IDENTITYCOORDINATEFUNCTIONTRAITS_HPP
