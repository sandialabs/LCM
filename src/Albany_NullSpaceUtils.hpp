// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_NULL_SPACE_UTILS_HPP
#define ALBANY_NULL_SPACE_UTILS_HPP

#include "Albany_ThyraTypes.hpp"

namespace Albany {

// Forward declaration of a helper class, used to hide Tpetra details
struct TraitsImplBase;

class RigidBodyModes
{
 public:
  //! Construct RBM object.
  RigidBodyModes(int numPDEs);

  //! Update the number of PDEs present.
  void
  setNumPDEs(int numPDEs_)
  {
    numPDEs = numPDEs_;
  }

  //! Set sizes of nullspace etc.
  void
  setParameters(
      int const  numPDEs,
      int const  numElasticityDim,
      int const  numScalar,
      int const  nullSpaceDim,
      bool const setNonElastRBM = false);

  //! Set Piro solver parameter list.
  void
  setPiroPL(const Teuchos::RCP<Teuchos::ParameterList>& piroParams);

  //! Update the parameter list.
  void
  updatePL(const Teuchos::RCP<Teuchos::ParameterList>& precParams);

  //! Is MueLu used on this problem?
  bool
  isMueLuUsed() const
  {
    return mueLuUsed;
  }

  //! Is FROSch used on this problem?
  bool
  isFROSchUsed() const
  {
    return froschUsed;
  }

  //! Pass coordinates and, if numElasticityDim > 0, the null space to ML,
  //! MueLu or FROSch. The data accessed through getCoordArrays must have
  //! been set. soln_map must be set only if using MueLu and numElasticityDim >
  //! 0. Both maps are nonoverlapping.
  void
  setCoordinatesAndNullspace(
      Teuchos::RCP<Thyra_MultiVector> const&       coordMV,
      Teuchos::RCP<Thyra_VectorSpace const> const& soln_vs = Teuchos::null,
      Teuchos::RCP<Thyra_VectorSpace const> const& soln_overlap_vs =
          Teuchos::null);

  //! Pass only the coordinates.
  void
  setCoordinates(Teuchos::RCP<Thyra_MultiVector> const& coordMV);

 private:
  int  numPDEs, numElasticityDim, numScalar, nullSpaceDim;
  bool mueLuUsed, froschUsed, setNonElastRBM;

  Teuchos::RCP<Teuchos::ParameterList> plist;

  Teuchos::RCP<Thyra_MultiVector> coordMV;

  Teuchos::RCP<TraitsImplBase> traits;
};

}  // namespace Albany

#endif  // ALBANY_NULL_SPACE_UTILS_HPP
