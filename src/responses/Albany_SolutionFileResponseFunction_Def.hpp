// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_SolutionFileResponseFunction.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Teuchos_CommHelpers.hpp"

namespace Albany {

template <class Norm>
SolutionFileResponseFunction<Norm>::SolutionFileResponseFunction(const Teuchos::RCP<Teuchos_Comm const>& comm)
    : SamplingBasedScalarResponseFunction(comm), solutionLoaded(false)
{
  // Nothing to be done here
}

template <class Norm>
void
SolutionFileResponseFunction<Norm>::evaluateResponse(
    double const /*current_time*/,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& /*xdot*/,
    Teuchos::RCP<Thyra_Vector const> const& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*p*/,
    Teuchos::RCP<Thyra_Vector> const& g)
{
  int MMFileStatus = 0;

  // Read the reference solution for comparison from "reference_solution.dat"

  // Note that this is of MatrixMarket array real general format

  if (!solutionLoaded) {
    RefSoln      = Thyra::createMember(x->space());
    MMFileStatus = MatrixMarketFile("reference_solution.dat", RefSoln);

    ALBANY_PANIC(
        MMFileStatus != 0,
        std::endl
            << "MatrixMarketFile, file " __FILE__ " line " << __LINE__ << " returned " << MMFileStatus << std::endl);

    solutionLoaded = true;
  }

  if (diff.is_null()) {
    // Build a vector to hold the difference between the actual and reference
    // solutions
    diff = Thyra::createMember(x->space());
    diff->assign(0.0);
  }

  // Thyra vectors do not support update method with 2 vectors, so we need to
  // use 'linear_combination'
  Teuchos::Array<ST> coeffs(2);
  coeffs[0] = 1.0;
  coeffs[1] = -1.0;
  Teuchos::Array<Teuchos::Ptr<Thyra_Vector const>> vecs(2);
  vecs[0] = x.ptr();
  vecs[1] = RefSoln.ptr();
  diff->linear_combination(coeffs, vecs, 0.0);

  // Get the norm
  g->assign(Norm::Norm(*diff));
}

template <class Norm>
void
SolutionFileResponseFunction<Norm>::evaluateGradient(
    double const /*current_time*/,
    Teuchos::RCP<Thyra_Vector const> const& x,
    Teuchos::RCP<Thyra_Vector const> const& /*xdot*/,
    Teuchos::RCP<Thyra_Vector const> const& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*p*/,
    ParamVec* /*deriv_p*/,
    Teuchos::RCP<Thyra_Vector> const&      g,
    Teuchos::RCP<Thyra_MultiVector> const& dg_dx,
    Teuchos::RCP<Thyra_MultiVector> const& dg_dxdot,
    Teuchos::RCP<Thyra_MultiVector> const& dg_dxdotdot,
    Teuchos::RCP<Thyra_MultiVector> const& dg_dp)
{
  int MMFileStatus = 0;
  if (!solutionLoaded) {
    RefSoln      = Thyra::createMember(x->space());
    MMFileStatus = MatrixMarketFile("reference_solution.dat", RefSoln);

    ALBANY_PANIC(
        MMFileStatus != 0,
        std::endl
            << "MatrixMarketFile, file " __FILE__ " line " << __LINE__ << " returned " << MMFileStatus << std::endl);

    solutionLoaded = true;
  }

  if (!g.is_null()) {
    if (diff.is_null()) {
      // Build a vector to hold the difference between the actual and reference
      // solutions
      diff = Thyra::createMember(x->space());
      diff->assign(0.0);
    }

    // Thyra vectors do not support update method with 2 vectors, so we need to
    // use 'linear_combination'
    Teuchos::Array<ST> coeffs(2);
    coeffs[0] = 1.0;
    coeffs[1] = -1.0;
    Teuchos::Array<Teuchos::Ptr<Thyra_Vector const>> vecs(2);
    vecs[0] = x.ptr();
    vecs[1] = RefSoln.ptr();
    diff->linear_combination(coeffs, vecs, 0.0);

    // Get the norm
    g->assign(Norm::Norm(*diff));
  }

  // Evaluate dg/dx
  if (!dg_dx.is_null()) {
    ALBANY_PANIC(dg_dx->domain()->dim() != 1, "Error! dg_dx has more than one column.\n");
    Norm::NormDerivative(*x, *RefSoln, *dg_dx->col(0));
  }

  // Evaluate dg/dxdot
  if (!dg_dxdot.is_null()) {
    dg_dxdot->assign(0.0);
  }

  // Evaluate dg/dxdotdot
  if (!dg_dxdotdot.is_null()) {
    dg_dxdotdot->assign(0.0);
  }

  // Evaluate dg/dp
  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}

template <class Norm>
int
SolutionFileResponseFunction<Norm>::MatrixMarketFile(char const* filename, Teuchos::RCP<Thyra_MultiVector> const& mv)
{
  int const lineLength  = 1025;
  int const tokenLength = 35;
  char      line[lineLength];
  char      token1[tokenLength];
  char      token2[tokenLength];
  char      token3[tokenLength];
  char      token4[tokenLength];
  char      token5[tokenLength];
  int       M, N;

  FILE* handle = 0;

  handle = fopen(filename, "r");  // Open file
  if (handle == 0)
    // file not found
    ALBANY_ABORT(std::endl << "Reference solution file \" " << filename << " \" not found" << std::endl);

  // Check first line, which should be "%%MatrixMarket matrix coordinate real
  // general" (without quotes)
  if (fgets(line, lineLength, handle) == 0)

    ALBANY_ABORT(
        std::endl
        << "Reference solution: MatrixMarket file is not in the proper "
           "format."
        << std::endl);

  if (sscanf(line, "%s %s %s %s %s", token1, token2, token3, token4, token5) == 0)

    ALBANY_ABORT(
        std::endl
        << "Incorrect number of arguments on first line of reference "
           "solution file."
        << std::endl);

  if (strcmp(token1, "%%MatrixMarket") || strcmp(token2, "matrix") || strcmp(token3, "array") ||
      strcmp(token4, "real") || strcmp(token5, "general"))

    ALBANY_ABORT(
        std::endl
        << "Incorrect type of arguments on first line of reference "
           "solution file."
        << std::endl);

  // Next, strip off header lines (which start with "%")
  do {
    if (fgets(line, lineLength, handle) == 0)
      ALBANY_ABORT(std::endl << "Reference solution file: invalid comment line." << std::endl);
  } while (line[0] == '%');

  // Next get problem dimensions: M, N
  if (sscanf(line, "%d %d", &M, &N) == 0)

    ALBANY_ABORT(std::endl << "Reference solution file: cannot compute problem dimensions" << std::endl);

  // Compute the offset for each processor for when it should start storing
  // values
  const auto spmd_vs = getSpmdVectorSpace(mv->range());
  int        offset;
  // map.Comm().ScanSum(&numMyPoints, &offset, 1); // ScanSum will compute
  // offsets for us offset -= numMyPoints; // readjust for my PE

  // Line to start reading in reference file
  //  offset = map.MinMyGID();

  if (spmd_vs->getComm()->getRank() == 0) {
    std::cout << "Reading reference solution from file \"" << filename << "\"" << std::endl;
    std::cout << "Reference solution contains " << N << " vectors, each with " << M << " rows." << std::endl;
    std::cout << std::endl;
  }

  // Now construct vector/multivector
  ALBANY_PANIC(
      N != static_cast<int>(mv->domain()->dim()),
      "Error! Input file is storing a Thyra MultiVector with a number of "
      "vectors "
      "different from the what was expected.\n");

  auto vals    = getNonconstLocalData(mv);
  auto indexer = createGlobalLocalIndexer(spmd_vs);
  for (int j = 0; j < N; j++) {
    Teuchos::ArrayRCP<ST> v = vals[j];

    // Now read in each value and store to the local portion of the array if the
    // row is owned.
    ST V;
    for (int i = 0; i < M; i++) {                // i is rownumber in file, or the GID
      if (fgets(line, lineLength, handle) == 0)  // Can't read the line

        ALBANY_ABORT(
            std::endl
            << "Reference solution file: cannot read line number " << i + offset << " in file." << std::endl);

      const LO lid = indexer->getLocalElement(i);
      if (lid >= 0) {  // we own this data value
        if (sscanf(line, "%lg\n", &V) == 0) {
          ALBANY_ABORT("Reference solution file: cannot parse line number " << i << " in file.\n");
        }
        v[lid] = V;
      }
    }
  }

  if (fclose(handle)) {
    ALBANY_ABORT("Cannot close reference solution file.\n");
  }

  return 0;
}

}  // namespace Albany
