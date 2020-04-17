// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef PHAL_WORKSET_HPP
#define PHAL_WORKSET_HPP

#include <list>
#include <set>
#include <string>

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_SacadoTypes.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Albany_TpetraTypes.hpp"
#include "Kokkos_ViewFactory.hpp"
#include "PHAL_Setup.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_RCP.hpp"

// Forward declarations
namespace Albany {
class AbstractDiscretization;
class CombineAndScatterManager;
class DistributedParameterLibrary;
// Forward declaration needed for Schwarz coupling
class Application;
}  // namespace Albany

namespace PHAL {

struct Workset
{
  Workset() {}

  unsigned int numCells{0};
  unsigned int wsIndex{0};
  unsigned int numEqs{0};

  // Solution vector (and time derivatives)
  Teuchos::RCP<Thyra_Vector const> x;
  Teuchos::RCP<Thyra_Vector const> xdot;
  Teuchos::RCP<Thyra_Vector const> xdotdot;

  Teuchos::RCP<ParamVec> params;

  // Component of Tangent vector direction along x, xdot, xdotdot, and p.
  // These are used to compute df/dx*Vx + df/dxdot*Vxdot + df/dxdotdot*Vxdotdot
  // + df/dp*Vp.
  Teuchos::RCP<const Thyra_MultiVector> Vx;
  Teuchos::RCP<const Thyra_MultiVector> Vxdot;
  Teuchos::RCP<const Thyra_MultiVector> Vxdotdot;
  Teuchos::RCP<const Thyra_MultiVector> Vp;

  // These are residual related.
  Teuchos::RCP<Thyra_Vector>      f;
  Teuchos::RCP<Thyra_LinearOp>    Jac;
  Teuchos::RCP<Thyra_MultiVector> JV;
  Teuchos::RCP<Thyra_MultiVector> fp;
  Teuchos::RCP<Thyra_MultiVector> fpV;
  Teuchos::RCP<Thyra_MultiVector> Vp_bc;

  Albany::DeviceView1d<ST>      f_kokkos;
  Albany::DeviceLocalMatrix<ST> Jac_kokkos;

  Teuchos::RCP<const Albany::NodeSetList>      nodeSets;
  Teuchos::RCP<const Albany::NodeSetCoordList> nodeSetCoords;
  Teuchos::RCP<const Albany::NodeSetGIDsList>  nodeSetGIDs;
  Teuchos::RCP<const Albany::SideSetList>      sideSets;

  // jacobian and mass matrix coefficients for matrix fill
  double j_coeff{0.0};
  double m_coeff{0.0};  // d(x_dot)/dx_{new}
  double n_coeff{0.0};  // d(x_dotdot)/dx_{new}

  double current_time{0.0};
  double time_step{0.0};

  // flag indicating whether to sum tangent derivatives, i.e.,
  // compute alpha*df/dxdot*Vxdot + beta*df/dx*Vx + omega*df/dxddotot*Vxdotdot +
  // df/dp*Vp or compute alpha*df/dxdot*Vxdot + beta*df/dx*Vx +
  // omega*df/dxdotdot*Vxdotdot and df/dp*Vp separately
  int num_cols_x{0};
  int num_cols_p{0};
  int param_offset{0};

  // Distributed parameter derivatives
  Teuchos::RCP<Albany::DistributedParameterLibrary>               distParamLib;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double>>> local_Vp;

  std::string dist_param_deriv_name{""};
  bool        transpose_dist_param_deriv{false};

  std::vector<PHX::index_size_type> Jacobian_deriv_dims;
  std::vector<PHX::index_size_type> Tangent_deriv_dims;

  Albany::WorksetConn                           wsElNodeEqID;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>      wsElNodeID;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*>> wsCoords;
  Teuchos::ArrayRCP<double>                     wsSphereVolume;
  Teuchos::ArrayRCP<double*>                    wsLatticeOrientation;
  std::string                                   EBName{""};

  // Needed for Schwarz coupling and for dirichlet conditions based on dist
  // parameters.
  Teuchos::RCP<Albany::AbstractDiscretization> disc;

  // Needed for Schwarz coupling
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> apps_;
  Teuchos::RCP<Albany::Application>                    current_app_;
  Teuchos::ArrayRCP<double*>                           cell_boundary_indicator;
  Teuchos::ArrayRCP<double*>                           face_boundary_indicator;
  Teuchos::ArrayRCP<double*>                           edge_boundary_indicator;
  std::map<GO, double*>                                node_boundary_indicator;
  std::set<int>                                        fixed_dofs_;
  bool                                                 is_schwarz_bc_{false};

  int spatial_dimension_{0};

  Albany::StateArray*              stateArrayPtr{nullptr};
  Teuchos::RCP<Tpetra_MultiVector> auxDataPtrT;

  bool transientTerms{false};
  bool accelerationTerms{false};

  // Flag indicating whether to ignore residual calculations in the
  // Jacobian calculation.  This only works for some problems where the
  // the calculation of the Jacobian doesn't require calculation of the
  // residual (such as linear problems), but if it does work it can
  // significantly reduce Jacobian calculation cost.
  bool ignore_residual{false};

  // Flag indicated whether we are solving the adjoint operator or the
  // forward operator.  This is used in the Albany application when
  // either the Jacobian or the transpose of the Jacobian is scattered.
  bool is_adjoint{false};

  // New field manager response stuff
  Teuchos::RCP<Teuchos::Comm<int> const> comm;

  // Combine and Scatter manager (for import-export of responses derivatives),
  // for both solution (x) and distributed parameter (p)
  Teuchos::RCP<const Albany::CombineAndScatterManager> x_cas_manager;
  Teuchos::RCP<const Albany::CombineAndScatterManager> p_cas_manager;

  // Response vector g and its derivatives
  Teuchos::RCP<Thyra_Vector>      g;
  Teuchos::RCP<Thyra_MultiVector> dgdx;
  Teuchos::RCP<Thyra_MultiVector> dgdxdot;
  Teuchos::RCP<Thyra_MultiVector> dgdxdotdot;
  Teuchos::RCP<Thyra_MultiVector> dgdp;

  // Overlapped version of response derivatives
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdx;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdxdot;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdxdotdot;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdp;

  // List of saved MDFields (needed for memoization)
  Teuchos::RCP<const StringSet> savedMDFields;

  // Meta-function class encoding T<EvalT::ScalarT> given EvalT
  // where T is any lambda expression (typically a placeholder expression)
  template <typename T>
  struct ApplyEvalT
  {
    template <typename EvalT>
    struct apply
    {
      typedef
          typename Sacado::mpl::apply<T, typename EvalT::ScalarT>::type type;
    };
  };

  // Meta-function class encoding RCP<ValueTypeSerializer<int,T> > for a given
  // type T.  This is to eliminate an error when using a placeholder expression
  // for the same thing in CreateLambdaKeyMap below
  struct ApplyVTS
  {
    template <typename T>
    struct apply
    {
      typedef Teuchos::RCP<Teuchos::ValueTypeSerializer<int, T>> type;
    };
  };

  void
  print(std::ostream& os)
  {
    os << "Printing workset data:" << std::endl;
    os << "\tEB name : " << EBName << std::endl;
    os << "\tnumCells : " << numCells << std::endl;
    os << "\twsElNodeEqID : " << std::endl;
    for (unsigned int i = 0; i < wsElNodeEqID.extent(0); i++)
      for (unsigned int j = 0; j < wsElNodeEqID.extent(1); j++)
        for (unsigned int k = 0; k < wsElNodeEqID.extent(2); k++)
          os << "\t\twsElNodeEqID(" << i << "," << j << "," << k
             << ") = " << wsElNodeEqID(i, j, k) << '\n';
    os << "\twsCoords : " << std::endl;
    for (int i = 0; i < wsCoords.size(); i++)
      for (int j = 0; j < wsCoords[i].size(); j++)
        os << "\t\tcoord0:" << wsCoords[i][j][0] << "][" << wsCoords[i][j][1]
           << std::endl;
  }
};

}  // namespace PHAL

#endif  // PHAL_WORKSET_HPP
