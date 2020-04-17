// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_EVALUATORUTILS_HPP
#define ALBANY_EVALUATORUTILS_HPP

#include <Phalanx_Evaluator.hpp>
#include <string>
#include <vector>

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "Intrepid2_Basis.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_VerboseObject.hpp"

namespace Albany {
/*!
 * \brief Generic Functions to construct evaluators more succinctly
 */
template <typename Traits>
class EvaluatorUtilsBase
{
 public:
  virtual ~EvaluatorUtilsBase() = default;

  //! Function to create parameter list for construction of GatherSolution
  //! evaluator with standard Field names
  Teuchos::RCP<PHX::Evaluator<Traits>> virtual constructGatherSolutionEvaluator(
      bool                           isVectorField,
      Teuchos::ArrayRCP<std::string> dof_names,
      Teuchos::ArrayRCP<std::string> dof_names_dot,
      int                            offsetToFirstDOF = 0) const = 0;

  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructGatherSolutionEvaluator(
      bool               isVectorField,
      std::string const& dof_name,
      std::string const& dof_name_dot,
      int                offsetToFirstDOF = 0) const
  {
    return constructGatherSolutionEvaluator(
        isVectorField,
        arcp_str(dof_name),
        arcp_str(dof_name_dot),
        offsetToFirstDOF);
  }

  //! Function to create parameter list for construction of GatherSolution
  //! evaluator with standard Field names.
  //! Tensor rank of solution variable is 0, 1, or 2
  Teuchos::RCP<PHX::Evaluator<Traits>> virtual constructGatherSolutionEvaluator(
      int                            tensorRank,
      Teuchos::ArrayRCP<std::string> dof_names,
      Teuchos::ArrayRCP<std::string> dof_names_dot,
      int                            offsetToFirstDOF = 0) const = 0;

  //! Function to create parameter list for construction of GatherSolution
  //! evaluator with acceleration terms
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructGatherSolutionEvaluator_withAcceleration(
          bool                           isVectorField,
          Teuchos::ArrayRCP<std::string> dof_names,
          Teuchos::ArrayRCP<std::string> dof_names_dot,  // can be Teuchos::null
          Teuchos::ArrayRCP<std::string> dof_names_dotdot,
          int                            offsetToFirstDOF = 0) const = 0;

  //! Function to create parameter list for construction of GatherSolution
  //! evaluator with acceleration terms.
  //! Tensor rank of solution variable is 0, 1, or 2
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructGatherSolutionEvaluator_withAcceleration(
          int                            tensorRank,
          Teuchos::ArrayRCP<std::string> dof_names,
          Teuchos::ArrayRCP<std::string> dof_names_dot,  // can be Teuchos::null
          Teuchos::ArrayRCP<std::string> dof_names_dotdot,
          int                            offsetToFirstDOF = 0) const = 0;

  //! Same as above, but no ability to gather time dependent x_dot field
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructGatherSolutionEvaluator_noTransient(
          bool                           isVectorField,
          Teuchos::ArrayRCP<std::string> dof_names,
          int                            offsetToFirstDOF = 0) const = 0;

  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructGatherSolutionEvaluator_noTransient(
      bool               isVectorField,
      std::string const& dof_name,
      int                offsetToFirstDOF = 0) const
  {
    return constructGatherSolutionEvaluator_noTransient(
        isVectorField, arcp_str(dof_name), offsetToFirstDOF);
  }

  //! Same as above, but no ability to gather time dependent x_dot field
  //! Tensor rank of solution variable is 0, 1, or 2
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructGatherSolutionEvaluator_noTransient(
          int                            tensorRank,
          Teuchos::ArrayRCP<std::string> dof_names,
          int                            offsetToFirstDOF = 0) const = 0;

  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructGatherSolutionEvaluator_noTransient(
      int                tensorRank,
      std::string const& dof_name,
      int                offsetToFirstDOF = 0) const
  {
    return constructGatherSolutionEvaluator_noTransient(
        tensorRank,
        Teuchos::ArrayRCP<std::string>(1, dof_name),
        offsetToFirstDOF);
  }

  //! Function to create parameter list for construction of ScatterResidual
  //! evaluator with standard Field names
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructScatterResidualEvaluator(
          bool                           isVectorField,
          Teuchos::ArrayRCP<std::string> resid_names,
          int                            offsetToFirstDOF = 0,
          std::string                    scatterName = "Scatter") const = 0;

  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructScatterResidualEvaluator(
      bool               isVectorField,
      std::string const& resid_name,
      int                offsetToFirstDOF = 0,
      std::string        scatterName      = "Scatter") const
  {
    return constructScatterResidualEvaluator(
        isVectorField, arcp_str(resid_name), offsetToFirstDOF, scatterName);
  }

  //! Function to create parameter list for construction of ScatterResidual
  //! evaluator with standard Field names
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructScatterResidualEvaluatorWithExtrudedParams(
          bool                                     isVectorField,
          Teuchos::ArrayRCP<std::string>           resid_names,
          Teuchos::RCP<std::map<std::string, int>> extruded_params_levels,
          int                                      offsetToFirstDOF = 0,
          std::string scatterName = "Scatter") const = 0;

  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructScatterResidualEvaluatorWithExtrudedParams(
      bool                                     isVectorField,
      std::string const&                       resid_name,
      Teuchos::RCP<std::map<std::string, int>> extruded_params_levels,
      int                                      offsetToFirstDOF = 0,
      std::string                              scatterName = "Scatter") const
  {
    return constructScatterResidualEvaluatorWithExtrudedParams(
        isVectorField,
        arcp_str(resid_name),
        extruded_params_levels,
        offsetToFirstDOF,
        scatterName);
  }

  //! Function to create parameter list for construction of ScatterResidual
  //! evaluator with standard Field names
  //! Tensor rank of solution variable is 0, 1, or 2
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructScatterResidualEvaluator(
          int                            tensorRank,
          Teuchos::ArrayRCP<std::string> resid_names,
          int                            offsetToFirstDOF = 0,
          std::string                    scatterName = "Scatter") const = 0;

  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructScatterResidualEvaluator(
      int                tensorRank,
      std::string const& resid_name,
      int                offsetToFirstDOF = 0,
      std::string        scatterName      = "Scatter") const
  {
    return constructScatterResidualEvaluator(
        tensorRank, arcp_str(resid_name), offsetToFirstDOF, scatterName);
  }

#if defined(ALBANY_CONTACT)
  //! Function to create parameter list for construction of
  //! MortarContactResidual evaluator with standard Field names Tensor rank of
  //! solution variable is 0, 1, or 2
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructMortarContactResidualEvaluator(
          Teuchos::ArrayRCP<std::string> resid_names,
          int                            offsetToFirstDOF = 0) const = 0;

#endif

  //! Function to create parameter list for construction of
  //! GatherScalarNodalParameter
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructGatherScalarNodalParameter(
          std::string const& param_name,
          std::string const& field_name = "") const = 0;

  //! Function to create parameter list for construction of
  //! ScatterScalarNodalParameter
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructScatterScalarNodalParameter(
          std::string const& param_name,
          std::string const& field_name = "") const = 0;

  //! Function to create parameter list for construction of
  //! GatherScalarExtruded2DNodalParameter
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructGatherScalarExtruded2DNodalParameter(
          std::string const& param_name,
          std::string const& field_name = "") const = 0;

  //! Function to create parameter list for construction of
  //! ScatterScalarExtruded2DNodalParameter
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructScatterScalarExtruded2DNodalParameter(
          std::string const& param_name,
          std::string const& field_name = "") const = 0;

  //! Function to create parameter list for construction of DOFInterpolation
  //! evaluator with standard field names
  //! AGS Note 10/13: oddsetToFirstDOF is added to DOF evaluators
  //!  for template specialization of Jacobian evaluation for
  //   performance. Otherwise it was not needed. With this info,
  //   the location of the nonzero partial derivatives can be
  //   computed, and the chain rule is coded with that known sparsity.
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructDOFInterpolationEvaluator(
          std::string const& dof_names,
          int                offsetToFirstDOF = -1) const = 0;

  //! Same as above, for Interpolating the Gradient
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructDOFGradInterpolationEvaluator(
          std::string const& dof_names,
          int                offsetToFirstDOF = -1) const = 0;

  //! Interpolation functions for vector quantities
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructDOFVecInterpolationEvaluator(
          std::string const& dof_names,
          int                offsetToFirstDOF = -1) const = 0;

  //! Same as above, for Interpolating the Gradient for Vector quantities
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructDOFVecGradInterpolationEvaluator(
          std::string const& dof_names,
          int                offsetToFirstDOF = -1) const = 0;

  //! Interpolation functions for Tensor quantities
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructDOFTensorInterpolationEvaluator(
          std::string const& dof_names,
          int                offsetToFirstDOF = -1) const = 0;
  //! Same as above, for Interpolating the Gradient for Tensor quantities
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructDOFTensorGradInterpolationEvaluator(
          std::string const& dof_names,
          int                offsetToFirstDOF = -1) const = 0;

  //! Interpolation functions for scalar quantities defined on a side set
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructDOFInterpolationSideEvaluator(
          std::string const& dof_names,
          std::string const& sideSetName) const = 0;

  //! Interpolation functions for vector defined on a side set
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructDOFVecInterpolationSideEvaluator(
          std::string const& dof_names,
          std::string const& sideSetName) const = 0;

  //! Interpolation functions for gradient of quantities defined on a side set
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructDOFGradInterpolationSideEvaluator(
          std::string const& dof_names,
          std::string const& sideSetName) const = 0;

  //! Interpolation functions for gradient of vector quantities defined on a
  //! side set
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructDOFVecGradInterpolationSideEvaluator(
          std::string const& dof_names,
          std::string const& sideSetName) const = 0;

  //! Function to create parameter list for construction of
  //! GatherCoordinateVector evaluator with standard Field names
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructGatherCoordinateVectorEvaluator(
          std::string strCurrentDisp = "") const = 0;

  //! Function to create parameter list for construction of MapToPhysicalFrame
  //! evaluator with standard Field names
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructMapToPhysicalFrameEvaluator(
          const Teuchos::RCP<shards::CellTopology>&            cellType,
          const Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature,
          const Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
              intrepidBasis = Teuchos::null) const = 0;

  //! Function to create parameter list for construction of
  //! MapToPhysicalFrameSide evaluator with standard Field names
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructMapToPhysicalFrameSideEvaluator(
          const Teuchos::RCP<shards::CellTopology>&            cellType,
          const Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature,
          std::string const& sideSetName) const = 0;

  //! Function to create evaluator for restriction to side set
  Teuchos::RCP<PHX::Evaluator<Traits>> virtual constructDOFCellToSideEvaluator(
      std::string const&                        cell_dof_name,
      std::string const&                        sideSetName,
      std::string const&                        layout,
      const Teuchos::RCP<shards::CellTopology>& cellType      = Teuchos::null,
      std::string const&                        side_dof_name = "") const = 0;

  //! Combo: restriction to side plus interpolation
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructDOFCellToSideQPEvaluator(
          std::string const&                        cell_dof_name,
          std::string const&                        sideSetName,
          std::string const&                        layout,
          const Teuchos::RCP<shards::CellTopology>& cellType = Teuchos::null,
          std::string const& side_dof_name                   = "") const = 0;

  //! Function to create evaluator for prolongation to cell
  Teuchos::RCP<PHX::Evaluator<Traits>> virtual constructDOFSideToCellEvaluator(
      std::string const&                        side_dof_name,
      std::string const&                        sideSetName,
      std::string const&                        layout,
      const Teuchos::RCP<shards::CellTopology>& cellType      = Teuchos::null,
      std::string const&                        cell_dof_name = "") const = 0;

  //! Function to create evaluator NodesToCellInterpolation
  //! (=DOFInterpolation+QuadPointsToCellInterpolation)
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructNodesToCellInterpolationEvaluator(
          std::string const& dof_name,
          bool               isVectorField = false) const = 0;

  //! Function to create evaluator QuadPointsToCellInterpolation
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructQuadPointsToCellInterpolationEvaluator(
          std::string const&                  dof_name,
          const Teuchos::RCP<PHX::DataLayout> qp_layout = Teuchos::null,
          const Teuchos::RCP<PHX::DataLayout> cell_layout =
              Teuchos::null) const = 0;

  //! Function to create evaluator QuadPointsToCellInterpolation
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructSideQuadPointsToSideInterpolationEvaluator(
          std::string const& dof_name,
          std::string const& sideSetName,
          int const          fieldDim = 0) const = 0;

  //! Function to create parameter list for construction of
  //! ComputeBasisFunctions evaluator with standard Field names
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructComputeBasisFunctionsEvaluator(
          const Teuchos::RCP<shards::CellTopology>& cellType,
          const Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
                                                               intrepidBasis,
          const Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature)
          const = 0;

  //! Function to create parameter list for construction of
  //! ComputeBasisFunctionsSide evaluator with standard Field names
  Teuchos::
      RCP<PHX::Evaluator<Traits>> virtual constructComputeBasisFunctionsSideEvaluator(
          const Teuchos::RCP<shards::CellTopology>& cellType,
          const Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
                                                               intrepidBasisSide,
          const Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubatureSide,
          std::string const&                                   sideSetName,
          bool const buildNormals = false) const = 0;

 protected:
  Teuchos::ArrayRCP<std::string>
  arcp_str(std::string const& s) const
  {
    return Teuchos::ArrayRCP<std::string>(1, s);
  }
};

template <typename EvalT, typename Traits, typename ScalarType>
class EvaluatorUtilsImpl : public EvaluatorUtilsBase<Traits>
{
 public:
  typedef typename EvalT::ScalarT      ScalarT;
  typedef typename EvalT::MeshScalarT  MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  EvaluatorUtilsImpl(Teuchos::RCP<Albany::Layouts> dl);

  const EvaluatorUtilsBase<Traits>&
  getSTUtils() const
  {
    if (std::is_same<ScalarType, ScalarT>::value) {
      return *this;
    } else if (utils_ST == Teuchos::null)
      utils_ST =
          Teuchos::rcp(new EvaluatorUtilsImpl<EvalT, Traits, ScalarT>(dl));
    return *utils_ST;
  }

  const EvaluatorUtilsBase<Traits>&
  getMSTUtils() const
  {
    if (std::is_same<ScalarType, MeshScalarT>::value) {
      return *this;
    } else if (utils_MST == Teuchos::null)
      utils_MST =
          Teuchos::rcp(new EvaluatorUtilsImpl<EvalT, Traits, MeshScalarT>(dl));
    return *utils_MST;
  }

  const EvaluatorUtilsBase<Traits>&
  getPSTUtils() const
  {
    if (std::is_same<ScalarType, ParamScalarT>::value) {
      return *this;
    } else if (utils_PST == Teuchos::null)
      utils_PST =
          Teuchos::rcp(new EvaluatorUtilsImpl<EvalT, Traits, ParamScalarT>(dl));
    return *utils_PST;
  }

  const EvaluatorUtilsBase<Traits>&
  getRTUtils() const
  {
    if (utils_RT == Teuchos::null)
      utils_RT =
          Teuchos::rcp(new EvaluatorUtilsImpl<EvalT, Traits, RealType>(dl));
    return *utils_RT;
  }

  // Do not hide base class inlined methods
  using EvaluatorUtilsBase<Traits>::constructGatherSolutionEvaluator;
  using EvaluatorUtilsBase<
      Traits>::constructGatherSolutionEvaluator_noTransient;
  using EvaluatorUtilsBase<Traits>::constructScatterResidualEvaluator;
  using EvaluatorUtilsBase<
      Traits>::constructScatterResidualEvaluatorWithExtrudedParams;

  //! Function to create parameter list for construction of GatherSolution
  //! evaluator with standard Field names
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructGatherSolutionEvaluator(
      bool                           isVectorField,
      Teuchos::ArrayRCP<std::string> dof_names,
      Teuchos::ArrayRCP<std::string> dof_names_dot,
      int                            offsetToFirstDOF = 0) const;

  //! Function to create parameter list for construction of GatherSolution
  //! evaluator with standard Field names.
  //! Tensor rank of solution variable is 0, 1, or 2
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructGatherSolutionEvaluator(
      int                            tensorRank,
      Teuchos::ArrayRCP<std::string> dof_names,
      Teuchos::ArrayRCP<std::string> dof_names_dot,
      int                            offsetToFirstDOF = 0) const;

  //! Function to create parameter list for construction of GatherSolution
  //! evaluator with acceleration terms
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructGatherSolutionEvaluator_withAcceleration(
      bool                           isVectorField,
      Teuchos::ArrayRCP<std::string> dof_names,
      Teuchos::ArrayRCP<std::string> dof_names_dot,  // can be Teuchos::null
      Teuchos::ArrayRCP<std::string> dof_names_dotdot,
      int                            offsetToFirstDOF = 0) const;

  //! Function to create parameter list for construction of GatherSolution
  //! evaluator with acceleration terms.
  //! Tensor rank of solution variable is 0, 1, or 2
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructGatherSolutionEvaluator_withAcceleration(
      int                            tensorRank,
      Teuchos::ArrayRCP<std::string> dof_names,
      Teuchos::ArrayRCP<std::string> dof_names_dot,  // can be Teuchos::null
      Teuchos::ArrayRCP<std::string> dof_names_dotdot,
      int                            offsetToFirstDOF = 0) const;

  //! Same as above, but no ability to gather time dependent x_dot field
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructGatherSolutionEvaluator_noTransient(
      bool                           isVectorField,
      Teuchos::ArrayRCP<std::string> dof_names,
      int                            offsetToFirstDOF = 0) const;

  //! Same as above, but no ability to gather time dependent x_dot field
  //! Tensor rank of solution variable is 0, 1, or 2
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructGatherSolutionEvaluator_noTransient(
      int                            tensorRank,
      Teuchos::ArrayRCP<std::string> dof_names,
      int                            offsetToFirstDOF = 0) const;

  //! Function to create parameter list for construction of ScatterResidual
  //! evaluator with standard Field names
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructScatterResidualEvaluator(
      bool                           isVectorField,
      Teuchos::ArrayRCP<std::string> resid_names,
      int                            offsetToFirstDOF = 0,
      std::string                    scatterName      = "Scatter") const;

  //! Function to create parameter list for construction of ScatterResidual
  //! evaluator with standard Field names
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructScatterResidualEvaluatorWithExtrudedParams(
      bool                                     isVectorField,
      Teuchos::ArrayRCP<std::string>           resid_names,
      Teuchos::RCP<std::map<std::string, int>> extruded_params_levels,
      int                                      offsetToFirstDOF = 0,
      std::string                              scatterName = "Scatter") const;

  //! Function to create parameter list for construction of ScatterResidual
  //! evaluator with standard Field names
  //! Tensor rank of solution variable is 0, 1, or 2
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructScatterResidualEvaluator(
      int                            tensorRank,
      Teuchos::ArrayRCP<std::string> resid_names,
      int                            offsetToFirstDOF = 0,
      std::string                    scatterName      = "Scatter") const;

#if defined(ALBANY_CONTACT)
  //! Function to create parameter list for construction of
  //! MortarContactResidual evaluator with standard Field names Tensor rank of
  //! solution variable is 0, 1, or 2
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructMortarContactResidualEvaluator(
      Teuchos::ArrayRCP<std::string> resid_names,
      int                            offsetToFirstDOF = 0) const;

#endif

  //! Function to create parameter list for construction of
  //! GatherScalarNodalParameter
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructGatherScalarNodalParameter(
      std::string const& param_name,
      std::string const& field_name = "") const;

  //! Function to create parameter list for construction of
  //! ScatterScalarNodalParameter
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructScatterScalarNodalParameter(
      std::string const& param_name,
      std::string const& field_name = "") const;

  //! Function to create parameter list for construction of
  //! GatherScalarExtruded2DNodalParameter
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructGatherScalarExtruded2DNodalParameter(
      std::string const& param_name,
      std::string const& field_name = "") const;

  //! Function to create parameter list for construction of
  //! ScatterScalarExtruded2DNodalParameter
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructScatterScalarExtruded2DNodalParameter(
      std::string const& param_name,
      std::string const& field_name = "") const;

  //! Function to create parameter list for construction of DOFInterpolation
  //! evaluator with standard field names
  //! AGS Note 10/13: oddsetToFirstDOF is added to DOF evaluators
  //!  for template specialization of Jacobian evaluation for
  //   performance. Otherwise it was not needed. With this info,
  //   the location of the nonzero partial derivatives can be
  //   computed, and the chain rule is coded with that known sparsity.
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructDOFInterpolationEvaluator(
      std::string const& dof_names,
      int                offsetToFirstDOF = -1) const;

  //! Same as above, for Interpolating the Gradient
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructDOFGradInterpolationEvaluator(
      std::string const& dof_names,
      int                offsetToFirstDOF = -1) const;

  //! Interpolation functions for vector quantities
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructDOFVecInterpolationEvaluator(
      std::string const& dof_names,
      int                offsetToFirstDOF = -1) const;

  //! Same as above, for Interpolating the Gradient for Vector quantities
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructDOFVecGradInterpolationEvaluator(
      std::string const& dof_names,
      int                offsetToFirstDOF = -1) const;

  //! Interpolation functions for Tensor quantities
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructDOFTensorInterpolationEvaluator(
      std::string const& dof_names,
      int                offsetToFirstDOF = -1) const;
  //! Same as above, for Interpolating the Gradient for Tensor quantities
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructDOFTensorGradInterpolationEvaluator(
      std::string const& dof_names,
      int                offsetToFirstDOF = -1) const;

  //! Interpolation functions for scalar quantities defined on a side set
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructDOFInterpolationSideEvaluator(
      std::string const& dof_names,
      std::string const& sideSetName) const;

  //! Interpolation functions for vector defined on a side set
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructDOFVecInterpolationSideEvaluator(
      std::string const& dof_names,
      std::string const& sideSetName) const;

  //! Interpolation functions for gradient of quantities defined on a side set
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructDOFGradInterpolationSideEvaluator(
      std::string const& dof_names,
      std::string const& sideSetName) const;

  //! Interpolation functions for gradient of vector quantities defined on a
  //! side set
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructDOFVecGradInterpolationSideEvaluator(
      std::string const& dof_names,
      std::string const& sideSetName) const;

  //! Function to create parameter list for construction of
  //! GatherCoordinateVector evaluator with standard Field names
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructGatherCoordinateVectorEvaluator(
      std::string strCurrentDisp = "") const;

  //! Function to create parameter list for construction of MapToPhysicalFrame
  //! evaluator with standard Field names
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructMapToPhysicalFrameEvaluator(
      const Teuchos::RCP<shards::CellTopology>&            cellType,
      const Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature,
      const Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
          intrepidBasis = Teuchos::null) const;

  //! Function to create parameter list for construction of
  //! MapToPhysicalFrameSide evaluator with standard Field names
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructMapToPhysicalFrameSideEvaluator(
      const Teuchos::RCP<shards::CellTopology>&            cellType,
      const Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature,
      std::string const&                                   sideSetName) const;

  //! Function to create evaluator for restriction to side set
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructDOFCellToSideEvaluator(
      std::string const&                        cell_dof_name,
      std::string const&                        sideSetName,
      std::string const&                        layout,
      const Teuchos::RCP<shards::CellTopology>& cellType      = Teuchos::null,
      std::string const&                        side_dof_name = "") const;

  //! Combo: restriction to side plus interpolation
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructDOFCellToSideQPEvaluator(
      std::string const&                        cell_dof_name,
      std::string const&                        sideSetName,
      std::string const&                        layout,
      const Teuchos::RCP<shards::CellTopology>& cellType      = Teuchos::null,
      std::string const&                        side_dof_name = "") const;

  //! Function to create evaluator for prolongation to cell
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructDOFSideToCellEvaluator(
      std::string const&                        side_dof_name,
      std::string const&                        sideSetName,
      std::string const&                        layout,
      const Teuchos::RCP<shards::CellTopology>& cellType      = Teuchos::null,
      std::string const&                        cell_dof_name = "") const;

  //! Function to create evaluator NodesToCellInterpolation
  //! (=DOFInterpolation+QuadPointsToCellInterpolation)
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructNodesToCellInterpolationEvaluator(
      std::string const& dof_name,
      bool               isVectorField = false) const;

  //! Function to create evaluator QuadPointsToCellInterpolation
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructQuadPointsToCellInterpolationEvaluator(
      std::string const&                  dof_name,
      const Teuchos::RCP<PHX::DataLayout> qp_layout   = Teuchos::null,
      const Teuchos::RCP<PHX::DataLayout> cell_layout = Teuchos::null) const;

  //! Function to create evaluator QuadPointsToCellInterpolation
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructSideQuadPointsToSideInterpolationEvaluator(
      std::string const& dof_name,
      std::string const& sideSetName,
      int const          fieldDim = 0) const;

  //! Function to create parameter list for construction of
  //! ComputeBasisFunctions evaluator with standard Field names
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructComputeBasisFunctionsEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
                                                           intrepidBasis,
      const Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature) const;

  //! Function to create parameter list for construction of
  //! ComputeBasisFunctionsSide evaluator with standard Field names
  Teuchos::RCP<PHX::Evaluator<Traits>>
  constructComputeBasisFunctionsSideEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
                                                           intrepidBasisSide,
      const Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubatureSide,
      std::string const&                                   sideSetName,
      bool const buildNormals = false) const;

 private:
  //! Evaluator Utils with different ScalarType. Mutable, so we can have getters
  //! with JIT build. NOTE: we CAN'T create them in the constructor, since we
  //! would have a never-ending construction.
  mutable Teuchos::RCP<EvaluatorUtilsBase<Traits>> utils_ST;
  mutable Teuchos::RCP<EvaluatorUtilsBase<Traits>> utils_MST;
  mutable Teuchos::RCP<EvaluatorUtilsBase<Traits>> utils_PST;
  mutable Teuchos::RCP<EvaluatorUtilsBase<Traits>> utils_RT;

  //! Struct of PHX::DataLayout objects defined all together.
  Teuchos::RCP<Albany::Layouts> dl;
};

template <typename EvalT, typename Traits>
using EvaluatorUtils =
    EvaluatorUtilsImpl<EvalT, Traits, typename EvalT::ScalarT>;

}  // Namespace Albany

#endif  // ALBANY_EVALUATORUTILS_HPP
