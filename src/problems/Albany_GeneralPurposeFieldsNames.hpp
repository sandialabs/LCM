#ifndef ALBANY_GENERAL_PURPOSE_FIELDS_NAMES_HPP
#define ALBANY_GENERAL_PURPOSE_FIELDS_NAMES_HPP

#include <string>

namespace Albany {

// Hard coding names for some fields that are used in many evaluators

static std::string const coord_vec_name        = "Coord Vec";
static std::string const weights_name          = "Weights";
static std::string const weighted_measure_name = "Weighted Measure";
static std::string const bf_name               = "BF";
static std::string const grad_bf_name          = "Grad BF";
static std::string const weighted_bf_name      = "wBF";
static std::string const weighted_grad_bf_name = "wGrad BF";
static std::string const jacobian_name         = "Jacobian";
static std::string const jacobian_det_name     = "Jacobian Det";
static std::string const jacobian_inv_name     = "Jacobian Inv";
static std::string const tangents_name         = "Tangents";
static std::string const metric_name           = "Metric";
static std::string const metric_det_name       = "Metric Det";
static std::string const metric_inv_name       = "Metric Inv";
static std::string const normal_name           = "Normal";

}  // namespace Albany

#endif  // ALBANY_GENERAL_PURPOSE_FIELDS_NAMES_HPP
