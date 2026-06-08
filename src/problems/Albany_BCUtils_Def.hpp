// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include <Phalanx_Evaluator_Factory.hpp>

#include "ACEcommon.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_Macros.hpp"

namespace {
std::string const decorator = "Evaluator for ";

// Name decorator.
inline std::string
evaluatorsToBuildName(std::string const& bc_name)
{
  std::stringstream ess;
  ess << decorator << bc_name;
  return ess.str();
}
}  // namespace

template <>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
Albany::BCUtils<Albany::NeumannTraits>::constructBCEvaluators(
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
    const Teuchos::RCP<Albany::MaterialDatabase>& materialDB)
{
  using PHAL::AlbanyTraits;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  if (!haveBCSpecified(params)) {  // If the BC sublist is not in the input file,
    // but we are inside this function, this means that
    // node sets are contained in the Exodus file but are not defined in the
    // problem statement.This is OK, we
    // just don't do anything

    return Teuchos::null;
  }

  // Build the list of evaluators to build, with all the needed parameters
  std::map<std::string, RCP<Teuchos::ParameterList>> evaluators_to_build;
  buildEvaluatorsList(
      evaluators_to_build, meshSpecs, bcNames, dof_names, isVectorField, offsetToFirstDOF, conditions, offsets, dl, params, paramLib, materialDB);

  // Build Field Evaluators for each evaluation type
  PHX::EvaluatorFactory<AlbanyTraits, Albany::NeumannTraits::factory_type> factory;

  Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator_TemplateManager<AlbanyTraits>>>> evaluators;
  evaluators = factory.buildEvaluators(evaluators_to_build);

  std::string allBC = "Evaluator for all Neumann BCs";

  return buildFieldManager(evaluators, allBC, dl->dummy);
}

template <>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
Albany::BCUtils<Albany::NeumannTraits>::constructBCEvaluators(
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
    const Teuchos::RCP<Albany::MaterialDatabase>&                        materialDB)
{
  using PHAL::AlbanyTraits;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  if (!haveBCSpecified(params)) {  // If the BC sublist is not in the input file,
    // but we are inside this function, this means that
    // node sets are contained in the Exodus file but are not defined in the
    // problem statement.This is OK, we
    // just don't do anything

    return Teuchos::null;
  }

  // Build the list of evaluators to build, with all the needed parameters
  std::map<std::string, RCP<Teuchos::ParameterList>> evaluators_to_build;
  buildEvaluatorsList(
      evaluators_to_build, meshSpecs, bcNames, dof_names, isVectorField, offsetToFirstDOF, conditions, offsets, dl, params, paramLib, materialDB);

  // Build Field Evaluators for each evaluation type
  PHX::EvaluatorFactory<AlbanyTraits, Albany::NeumannTraits::factory_type> factory;

  RCP<std::vector<RCP<PHX::Evaluator_TemplateManager<AlbanyTraits>>>> evaluators;
  evaluators = factory.buildEvaluators(evaluators_to_build);

  std::string                          allBC = "Evaluator for all Neumann BCs";
  RCP<PHX::FieldManager<AlbanyTraits>> fm    = buildFieldManager(evaluators, allBC, dl->dummy);

  std::vector<RCP<PHX::Evaluator<AlbanyTraits>>>::const_iterator it;
  for (it = extra_evaluators.begin(); it != extra_evaluators.end(); ++it) {
    fm->registerEvaluatorForAllEvaluationTypes(*it);
  }

  return fm;
}

template <>
void
Albany::BCUtils<Albany::NeumannTraits>::buildEvaluatorsList(
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
    const Teuchos::RCP<Albany::MaterialDatabase>&                materialDB)
{
  using PHAL::AlbanyTraits;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::string;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  // Drop into the "Neumann BCs" sublist
  ParameterList BCparams = params->sublist(traits_type::bcParamsPl);
  BCparams.validateParameters(*(traits_type::getValidBCParameters(meshSpecs->ssNames, bcNames, conditions)), 0);

  RCP<std::vector<string>> bcs = rcp(new std::vector<string>);

  // Check for all possible standard BCs (every dof on every sideset) to see
  // which is set
  for (std::size_t i = 0; i < meshSpecs->ssNames.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      for (std::size_t k = 0; k < conditions.size(); k++) {
        // construct input.xml string like:
        // "NBC on SS sidelist_12 for DOF T set dudn"
        //  or
        // "NBC on SS sidelist_12 for DOF T set (dudx, dudy)"
        // or
        // "NBC on SS surface_1 for DOF all set P"

        // Set logic for certain NBCs which allow array inputs
        bool allowArrayNBC = false;
        if ((conditions[k] == "robin") || (conditions[k] == "radiate") || (conditions[k].find("(") < conditions[k].length())) {
          allowArrayNBC = true;
        }

        string ss = traits_type::constructBCName(meshSpecs->ssNames[i], bcNames[j], conditions[k]);

        // Have a match of the line in input.xml

        if (BCparams.isParameter(ss)) {
          //           std::cout << "Constructing NBC: " << ss << std::endl;

          ALBANY_PANIC(
              BCparams.isType<string>(ss),
              "NBC array information in XML/YAML file must be of type "
              "Array(double)\n");

          // These are read in the Albany::Neumann constructor
          // (PHAL_Neumann_Def.hpp)

          RCP<ParameterList> p = rcp(new ParameterList);

          p->set<int>("Type", traits_type::type);

          p->set<RCP<ParamLib>>("Parameter Library", paramLib);

          p->set<string>("Side Set ID", meshSpecs->ssNames[i]);
          p->set<Teuchos::Array<int>>("Equation Offset", offsets[j]);
          p->set<RCP<Albany::Layouts>>("Layouts Struct", dl);
          p->set<RCP<MeshSpecsStruct>>("Mesh Specs Struct", meshSpecs);

          p->set<string>("Coordinate Vector Name", "Coord Vec");
          p->set<int>("Cubature Degree", BCparams.get("Cubature Degree", 0));  // if set to zero, the
                                                                               // cubature degree of the
                                                                               // side will be set to that
                                                                               // of the element

          if (conditions[k] == "robin" || conditions[k] == "radiate") {
            p->set<string>("DOF Name", dof_names[j]);
            p->set<bool>("Vector Field", isVectorField);

            if (isVectorField)
              p->set<RCP<DataLayout>>("DOF Data Layout", dl->node_vector);
            else
              p->set<RCP<DataLayout>>("DOF Data Layout", dl->node_scalar);
          }

          // Pass the input file line
          p->set<string>("Neumann Input String", ss);

          Teuchos::Array<double> niv = BCparams.get<Teuchos::Array<double>>(ss);
          // Note, we use a Teuchos::Array  here to allow the user to specify
          // multiple components of the traction vector.
          // This is only allowed for certain BCs (see how allowArrayNBC) is
          // set.
          if (!allowArrayNBC) {
            if (niv.size() != 1) {
              ALBANY_ABORT(
                  "NBC takes a scalar value.  You attempted to provide an "
                  "array!");
            }
          } else {
            if ((conditions[k] == "robin") || (conditions[k] == "radiate")) {
              if (niv.size() != 2) {
                ALBANY_ABORT("Robin NBC takes a 2-array!");
              }
            } else {
              if (niv.size() != meshSpecs->numDim) {
                ALBANY_ABORT("Traction NBC takes an array of size numDim!");
              }
            }
          }

          p->set<Teuchos::Array<double>>("Neumann Input Value", niv);
          p->set<string>("Neumann Input Conditions", conditions[k]);

          // If we are doing a Neumann internal boundary with a "scaled jump",
          // the material DB database needs to be passed to the BC object
          // Note: 'robin' is a very generic name. It is ok to allow some
          // 'complex'
          //       robin conditions (with a scaled jump), but we should allow
          //       one to use 'robin' bc for the classic du/dn + alpha*u = g,
          //       which means the user should not have to specify a material DB

          if (conditions[k] == "scaled jump" || conditions[k] == "radiate") {
            ALBANY_PANIC(materialDB == Teuchos::null, "This BC needs a material database specified");
          }
          p->set<RCP<Albany::MaterialDatabase>>("MaterialDB", materialDB);

          // Inputs: X, Y at nodes, Cubature, and Basis
          // p->set<string>("Node Variable Name", "Neumann");

          evaluators_to_build[evaluatorsToBuildName(ss)] = p;

          bcs->push_back(ss);
        }
      }
    }
  }

  ///
  /// Time dependent BC specific
  ///
  for (std::size_t i = 0; i < meshSpecs->ssNames.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      for (std::size_t k = 0; k < conditions.size(); k++) {
        // construct input.xml string like:
        // "Time Dependent NBC on SS sidelist_12 for DOF T set dudn"
        //  or
        // "Time Dependent NBC on SS sidelist_12 for DOF T set (dudx, dudy)"
        // or
        // "Time Dependent NBC on SS surface_1 for DOF all set P"

        // Set logic for certain NBCs which allow array inputs
        bool allowArrayNBC = false;
        if ((conditions[k] == "robin") || (conditions[k] == "radiate") || (conditions[k].find("(") < conditions[k].length())) {
          allowArrayNBC = true;
        }

        string ss = traits_type::constructTimeDepBCName(meshSpecs->ssNames[i], bcNames[j], conditions[k]);

        // Have a match of the line in input.xml

        if (BCparams.isSublist(ss)) {
          // grab the sublist
          ParameterList& sub_list = BCparams.sublist(ss);

          //           std::cout << "Constructing Time Dependent NBC: " << ss <<
          //           std::endl;

          // These are read in the LCM::TimeTracBC constructor
          // (LCM/evaluators/TimeTrac_Def.hpp)

          RCP<ParameterList> p = rcp(new ParameterList);

          p->set<int>("Type", traits_type::typeTd);

          Teuchos::Array<RealType> timevals;
          if (sub_list.isParameter("Time Values")) {
            timevals = sub_list.get<Teuchos::Array<RealType>>("Time Values");
          } else {
            if (sub_list.isParameter("Time Values File")) {
              std::string const          t_file       = sub_list.get<std::string>("Time Values File");
              std::vector<RealType>      timevals_vec = LCM::vectorFromFile(t_file);
              Teuchos::ArrayView<double> timevals_av  = Teuchos::arrayViewFromVector(timevals_vec);
              timevals                                = Teuchos::Array<double>(timevals_av);
            } else {
              ALBANY_ABORT("'Time Values' or 'Time Values File' are not specified!");
            }
          }

          // Note, we use a TwoDArray here to allow the user to specify
          // multiple components of the traction vector at each "time" step.
          // This is only allowed for certain BCs (see how allowArrayNBC) is
          // set.
          Teuchos::TwoDArray<RealType> bcvals;
          if (sub_list.isParameter("BC Values")) {
            bcvals = sub_list.get<Teuchos::TwoDArray<RealType>>("BC Values");
          } else {
            if (sub_list.isParameter("BC Values File")) {
              std::string const                  bc_file     = sub_list.get<std::string>("BC Values File");
              std::vector<std::vector<RealType>> bc_file_vec = LCM::twoDvectorFromFile(bc_file);
              auto                               nRows       = bc_file_vec.size();
              if (nRows < 1) {
                ALBANY_ABORT("'Invalid size for 'BC Values File' array!");
              }
              auto nCols = bc_file_vec[0].size();
              for (int i = 1; i < nRows; i++) {
                auto nColsi = bc_file_vec[i].size();
                if (nColsi != nCols) {
                  ALBANY_ABORT("'Invalid size for 'BC Values File' array!");
                }
              }
              bcvals = Teuchos::TwoDArray<RealType>(nRows, nCols);
              for (int i = 0; i < nRows; i++) {
                std::vector<RealType> veci = bc_file_vec[i];
                for (int j = 0; j < nCols; j++) {
                  bcvals(i, j) = veci[j];
                }
              }
            } else {
              ALBANY_ABORT("'BC Values' or 'BC Values File' are not specified!");
            }
          }

          // Check that bcvals and timevals have the same size.  If they do not,
          // throw an error.
          if (timevals.size() != bcvals.getNumRows()) {
            ALBANY_ABORT(
                "'Time Values' array must have same length as 'BC Values' "
                "array!");
          }

          // IKT, 2/15/2020: Currently, the code downstream of this
          // assumes bcvals is a scalar for all but a few NBCs (see comment
          // above). Throw an error if user attempts to specify array for NBCs
          // where this is not allowed.
          if (!allowArrayNBC) {
            if (bcvals.getNumCols() != 1) {
              ALBANY_ABORT(
                  "Time Dependent NBC takes 1D array for 'BC Values'.  You "
                  "attempted to provide a multi-D array!");
            }
          } else {
            if ((conditions[k] == "robin") || (conditions[k] == "radiate")) {
              if (bcvals.getNumCols() != 2) {
                ALBANY_ABORT(
                    "Time Dependent robin NBC takes a 2-array for 'BC Values' "
                    "at each time!");
              }
            } else {
              if (bcvals.getNumCols() != meshSpecs->numDim) {
                ALBANY_ABORT(
                    "Time Dependent traction NBC takes an array of size numDim "
                    "for 'BC Values' at each time!");
              }
            }
          }

          p->set<Teuchos::Array<RealType>>("Time Values", timevals);

          p->set<Teuchos::TwoDArray<RealType>>("BC Values", bcvals);

          p->set<RCP<ParamLib>>("Parameter Library", paramLib);

          p->set<string>("Side Set ID", meshSpecs->ssNames[i]);
          p->set<Teuchos::Array<int>>("Equation Offset", offsets[j]);
          p->set<RCP<Albany::Layouts>>("Layouts Struct", dl);
          p->set<RCP<MeshSpecsStruct>>("Mesh Specs Struct", meshSpecs);
          p->set<int>("Cubature Degree", BCparams.get("Cubature Degree", 0));  // if set to zero, the
                                                                               // cubature degree of the
                                                                               // side will be set to that
                                                                               // of the element

          p->set<string>("Coordinate Vector Name", "Coord Vec");

          if (conditions[k] == "robin") {
            p->set<string>("DOF Name", dof_names[j]);
            p->set<bool>("Vector Field", isVectorField);

            if (isVectorField)
              p->set<RCP<DataLayout>>("DOF Data Layout", dl->node_vector);
            else
              p->set<RCP<DataLayout>>("DOF Data Layout", dl->node_scalar);
          }

          // Pass the input file line
          p->set<string>("Neumann Input String", ss);
          p->set<Teuchos::Array<double>>("Neumann Input Value", Teuchos::tuple<double>(0.0, 0.0, 0.0));
          p->set<string>("Neumann Input Conditions", conditions[k]);

          // If we are doing a Neumann internal boundary with a "scaled jump"
          // (includes "robin" too)
          // The material DB database needs to be passed to the BC object

          if (conditions[k] == "scaled jump" || conditions[k] == "robin") {
            ALBANY_PANIC(materialDB == Teuchos::null, "This BC needs a material database specified");

            p->set<RCP<Albany::MaterialDatabase>>("MaterialDB", materialDB);
          }

          evaluators_to_build[evaluatorsToBuildName(ss)] = p;

          bcs->push_back(ss);
        }
      }
    }
  }

  ///
  /// ACE time dependent BC specific
  ///
  for (std::size_t i = 0; i < meshSpecs->ssNames.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      for (std::size_t k = 0; k < conditions.size(); k++) {
        // construct input.xml string like:
        // "ACE Time Dependent NBC on SS surface_1 for DOF all set P"

        string ss = traits_type::constructACETimeDepBCName(meshSpecs->ssNames[i], bcNames[j], conditions[k]);

        // Have a match of the line in input.xml

        if (BCparams.isSublist(ss)) {
          // grab the sublist
          ParameterList& sub_list = BCparams.sublist(ss);

          //           std::cout << "Constructing ACE Time Dependent NBC: " << ss <<
          //           std::endl;

          // These are read in the LCM::TimeTracBC constructor
          // (LCM/evaluators/TimeTrac_Def.hpp)

          RCP<ParameterList> p = rcp(new ParameterList);

          p->set<int>("Type", traits_type::typeATd);

          std::string const      t_file       = sub_list.get<std::string>("ACE Time Values File");
          std::vector<double>    timevals_vec = LCM::vectorFromFile(t_file);
          Teuchos::Array<double> timevals(timevals_vec);
          Teuchos::Array<double> Lvals;
          Teuchos::Array<double> kvals;
          Teuchos::Array<double> wvals;
          Teuchos::Array<double> hvals;
          Teuchos::Array<double> waterHvals;
          if (sub_list.isParameter("ACE Wave Length Values File")) {
            std::string const          L_file    = sub_list.get<std::string>("ACE Wave Length Values File");
            std::vector<double>        Lvals_vec = LCM::vectorFromFile(L_file);
            Teuchos::ArrayView<double> Lvals_av  = Teuchos::arrayViewFromVector(Lvals_vec);
            Lvals                                = Teuchos::Array<double>(Lvals_av);
          }
          if (sub_list.isParameter("ACE Wave Number Values File")) {
            std::string const          k_file    = sub_list.get<std::string>("ACE Wave Number Values File");
            std::vector<double>        kvals_vec = LCM::vectorFromFile(k_file);
            Teuchos::ArrayView<double> kvals_av  = Teuchos::arrayViewFromVector(kvals_vec);
            kvals                                = Teuchos::Array<double>(kvals_av);
          }
          if (sub_list.isParameter("ACE Wave Height Values File")) {
            std::string const          w_file    = sub_list.get<std::string>("ACE Wave Height Values File");
            std::vector<double>        wvals_vec = LCM::vectorFromFile(w_file);
            Teuchos::ArrayView<double> wvals_av  = Teuchos::arrayViewFromVector(wvals_vec);
            wvals                                = Teuchos::Array<double>(wvals_av);
          }
          if (sub_list.isParameter("ACE Still Water Level Values File")) {
            std::string const          s_file    = sub_list.get<std::string>("ACE Still Water Level Values File");
            std::vector<double>        svals_vec = LCM::vectorFromFile(s_file);
            Teuchos::ArrayView<double> svals_av  = Teuchos::arrayViewFromVector(svals_vec);
            hvals                                = Teuchos::Array<double>(svals_av);
          }

          if (sub_list.isParameter("ACE WaterH Values File")) {
            std::string const          waterH_file    = sub_list.get<std::string>("ACE WaterH Values File");
            std::vector<double>        waterHvals_vec = LCM::vectorFromFile(waterH_file);
            Teuchos::ArrayView<double> waterHvals_av  = Teuchos::arrayViewFromVector(waterHvals_vec);
            waterHvals                                = Teuchos::Array<double>(waterHvals_av);
          }

          p->set<Teuchos::Array<RealType>>("Time Values", timevals);
          p->set<Teuchos::Array<RealType>>("Wave Length Values", Lvals);
          p->set<Teuchos::Array<RealType>>("Wave Number Values", kvals);
          p->set<Teuchos::Array<RealType>>("Still Water Level Values", hvals);
          p->set<Teuchos::Array<RealType>>("Wave Height Values", wvals);
          p->set<Teuchos::Array<RealType>>("WaterH Values", waterHvals);
          p->set<RCP<ParamLib>>("Parameter Library", paramLib);

          p->set<string>("Side Set ID", meshSpecs->ssNames[i]);
          p->set<Teuchos::Array<int>>("Equation Offset", offsets[j]);
          p->set<RCP<Albany::Layouts>>("Layouts Struct", dl);
          p->set<RCP<MeshSpecsStruct>>("Mesh Specs Struct", meshSpecs);
          p->set<int>("Cubature Degree", BCparams.get("Cubature Degree", 0));  // if set to zero, the
                                                                               // cubature degree of the
                                                                               // side will be set to that
                                                                               // of the element

          p->set<string>("Coordinate Vector Name", "Coord Vec");

          // Get additional parameters
          double tm                       = sub_list.get<double>("Impact Duration", 0.04);
          double g                        = sub_list.get<double>("Gravity", 9.806);
          double rho                      = sub_list.get<double>("Water Density", 1022.0);
          double delta                    = sub_list.get<double>("Critical Wave Ratio", 15.0);
          double zmin                     = sub_list.get<double>("Min z-Value", 0.0);
          bool   dump_wave_press_nbc_data = sub_list.get<bool>("Dump Wave Press NBC Data Files", false);

          // Check that parameters are physical
          if (tm <= 0.0) {
            ALBANY_ABORT("Impact Duration parameter must be > 0!");
          }
          if (g <= 0.0) {
            ALBANY_ABORT("Gravity parameter must be > 0!");
          }
          if (rho <= 0.0) {
            ALBANY_ABORT("Water Density parameter must be > 0!");
          }

          // Put parameters into vector to create Teuchos::array
          std::vector<double> param_vec(6);
          param_vec[0] = tm;
          param_vec[1] = g;
          param_vec[2] = rho;
          param_vec[3] = zmin;
          param_vec[4] = delta;
          param_vec[5] = dump_wave_press_nbc_data;

          Teuchos::Array<double> param_array(param_vec);

          // Pass the input file line
          p->set<string>("Neumann Input String", ss);
          // p->set<Teuchos::Array<double>>("Neumann Input Value", Teuchos::tuple<double>(0.0, 0.0, 0.0));
          p->set<Teuchos::Array<double>>("Neumann Input Value", param_array);
          p->set<string>("Neumann Input Conditions", conditions[k]);

          evaluators_to_build[evaluatorsToBuildName(ss)] = p;

          bcs->push_back(ss);
        }
      }
    }
  }

  // Build evaluator for Gather Coordinate Vector
  string NeuGCV = "Evaluator for Gather Coordinate Vector";
  {
    RCP<ParameterList> p = rcp(new ParameterList);
    p->set<int>("Type", traits_type::typeGCV);

    // Input: Periodic BC flag
    p->set<bool>("Periodic BC", false);

    // Output:: Coordindate Vector at vertices
    p->set<RCP<DataLayout>>("Coordinate Data Layout", dl->vertices_vector);
    p->set<string>("Coordinate Vector Name", "Coord Vec");

    evaluators_to_build[NeuGCV] = p;
  }

  // Build evaluator for Gather Solution
  string NeuGS = "Evaluator for Gather Solution";
  {
    RCP<ParameterList> p = rcp(new ParameterList());
    p->set<int>("Type", traits_type::typeGS);

    // for new way
    p->set<RCP<Albany::Layouts>>("Layouts Struct", dl);

    p->set<Teuchos::ArrayRCP<string>>("Solution Names", dof_names);

    p->set<bool>("Vector Field", isVectorField);

    if (isVectorField)
      p->set<RCP<DataLayout>>("Data Layout", dl->node_vector);

    else
      p->set<RCP<DataLayout>>("Data Layout", dl->node_scalar);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<bool>("Disable Transient", true);

    evaluators_to_build[NeuGS] = p;
  }

  // Build evaluator that causes the evaluation of all the NBCs
  string allBC = "Evaluator for all Neumann BCs";
  {
    RCP<ParameterList> p = rcp(new ParameterList);
    p->set<int>("Type", traits_type::typeNa);

    p->set<RCP<std::vector<string>>>("NBC Names", bcs);
    p->set<RCP<DataLayout>>("Data Layout", dl->dummy);
    p->set<string>("NBC Aggregator Name", allBC);
    evaluators_to_build[allBC] = p;
  }
}

template <typename BCTraits>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
Albany::BCUtils<BCTraits>::buildFieldManager(
    const Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator_TemplateManager<PHAL::AlbanyTraits>>>> evaluators,
    std::string&                                                                                      allBC,
    Teuchos::RCP<PHX::DataLayout>&                                                                    dummy)
{
  using PHAL::AlbanyTraits;

  // Create a DirichletFieldManager
  Teuchos::RCP<PHX::FieldManager<AlbanyTraits>> fm = Teuchos::rcp(new PHX::FieldManager<AlbanyTraits>);

  // Register all Evaluators
  PHX::registerEvaluators(evaluators, *fm);

  PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::Residual>(res_tag0);

  PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::Jacobian>(jac_tag0);

  return fm;
}

// Various specializations

Teuchos::RCP<Teuchos::ParameterList const>
Albany::NeumannTraits::getValidBCParameters(
    std::vector<std::string> const& sideSetIDs,
    std::vector<std::string> const& bcNames,
    std::vector<std::string> const& conditions)
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::rcp(new Teuchos::ParameterList("Valid Neumann BC List"));
  ;

  for (std::size_t i = 0; i < sideSetIDs.size(); i++) {      // loop over all side sets in the mesh
    for (std::size_t j = 0; j < bcNames.size(); j++) {       // loop over all possible types of condition
      for (std::size_t k = 0; k < conditions.size(); k++) {  // loop over all possible types of condition

        std::string ss  = Albany::NeumannTraits::constructBCName(sideSetIDs[i], bcNames[j], conditions[k]);
        std::string tt  = Albany::NeumannTraits::constructTimeDepBCName(sideSetIDs[i], bcNames[j], conditions[k]);
        std::string att = Albany::NeumannTraits::constructACETimeDepBCName(sideSetIDs[i], bcNames[j], conditions[k]);

        Teuchos::Array<double> defaultData;
        validPL->set<Teuchos::Array<double>>(ss, defaultData, "Value of BC corresponding to sideSetID and boundary condition");

        validPL->sublist(tt, false, "SubList of BC corresponding to sideSetID and boundary condition");
        validPL->sublist(att, false, "SubList of BC corresponding to sideSetID and boundary condition");
      }
    }
  }

  validPL->set<int>("Cubature Degree", 3, "Cubature Degree for Neumann BC");
  return validPL;
}

std::string
Albany::NeumannTraits::constructBCName(std::string const& ns, std::string const& dof, std::string const& condition)
{
  std::stringstream ss;
  ss << "NBC on SS " << ns << " for DOF " << dof << " set " << condition;
  return ss.str();
}

std::string
Albany::NeumannTraits::constructTimeDepBCName(std::string const& ns, std::string const& dof, std::string const& condition)
{
  std::stringstream ss;
  ss << "Time Dependent " << Albany::NeumannTraits::constructBCName(ns, dof, condition);
  return ss.str();
}

std::string
Albany::NeumannTraits::constructACETimeDepBCName(std::string const& ns, std::string const& dof, std::string const& condition)
{
  std::stringstream ss;
  ss << "ACE Time Dependent " << Albany::NeumannTraits::constructBCName(ns, dof, condition);
  return ss.str();
}
