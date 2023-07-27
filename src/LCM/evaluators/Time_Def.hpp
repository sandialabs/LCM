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
Time<EvalT, Traits>::Time(Teuchos::ParameterList& p)
    : time(p.get<std::string>("Time Name"), p.get<Teuchos::RCP<PHX::DataLayout>>("Workset Scalar Data Layout")),
      deltaTime(p.get<std::string>("Delta Time Name"), p.get<Teuchos::RCP<PHX::DataLayout>>("Workset Scalar Data Layout")),
      timeValue(0.0)
{
  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else
    enableTransient = true;

  // Add Time as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);
  this->registerSacadoParameter("Time", paramLib);

  this->addEvaluatedField(time);
  this->addEvaluatedField(deltaTime);

  timeName = p.get<std::string>("Time Name") + "_old";
  this->setName("Time" + PHX::print<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
Time<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(time, fm);
  this->utils.setFieldData(deltaTime, fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
Time<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  time(0) = workset.current_time;

  std::cout << "IKT timeName = " << timeName << "\n"; 
  Albany::MDArray timeOld = (*workset.stateArrayPtr)[timeName];
  deltaTime(0)            = time(0) - timeOld(0);
  std::cout << "IKT setting Delta Time!  time, timeOld, deltaTime = " << time(0) << ", " << 
	    timeOld(0) << ", " << deltaTime(0) << "!\n";  
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename Time<EvalT, Traits>::ScalarT&
Time<EvalT, Traits>::getValue(std::string const& n)
{
  return timeValue;
}

// **********************************************************************
// **********************************************************************
}  // namespace LCM
