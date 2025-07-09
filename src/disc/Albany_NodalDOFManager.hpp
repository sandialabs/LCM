// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#ifndef ALBANY_NODAL_DOF_MANAGER_HPP
#define ALBANY_NODAL_DOF_MANAGER_HPP

#include "Albany_ScalarOrdinalTypes.hpp"

namespace Albany {

class NodalDOFManager
{
 public:
  NodalDOFManager() : _numComponents(0), _numLocalDOF(0), _numGlobalDOF(0), _interleaved(true) {};

  void
  setup(int numComponents, LO numLocalDOF, GO numGlobalDOF, bool interleaved = true)
  {
    _numComponents = numComponents;
    _numLocalDOF   = numLocalDOF;
    _numGlobalDOF  = numGlobalDOF;
    _interleaved   = interleaved;
  }

  inline LO
  getLocalDOF(LO inode, int icomp) const
  {
    return (_interleaved) ? inode * _numComponents + icomp : inode + _numLocalDOF * icomp;
  }
  inline GO
  getGlobalDOF(GO node, int icomp) const
  {
    return (_interleaved) ? node * _numComponents + icomp : node + _numGlobalDOF * icomp;
  }
  int
  numComponents() const
  {
    return _numComponents;
  }

 private:
  int  _numComponents;
  LO   _numLocalDOF;
  GO   _numGlobalDOF;
  bool _interleaved;
};

}  // namespace Albany

#endif  // ALBANY_NODAL_DOF_MANAGER_HPP
