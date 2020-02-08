#!/bin/bash

if [ -f ./CMakeCache.txt ]; then
    rm CMakeCache.txt
fi

# The Trilinos Dir is the same as the PREFIX entry from the
# Trilinos configuration script

cmake \
 -D ALBANY_TRILINOS_DIR:FILEPATH=$REMOTE/trilinos-install-gcc-release \
 -D CMAKE_CXX_FLAGS:STRING="-msse3 -DNDEBUG" \
 -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
 -D ENABLE_LCM:BOOL=ON \
 -D ENABLE_CHECK_FPE:BOOL=OFF \
 -D ENABLE_FLUSH_DENORMALS:BOOL=OFF \
 -D ALBANY_ENABLE_FORTRAN:BOOL=OFF \
 -D ENABLE_SLFAD:BOOL=OFF \
\
 -D ENABLE_64BIT_INT:BOOL=OFF \
 -D ENABLE_INSTALL:BOOL=OFF \
 -D CMAKE_INSTALL_PREFIX:PATH=$REMOTE/albany-build-gcc-release \
 -D ENABLE_DEMO_PDES:BOOL=OFF \
  \
 $REMOTE/src/Albany
