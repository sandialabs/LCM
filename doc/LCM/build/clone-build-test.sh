#!/bin/bash
# Clone, build, and test LCM with all supported compilers.
# Usage: clone-build-test.sh [threads]
#   threads: number of build threads (default: nproc)

set -e

THREADS="${1:-$(nproc)}"
export LCM_DIR="${LCM_DIR:-$(pwd)}"
LCM_SCRIPT="$LCM_DIR/LCM/doc/LCM/build/lcm"

echo "=== LCM Clone-Build-Test ==="
echo "LCM_DIR:  $LCM_DIR"
echo "Threads:  $THREADS"

#---------------------------------------------------------------------------
# Clone repositories
#---------------------------------------------------------------------------
cd "$LCM_DIR"

TRILINOS_URL="https://github.com/trilinos/Trilinos.git"
LCM_URL="https://github.com/sandialabs/LCM.git"
DTK_URL="https://github.com/ikalash/DataTransferKit.git"

if [ ! -d "$LCM_DIR/Trilinos" ]; then
    echo "--- Cloning Trilinos (develop) ---"
    git clone -b develop "$TRILINOS_URL" Trilinos
else
    echo "--- Trilinos already exists, pulling latest ---"
    cd "$LCM_DIR/Trilinos" && git pull && cd "$LCM_DIR"
fi

if [ ! -d "$LCM_DIR/LCM" ]; then
    echo "--- Cloning LCM (main) ---"
    git clone -b main "$LCM_URL" LCM
else
    echo "--- LCM already exists, pulling latest ---"
    cd "$LCM_DIR/LCM" && git pull && cd "$LCM_DIR"
fi

if [ ! -d "$LCM_DIR/DataTransferKit" ]; then
    echo "--- Cloning DTK (dtk-2.0-tpetra-static-graph) ---"
    git clone -b dtk-2.0-tpetra-static-graph "$DTK_URL" DataTransferKit
else
    echo "--- DTK already exists, pulling latest ---"
    cd "$LCM_DIR/DataTransferKit" && git pull && cd "$LCM_DIR"
fi

#---------------------------------------------------------------------------
# Copy DTK into Trilinos
#---------------------------------------------------------------------------
echo "--- Copying DTK into Trilinos/DataTransferKit ---"
if [ -d "$LCM_DIR/Trilinos/DataTransferKit" ]; then
    rm -rf "$LCM_DIR/Trilinos/DataTransferKit"
fi
cp -a "$LCM_DIR/DataTransferKit" "$LCM_DIR/Trilinos/DataTransferKit"

#---------------------------------------------------------------------------
# Build and test with each compiler
#---------------------------------------------------------------------------
COMPILERS="gcc clang"

for COMPILER in $COMPILERS; do
    MODULE="serial-${COMPILER}-release"
    echo ""
    echo "========================================"
    echo "=== Building with $COMPILER ($MODULE) ==="
    echo "========================================"

    LCM_MODULE="$MODULE" "$LCM_SCRIPT" all "$THREADS"

    echo "=== $COMPILER build complete ==="
done

echo ""
echo "=== All builds and tests complete ==="
