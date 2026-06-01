#!/bin/bash
#
# Nightly clone, build and test script for LCM.
# Clones Trilinos and LCM from scratch, then builds and tests.
# DTK is absorbed into LCM (no separate clone needed).
#
set -e

export LCM_DIR=$(pwd)
export MODULEPATH=$LCM_DIR/LCM/doc/LCM/modulefiles:$MODULEPATH

echo "=== LCM Nightly Build ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "LCM_DIR: $LCM_DIR"
echo

# Clone repositories
echo "--- Cloning repositories ---"

for REPO_INFO in \
    "Trilinos|git@github.com:trilinos/Trilinos.git|develop" \
    "LCM|git@github.com:sandialabs/LCM.git|main"
do
    IFS='|' read -r NAME URL BRANCH <<< "$REPO_INFO"
    echo "Cloning $NAME ($BRANCH)..."
    rm -rf "$NAME"
    git clone -q -b "$BRANCH" "$URL" "$NAME" 2>&1 | tail -1
done

echo

# Create lcm symlink
ln -sf LCM/doc/LCM/build/lcm .

# Build and test with each compiler.
# Use 'nproc --all' to report installed CPUs regardless of OMP_NUM_THREADS,
# which is set to 1 in the user environment for Sierra but should not cap
# the LCM build parallelism.
NUM_PROCS=$(nproc --all)

for MODULE in serial-gcc-release serial-clang-release; do
    echo "=== Building with module: $MODULE ==="
    ./lcm all "$NUM_PROCS" --module="$MODULE" --cdash || {
        echo "FAILED: $MODULE"
        continue
    }
    echo "PASSED: $MODULE"
    echo
done

echo "=== Nightly build complete: $(date) ==="
