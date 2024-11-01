#!/bin/bash

source ./env-all.sh

cd "$LCM_DIR"
SCRIPT_NAME=`basename $0`

case "$SCRIPT_NAME" in
    build-all.sh)
	;&
    config-all.sh)
	;&
    clean-all.sh)
	;&
    test-all.sh)
	;&
    dash-all.sh)
	;&
    clean-config-all.sh)
	;&
    clean-config-build-all.sh)
	;&
    clean-config-build-test-all.sh)
	;&
    clean-config-build-test-dash-all.sh)
	;&
    config-build-all.sh)
	;&
    config-build-test-all.sh)
	;&
    config-build-test-dash-all.sh)
	;&
    build-test-all.sh)
	;&
    build-test-dash-all.sh)
	;&
    test-dash-all.sh)
	COMMAND="$LCM_DIR/${SCRIPT_NAME%-*}.sh"
	;;
    *)
	echo "Unrecognized script name in build-all: $SCRIPT_NAME"
	exit 1
	;;
esac

KERNEL_VERSION=`uname -r`
PLATFORM="unknown"
if [[ ${KERNEL_VERSION} == *"fc"* ]]; then
    PLATFORM="fedora"
elif [[ ${KERNEL_VERSION} == *"el"* ]]; then
    PLATFORM="rhel"
elif [[ ${KERNEL_VERSION} == *"chaos"* ]]; then
    PLATFORM="cluster"
elif [[ ${KERNEL_VERSION} == *"generic"* ]]; then
    PLATFORM="ubuntu"
else
    echo "Unrecongnized platform. Valid platforms: fc, el, chaos"
    uname -r
    exit 1
fi

# Use different variable names for loop counters so they do not
# conflict with the variables defined by the module command.
for P in $PACKAGES; do
    for A in $ARCHES; do
        for TC in $TOOL_CHAINS; do
            for BT in $BUILD_TYPES; do
                MODULE="$A"-"$TC"-"$BT"
                echo "MODULE: $MODULE"
                echo "PLATFORM: $PLATFORM"
                module purge
                module load "$MODULE"
                "$COMMAND" "$P" "$NUM_PROCS"
            done
        done
    done
done

cd "$LCM_DIR"
