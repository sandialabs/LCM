#!/bin/bash

source ./env-all.sh

cd "$LCM_DIR"

# Clone package repositories.
for PACKAGE in $PACKAGES; do
    case "$PACKAGE" in
	trilinos)
	    PACKAGE_NAME="Trilinos"
	    REPO="git@github.com:trilinos/Trilinos.git"
            BRANCH="develop"
            SHA="84124e66709dc3679e6dbef1f8352c9909ce7900"
	    ;;
	lcm)
	    PACKAGE_NAME="LCM"
	    REPO="git@github.com:sandialabs/LCM.git"
            BRANCH="main"
            SHA="head"
	    ;;
	dtk)
	    PACKAGE_NAME="DataTransferKit"
	    REPO="git@github.com:ikalash/DataTransferKit"
            BRANCH="dtk-2.0-tpetra-static-graph"
            SHA="head"
	    ;;
	*)
	    echo "Unrecognized package option"
	    exit 1
	    ;;
    esac
    PACKAGE_DIR="$LCM_DIR/$PACKAGE_NAME"
    CHECKOUT_LOG="$PACKAGE-checkout.log"
    if [ -d "$PACKAGE_DIR" ]; then
	rm "$PACKAGE_DIR" -rf
    fi
    git clone -v -b "$BRANCH" "$REPO" "$PACKAGE_NAME" &> "$CHECKOUT_LOG"
    if [[ "$SHA" != *"head"* ]]; then
        git reset --hard "$SHA"
    fi
done

# For now assume that if there is a DTK directory in the main LCM
# directory, it contains a DTK version that we can use for
# Trilinos.
if [ -e DataTransferKit ]; then
    cp -p -r DataTransferKit Trilinos
fi

./clean-config-build-test-dash-all.sh

cd "$LCM_DIR"
