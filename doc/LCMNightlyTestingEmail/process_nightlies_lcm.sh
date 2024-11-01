#!/bin/bash

rm -rf TriBITS
rm -rf test_history
rm -rf *html
rm -rf *out
rm -rf *json
rm -rf  lcmNightlyBuildsTwoif.csv 

git clone git@github.com:TriBITSPub/TriBITS.git


now=$(date +"%Y-%m-%d")

./lcm_cdash_status.sh --date=$now --email-from-address=ikalash@cee-compute003.sandia.gov --send-email-to=ikalash@sandia.gov,amota@sandia.gov,jmfrede@sandia.gov


