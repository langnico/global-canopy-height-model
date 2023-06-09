#!/bin/bash

# parse and create directory from first argument
trained_models_dir=${1}
mkdir -p ${trained_models_dir}
cd ${trained_models_dir}

url="https://zenodo.org/record/7885610/files/gchm_deploy_example.zip?download=1"
# download zip file
curl $url --output "gchm_deploy_example.zip"
# unzip
unzip gchm_deploy_example.zip
# delete zip file
rm gchm_deploy_example.zip

echo "DONE. Trained models extracted in:"
pwd

