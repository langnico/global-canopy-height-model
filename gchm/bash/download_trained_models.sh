#!/bin/bash

# parse and create directory from first argument
trained_models_dir=${1}
mkdir -p ${trained_models_dir}
cd ${trained_models_dir}

url_trained_models="https://github.com/langnico/global-canopy-height-model/releases/download/v1.0-trained-model-weights/trained_models_GLOBAL_GEDI_2019_2020.zip"
# download zip file
curl $url_trained_models --output "trained_models_GLOBAL_GEDI_2019_2020.zip"
# unzip
unzip trained_models_GLOBAL_GEDI_2019_2020.zip
# delete zip file
rm trained_models_GLOBAL_GEDI_2019_2020.zip

echo "DONE. Trained models extracted in:"
pwd

