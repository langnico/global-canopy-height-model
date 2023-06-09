#!/bin/bash
# This script runs one job per sentinel-2 tile name. It runs the prediction on a list of images and merges the individual predictions.

# ---- SET tile_name ----
tile_name="32TMT"
# -----------------------

export BASH_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source ${BASH_PATH}/config.sh

sentinel2_dir=${GCHM_DEPLOY_SENTINEL2_DIR}
# get image paths text file for current tile_name
image_paths_file=${GCHM_DEPLOY_IMAGE_PATHS_DIR}/${tile_name}.txt

if [ ${GCHM_DOWNLOAD_FROM_AWS} == "True" ] ; then
    echo "setting aws credentials!"
    source ${GCHM_AWS_CONFIGS_FILE}
    # set to the empty directory to save sentinel2 data downloaded from aws
    sentinel2_dir=${GCHM_DEPLOY_SENTINEL2_AWS_DIR}
    mkdir -p ${sentinel2_dir}
fi

filepath_failed_image_paths=${GCHM_DEPLOY_IMAGE_PATHS_LOG_DIR}/${tile_name}_failed.txt
touch ${filepath_failed_image_paths}

echo 'tile_name:                   ' ${tile_name}
echo 'GCHM_DEPLOY_DIR:             ' ${GCHM_DEPLOY_DIR}
echo 'GCHM_MODEL_DIR:              ' ${GCHM_MODEL_DIR}
echo 'GCHM_NUM_MODELS:             ' ${GCHM_NUM_MODELS}
echo 'finetune_strategy:           ' ${finetune_strategy}
echo 'filepath_failed_image_paths: ' ${filepath_failed_image_paths}
echo 'sentinel2_dir:               ' ${sentinel2_dir}

# read image paths to process for this tile
readarray -t tile_image_filenames < ${image_paths_file}
num_tile_image_filenames=${#tile_image_filenames[@]}
echo "Number of tile_image_filenames: " ${num_tile_image_filenames}
tile_image_filenames=${tile_image_filenames[@]}

# Loop through all images paths and deploy model
count=$((count + 1))
for tile_image_filename in ${tile_image_filenames}; do
    echo "*************************************"
    if [ ${GCHM_DOWNLOAD_FROM_AWS} == "True" ] ; then
    	deploy_image_path=${tile_image_filename}
    else
	# set path to image on disk
	deploy_image_path=${sentinel2_dir}/${tile_image_filename}
    fi
    echo 'deploy_image_path:' ${deploy_image_path}
    echo "tile image: ${count} / ${num_tile_image_filenames}"
    count=$((count + 1))
    
    python3 gchm/deploy.py --model_dir=${GCHM_MODEL_DIR} \
                      	   --deploy_image_path=${deploy_image_path} \
                      	   --deploy_dir=${GCHM_DEPLOY_DIR} \
                      	   --deploy_patch_size=512 \
                      	   --num_workers_deploy=4 \
                      	   --num_models=${GCHM_NUM_MODELS} \
                      	   --finetune_strategy="FT_Lm_SRCB" \
                      	   --filepath_failed_image_paths=${filepath_failed_image_paths} \
                      	   --download_from_aws=${GCHM_DOWNLOAD_FROM_AWS} \
                      	   --sentinel2_dir=${sentinel2_dir} \
                      	   --remove_image_after_pred="False"

    # check if proxy error
    exit_status=$?  # store the exit status for later use
    if [ $exit_status -eq 222 ]; then
        echo "Proxy error, abort!"
        exit $exit_status  # exit the bash script with the same status
    fi
done


#############################
# merge per image predictions
reduction="inv_var_mean"
out_dir=${GCHM_DEPLOY_DIR}_merge/${year}/preds_${reduction}/
mkdir -p ${out_dir}

echo "*************************************"
echo "merging... $reduction"
echo "reduction: " ${reduction}
echo "out_dir: " ${out_dir}

python3 gchm/merge_predictions_tile.py --tile_name=${tile_name} \
                                       --out_dir=${out_dir} \
                                       --deploy_dir=${GCHM_DEPLOY_DIR} \
                                       --out_type="uint8" \
                                       --nodata_value=255 \
                                       --from_aws=False \
                                       --reduction=${reduction} \
                                       --finetune_strategy=${finetune_strategy}


#############################
# filter with ESA WorldCover land cover classification
echo "filtering with ESA WorldCover..."
worldcover_path="${GCHM_DEPLOY_PARENT_DIR}/ESAworldcover/2020/sentinel2_tiles/ESA_WorldCover_10m_2020_v100_${tile_name}.tif"

# mask height prediction
input_file_path="${out_dir}/${tile_name}_pred.tif"
output_file_path=${input_file_path}
python3 gchm/postprocess/mask_with_ESAworldcover.py ${worldcover_path} ${input_file_path} ${output_file_path}

# mask standard deviation
input_file_path="${out_dir}/${tile_name}_std.tif"
output_file_path=${input_file_path}
python3 gchm/postprocess/mask_with_ESAworldcover.py ${worldcover_path} ${input_file_path} ${output_file_path}

echo "final maps saved at:         ${out_dir}"
echo "failed image paths saved at: ${filepath_failed_image_paths}"
