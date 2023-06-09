#!/bin/bash
#SBATCH --job-name=gchm_train
#SBATCH --gpus=rtx_2080_ti:1    # rtx_2080_ti:1  rtx_3090:1
#SBATCH --array=0 		          # --array=0-5%10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20G  # this is muliplied by cpus-per-task AND ntasks
#SBATCH --time=0-48:00:00  # might take up to 350 hours on rtx_2080_ti
#SBATCH --output=/cluster/work/igp_psr/nlang/experiments/gchm/slurm_logs/slurm-%A_%a_%x.out
# Temporary disk space on the node
# #SBATCH --tmp=2400G
# Send email
#SBATCH --mail-type=ARRAY_TASKS  # send email for each individual array task, otherwise only for the job array as a whole.
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails

# NOTE train: --cpus-per-task=8 --mem-per-cpu=20G (total max RAM 160GB)
# NOTE test:  --cpus-per-task=18 --mem-per-cpu=21G (total max RAM 378 GB)

# Set path to repository
CODE_PATH="${HOME}/code/global-canopy-height-model-private"

cd ${CODE_PATH}
source gchm/bash/config.sh

# set wandb api key
source ${HOME}/.config_wandb 

num_workers=8

# ----- CONFIGURATION -----
# Set fine-tune strategy after training from scratch
finetune_strategy="NONE"          # Train from scratch
# finetune_strategy="FT_Lm_SRCB"  # Fine-tune model with square root inverse class frequency re-weighting (class-balanced)

do_train=true  # false: run validation only

# architecture
architecture="xceptionS2_08blocks_256"
input_lat_lon=True
long_skip=True

# dataset
merged_h5_files=true
if [ "${merged_h5_files}" = "true" ]; then
    # shuffled samples
    h5_dir=${GCHM_TRAINING_DATA_DIR}
    data_stats_dir=${GCHM_TRAINING_DATA_DIR}/stats
else
    h5_dir=${GCHM_TRAINING_DATA_DIR}/GLOBAL_GEDI_2019_2020/tiles
    data_stats_dir=None
fi

# input patch dimensions
channels=12 
patch_size=15

# loss
return_variance=True
loss_key='GNLL'

# optimizer
optimizer='ADAM'
base_learning_rate=1e-4
scheduler='MultiStepLR'
lr_milestones=( 400 700 )
max_grad_value=1e3
l2_lambda=0

batch_size=64
nb_epoch=1000  
iterations_per_epoch=5000  # if None: one epoch corresponds to the full dataset len(dl_train)

# Note that SliceBatchSampler expects the samples to be shuffled already in the h5 file.
custom_sampler=SliceBatchSampler  # None, SliceBatchSampler, BatchSampler

if [ "${merged_h5_files}" = "true" ]; then
    train_tiles=None
    val_tiles=None
else
    # read tile name split from txt file with all tile names
    tile_names_dir=${GCHM_TRAINING_DATA_DIR}/sentinel2_tilenames/${dataset}
    readarray -t train_tiles < ${tile_names_dir}/tile_names_${dataset}_train.txt
    readarray -t val_tiles < ${tile_names_dir}/tile_names_${dataset}_val.txt
    train_tiles=${train_tiles[@]}
    val_tiles=${val_tiles[@]}
fi

# NOTE: in deploy the model index is expected to start at 0
n_models=5
model_idx=$((SLURM_ARRAY_TASK_ID))
echo "model_idx: ${model_idx}"
echo 'h5_dir: ' ${h5_dir}

out_dir=${GCHM_TRAINING_EXPERIMENT_DIR}/model_${model_idx}/

# overwrite arguments for fine-tuning
if [[ "${finetune_strategy}" == "NONE" ]]; then
    echo "Training from scratch"
    base_model_dir="NONE"
    model_weights_path=None
else
    echo "Fine-tuning using strategy: ${finetune_strategy}"
    nb_epoch=1150  # this will fine-tune for additional 150 epochs (after 1000 epochs were trained from scratch)
    base_model_dir=${out_dir}
    out_dir=${base_model_dir}/${finetune_strategy}
fi

echo 'out_dir:' ${out_dir}

${PYTHON} gchm/train_val.py --out_dir=${out_dir} \
                          --num_workers=${num_workers} \
                          --architecture=${architecture} \
                          --do_train=${do_train} \
                          --h5_dir=${h5_dir} \
                          --num_samples_statistics="1e6" \
                          --input_lat_lon=${input_lat_lon} \
                          --channels=${channels} \
                          --patch_size=${patch_size} \
                          --long_skip=${long_skip} \
                          --return_variance=${return_variance} \
                          --loss_key=${loss_key} \
                          --optimizer=${optimizer} \
                          --base_learning_rate=${base_learning_rate} \
                          --batch_size=${batch_size} \
                          --model_weights_path=${model_weights_path} \
                          --nb_epoch=${nb_epoch} \
                          --iterations_per_epoch=${iterations_per_epoch} \
                          --normalize_targets=True \
                          --train_tiles ${train_tiles} \
                          --val_tiles ${val_tiles} \
                          --data_stats_dir=${data_stats_dir} \
                          --weight_key=${weight_key} \
                          --merged_h5_files=${merged_h5_files} \
                          --region_name="GLOBAL_GEDI" \
                          --do_profile=False \
                          --max_grad_value=${max_grad_value} \
                          --debug=False \
                          --custom_sampler=${custom_sampler} \
                          --scheduler=${scheduler} \
                          --lr_milestones ${lr_milestones[@]} \
                          --l2_lambda=${l2_lambda} \
                          --finetune_strategy=${finetune_strategy} \
                          --base_model_dir=${base_model_dir}

