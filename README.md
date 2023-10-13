# A high-resolution canopy height model of the Earth

This repository contains the code used to create the results presented in the paper: [A high-resolution canopy height model of the Earth](https://arxiv.org/abs/2204.08322).
Here, we developed a model to estimate canopy top height anywhere on Earth. The model estimates canopy top height for every Sentinel-2 image pixel and was trained using sparse GEDI LIDAR data as a reference.

See our [project page](https://langnico.github.io/globalcanopyheight) for an interactive [demo](https://nlang.users.earthengine.app/view/global-canopy-height-2020) and more information.

## Table of Contents
1. [Data availability](https://github.com/langnico/global-canopy-height-model#data-availability)
2. [Installation and credentials](https://github.com/langnico/global-canopy-height-model#installation-and-credentials)
3. [Loading the model](https://github.com/langnico/global-canopy-height-model#loading-the-model)
4. [Deploying](https://github.com/langnico/global-canopy-height-model#deploying)
5. [Training](https://github.com/langnico/global-canopy-height-model#training)
6. [ALS preprocessing for independent comparison](https://github.com/langnico/global-canopy-height-model#als-preprocessing-for-independent-comparison)
7. [Citation](https://github.com/langnico/global-canopy-height-model#citation)

## Data availability
This is a summary of all the published data:

- Global canopy top height map for 2020 ([ETH Research Collection](https://doi.org/10.3929/ethz-b-000609802))
- Train-val dataset ([ETH Research Collection](https://doi.org/10.3929/ethz-b-000609845))
- Rasterized canopy top height models from airborne lidar ([Zenodo](https://doi.org/10.5281/zenodo.7885699))
- Trained model weights ([Github release](https://github.com/langnico/global-canopy-height-model/releases/download/v1.0-trained-model-weights/trained_models_GLOBAL_GEDI_2019_2020.zip))
- Demo data for example scripts ([Zenodo](https://doi.org/10.5281/zenodo.7885610))
- Sparse GEDI canopy top height data ([Zenodo](https://doi.org/10.5281/zenodo.7737946))
- ESA WorldCover 10 m 2020 v100 reprojected to Sentinel-2 tiles ([Zenodo](https://doi.org/10.5281/zenodo.7888150))

## Installation and credentials
Please follow the instructions in [INSTALL.md](INSTALL.md).

## Loading the model 

```python
from gchm.models.xception_sentinel2 import xceptionS2_08blocks_256
# load the model with random initialization
model = xceptionS2_08blocks_256()
```
Please see the [example notebook](gchm/notebooks/example_loading_pretrained_models.ipynb) on how to load the model with the trained weights. 

## Deploying

This is a demo how to run the trained ensemble to compute the canopy height map for a Sentinel-2 tile (approx. 100 km x 100 km).

### Preparation:
1. Download the demo data which contains Sentinel-2 images for one tile: 
    ```
    bash gchm/bash/download_demo_data.sh ./
    ```
   This creates the following directory:
    ```
    deploy_example/
    ├── ESAworldcover
    │   └── 2020
    │       └── sentinel2_tiles
    │           └── ESA_WorldCover_10m_2020_v100_32TMT.tif
    ├── image_paths
    │   └── 2020
    │       └── 32TMT.txt
    ├── image_paths_logs
    │   └── 2020
    ├── predictions_provided
    │   ├── 2020
    │   │   ├── S2A_MSIL2A_20200623T103031_N0214_R108_T32TMT_20200623T142851_predictions.tif
    │   │   ├── S2A_MSIL2A_20200623T103031_N0214_R108_T32TMT_20200623T142851_std.tif
    │   │   ├── ...
    │   ├── 2020_merge
    │   │   └── preds_inv_var_mean
    │   │       ├── 32TMT_pred.tif
    │   │       └── 32TMT_std.tif
    │   └── 2020_merge_logs
    │       └── preds_inv_var_mean
    │           └── 32TMT.txt
    ├── sentinel2
    │   └── 2020
    │       ├── S2A_MSIL2A_20200623T103031_N0214_R108_T32TMT_20200623T142851.zip
    │       ├── ...
    └── sentinel2_aws
        └── 2020
    ```
2. Download the trained model weights:
    ```
    bash gchm/bash/download_trained_models.sh ./trained_models
    ```
   
    This creates the following directory:
    
    ```
    trained_models/
    └── GLOBAL_GEDI_2019_2020
        ├── model_0
        │   ├── FT_Lm_SRCB
        │   │   ├── args.json
        │   │   ├── checkpoint.pt
        │   │   ├── train_input_mean.npy
        │   │   ├── train_input_std.npy
        │   │   ├── train_target_mean.npy
        │   │   └── train_target_std.npy
        │   ├── args.json
        │   ├── checkpoint.pt
        │   ├── train_input_mean.npy
        │   ├── train_input_std.npy
        │   ├── train_target_mean.npy
        │   └── train_target_std.npy
        ├── model_1
        │   ├── ...
        ├── model_2
        │   ├── ...
        ├── model_3
        │   ├── ...
        ├── model_4
        │   ├── ...
    ```
   The checkpoint.pt files contain the model weights. The subdirectories `FT_Lm_SRCB` contain the models finetuned with a re-weighted loss function.
            
### Deploy example for a single Sentinel-2 image
This [demo script](gchm/bash/deploy_example.sh) processes a single image (from the year 2020) for the tile "32TMT" in Switzerland. Run: 
```
bash gchm/bash/deploy_example.sh
```

### Deploy and merge example for multiple images of a Sentinel-2 tile
This [demo script](gchm/bash/run_tile_deploy_merge.sh) processes 10 images (from the year 2020) for the tile "32TMT" in Switzerland and aggregates the individual per-image maps to a final annual map.

Provide a text file with the image filenames per tile saved as `${TILE_NAME}.txt`. The demo data contains the following file: 
```
cat ./deploy_example/image_paths/2020/32TMT.txt 
S2A_MSIL2A_20200623T103031_N0214_R108_T32TMT_20200623T142851.zip
S2A_MSIL2A_20200723T103031_N0214_R108_T32TMT_20200723T142801.zip
S2A_MSIL2A_20200812T103031_N0214_R108_T32TMT_20200812T131334.zip
...
```
The corresponding images are stored in `./deploy_example/sentinel2/2020/`.


1. Set the paths in `gchm/bash/config.sh`
2. Set the tile_name in `gchm/bash/run_tile_deploy_merge.sh`
3. Run the script:
    ```
    bash gchm/bash/run_tile_deploy_merge.sh
    ```

#### Note on ESA World Cover post-processing: 
The ESA WorldCover 10 m 2020 v100 reprojected to Sentinel-2 tiles is available on [Zenodo](https://doi.org/10.5281/zenodo.7888150). 
We apply minimal post-processing and mask out built-up areas, snow,
 ice and permanent water bodies, setting their canopy height to ”no data” (value: 255). See the script [here](gchm/postprocess/mask_with_ESAworldcover.py).

#### Note on AWS: 
Sentinel-2 images can be downloaded on the fly from AWS S3 by setting `GCHM_DOWNLOAD_FROM_AWS="True"` 
and providing the AWS credentials as described above. 
This was tested for 2020 data, but might need some update in the sentinelhub routine to handle newer versions.


## Training

### Data preparation
1. Download the train-val h5 datasets from [here](https://doi.org/10.3929/ethz-b-000609845).
2. Merge the parts file to a single `train.h5` and `val.h5` by running this [script](gchm/preprocess/run_merge_h5_files_per_split.sh). 
   Before running it, set the variables `in_h5_dir_parts` and `out_h5_dir` to your paths. Then run:
    ```
    bash gchm/preprocess/run_merge_h5_files_per_split.sh`
    ```

### Running the training script
A [slurm training script](gchm/bash/run_training.sh) is provided and submitted as follows.
Before submitting, set the variable `CODE_PATH` at the top of the script and set the paths in `gchm/bash/config.sh`. Then run:
```
sbatch < gchm/bash/run_training.sh
```

## ALS preprocessing for independent comparison

In cases where rastered high-resolution canopy height models are available (e.g. from airborne LIDAR campaigns) for independent evaluation, some preprocessing steps are required to 
make the data comparable to GEDI canopy top height estimates corresponding to the canopy top within a 25 meter footprint.

1. A rastered canopy height model with a 1m GSD should be created (E.g. using `gdalwarp`).
2. The 1m canopy height model can then be processed with a circular max pooling operation to approximate "GEDI-like" canopy top heights. This step is provided as a [pytorch implementation](gchm/preprocess/ALS_maxpool_GEDI_footprint.py).

**Example**:
Download the example CHM at 1m GSD from [here](https://zenodo.org/record/7885610/files/ALS_example_CTHM_GSD1m.tif). Then run: 
```
python3 gchm/preprocess/ALS_maxpool_GEDI_footprint.py "path/to/input/tif" "path/to/output/tif"
```

## Citation

Please cite our paper if you use this code or any of the provided data.

Lang, N., Jetz, W., Schindler, K., & Wegner, J. D. (2023). A high-resolution canopy height model of the Earth. Nature Ecology & Evolution, 1-12.
```
@article{lang2023high,
  title={A high-resolution canopy height model of the Earth},
  author={Lang, Nico and Jetz, Walter and Schindler, Konrad and Wegner, Jan Dirk},
  journal={Nature Ecology \& Evolution},
  pages={1--12},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

