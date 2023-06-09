from osgeo import gdal, osr, ogr, gdalconst
import os
import sys
import numpy as np
import argparse
from pathlib import Path
import copy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import botocore
import urllib3

from gchm.models.architectures import Architectures
from gchm.utils.transforms import Normalize, NormalizeVariance, denormalize
from gchm.datasets.dataset_sentinel2_deploy import Sentinel2Deploy
from gchm.utils.gdal_process import save_array_as_geotif
from gchm.utils.parser import load_args_from_json, str2bool, str_or_none
from gchm.utils.aws import download_and_zip_safe_from_aws


DEVICE = torch.device("cuda:0")
print('DEVICE: ', DEVICE, torch.cuda.get_device_name(0))
gdal.UseExceptions()


def setup_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default='./tmp/', help="ensemble directory containing subdirectories (model_i) with checkpoint (model weights) and normalization statistics. ")
    parser.add_argument("--deploy_image_path", help="path to Sentinel-2 image .zip file with SAFE format")
    parser.add_argument("--deploy_dir", help="output directory to save predictions")
    parser.add_argument("--filepath_failed_image_paths", help="path to .txt file to write failed deploy_image_path")
    parser.add_argument("--deploy_patch_size", default=512, help="Size of square patch (height=width)", type=int)
    parser.add_argument("--deploy_batch_size", default=2, help="Batch size: Number of patches per batch during prediction (deploy).", type=int)
    parser.add_argument("--num_workers_deploy", default=0, help="number of workers in dataloader", type=int)
    parser.add_argument("--num_models", default=1, help="number of models in ensemble (model_dir/model_0)", type=int)
    parser.add_argument("--save_latlon_masks", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: save the latlon masks used for prediction as geotif.")

    parser.add_argument("--from_aws", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: image is loaded from aws s3 (not for free, needs aws credentials)")

    parser.add_argument("--download_from_aws", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: downloads the image first from aws in SAFE format.")
    parser.add_argument("--remove_image_after_pred", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: deletes the image after saving the prediction.")
    parser.add_argument("--sentinel2_dir", help="directory to save sentinel2 data (temporarily)")

    # fine-tune and re-weighting strategies
    parser.add_argument("--finetune_strategy", default='FT_Lm_SRCB',
                        help="finetuned model, subdirectory in the model_i directory.",
                        choices=['', None,
                                 'FT_ALL_CB', 'FT_L_CB', 'RT_L_CB',
                                 'FT_ALL_SRCB', 'FT_L_SRCB', 'RT_L_SRCB',
                                 'FT_Lm_SRCB', 'RT_Lm_SRCB',
                                 'RT_L_IB',
                                 'ST_geoshift_IB', 'ST_geoshiftscale_IB'])
    return parser


def predict(model, args, model_weights=None,
            ds_pred=None, batch_size=1, num_workers=8,
            train_target_mean=0, train_target_std=1):

    # convert train statistics to tensor on cpu
    train_target_mean = torch.tensor(train_target_mean)
    train_target_std = torch.tensor(train_target_std)

    dl_pred = DataLoader(ds_pred, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # load best model weights
    model.load_state_dict(model_weights)
    model.eval()

    # init predictions and estimated standard deviations
    pred_dict = {'predictions': []}
    if args.return_variance:
        pred_dict['std'] = []

    with torch.no_grad():
        # Note: file=sys.stdout is needed to avoid error logging. per default tqdm writes to sys.stderr
        for step, data_dict in enumerate(tqdm(dl_pred, ncols=100, desc='pred', file=sys.stdout)):  # for each training step

            inputs = data_dict[args.input_key]
            inputs = inputs.to(DEVICE, non_blocking=True)

            if args.return_variance:
                predictions, variances = model.forward(inputs)
                std = torch.sqrt(variances)
                pred_dict['std'].extend(list(std.cpu()))
            else:
                predictions = model.forward(inputs)
            pred_dict['predictions'].extend(list(predictions.cpu()))

        for key in pred_dict.keys():
            if pred_dict[key]:
                pred_dict[key] = torch.stack(pred_dict[key], dim=0)
                print("val_dict['{}'].shape: ".format(key), pred_dict[key].shape)

    # denormalize predictions and targets
    if args.normalize_targets:
        pred_dict['predictions'] = denormalize(pred_dict['predictions'], train_target_mean, train_target_std)
        if args.return_variance:
            # denormalize the variances by multiplying with the target variance
            pred_dict['std'] *= train_target_std

    # convert torch tensor to numpy
    for key in pred_dict.keys():
        pred_dict[key] = pred_dict[key].data.cpu().numpy()

    return pred_dict


if __name__ == "__main__":

    # parse deploy arguments
    parser = setup_parser()
    args, unknown = parser.parse_known_args()

    # sample model id from ensemble
    args.model_id = np.random.choice(args.num_models)
    args.model_dir = os.path.join(args.model_dir, "model_{}".format(args.model_id), args.finetune_strategy)
    print("Sampled model_id: {} out of {} models in ensemble.".format(args.model_id, args.num_models))
    print("Using args.model_dir: ", args.model_dir)

    args_dict = vars(args)
    args_dict_deploy = copy.deepcopy(args_dict)
    # load args from experiment dir with full train set
    print("Loading args from trained models directory...")
    args_saved = load_args_from_json(os.path.join(args.model_dir, 'args.json'))
    # update args with model_dir args
    args_dict.update(args_saved)
    # updage args with deploy args
    args_dict.update(args_dict_deploy)

    # set missing args:
    if not hasattr(args, 'manual_init'):
        args.manual_init = False
    if not hasattr(args, 'freeze_last_mean'):
        args.freeze_last_mean = False
    if not hasattr(args, 'freeze_last_var'):
        args.freeze_last_var = False
    if not hasattr(args, 'geo_shift'):
        args.geo_shift = False
    if not hasattr(args, 'geo_scale'):
        args.geo_scale = False
    if not hasattr(args, 'separate_lat_lon'):
        args.separate_lat_lon = False
    print(args_dict)

    # get the file_name used for saving the prediction file
    if '.zip' in args.deploy_image_path:
        file_name = os.path.basename(args.deploy_image_path).strip(".zip")
    elif 'tiles/' in args.deploy_image_path:
        # replace all "/" with "_" to get a filename instead of path
        file_name = args.deploy_image_path.replace('/', '_')
        args.from_aws = True
    else:
        # the deploy_image_path was set to the filename (e.g. "S2B_MSIL2A_20200807T102559_N0214_R108_T32TMT_20200807T132026")
        file_name = args.deploy_image_path

    print('file_name:', file_name)

    # download the sentinel2 images from aws
    if args.download_from_aws:
        print('downloading image from aws... ', file_name)
        try:
            time.sleep(float(np.random.rand(1)*20))
            start = time.time()
            path_zip_file = download_and_zip_safe_from_aws(image_name=file_name,
                                                           path_sentinel_2A=args.sentinel2_dir)
            args.deploy_image_path = path_zip_file
            end = time.time()
            print("TIME DOWNLOAD FROM AWS:", time.strftime('%H:%M:%S', time.gmtime(end - start)))
            print("args.deploy_image_path is set to: ", path_zip_file)

        except RuntimeError:
            # write failed path to txt file
            print("logging failed path to: ", args.filepath_failed_image_paths)
            with open(args.filepath_failed_image_paths, 'a') as f:
                f.write('{}\n'.format(args.deploy_image_path))
            raise RuntimeError("Sentinel-2 image could not be downloaded for: {}".format(args.deploy_image_path))
        except botocore.exceptions.ProxyConnectionError:
            print('Error with proxy connection: botocore.exceptions.ProxyConnectionError')
            sys.exit(222)
        except urllib3.exceptions.ProxyError:
            print('Error with proxy connection: urllib3.exceptions.ProxyError')
            sys.exit(222)

    if args.input_lat_lon:
        args.channels = 15  # 12 sentinel2 bands + 3 channels (lat, sin(lon), cos(lon))

    # ----------------------------------------
    if not os.path.exists(args.deploy_dir):
        os.makedirs(args.deploy_dir)

    # Load  train statistics
    if os.path.exists(os.path.join(args.model_dir, 'train_input_mean.npy')):
        train_input_mean = np.load(os.path.join(args.model_dir, 'train_input_mean.npy'))
        train_input_std = np.load(os.path.join(args.model_dir, 'train_input_std.npy'))
    else:
        train_input_mean, train_input_std = 0, 1

    if os.path.exists(os.path.join(args.model_dir, 'train_target_mean.npy')):
        train_target_mean = np.load(os.path.join(args.model_dir, 'train_target_mean.npy'))
        train_target_std = np.load(os.path.join(args.model_dir, 'train_target_std.npy'))
    else:
        train_target_mean, train_target_std = 0, 1

    print('train_input_mean', train_input_mean)
    print('train_input_std', train_input_std)

    print('train_target_mean', train_target_mean)
    print('train_target_std', train_target_std)

    # setup input transforms
    input_transforms = Normalize(mean=train_input_mean, std=train_input_std)

    # create dataset
    try:
        start = time.time()
        ds_pred = Sentinel2Deploy(path=args.deploy_image_path,
                                  input_transforms=input_transforms,
                                  input_lat_lon=args.input_lat_lon,
                                  patch_size=args.deploy_patch_size,
                                  border=16,
                                  from_aws=args.from_aws)
        end = time.time()
        print("TIME LOADING BANDS:", time.strftime('%H:%M:%S', time.gmtime(end - start)))
    except RuntimeError:
        # write failed path to txt file
        print("logging failed path to: ", args.filepath_failed_image_paths)
        with open(args.filepath_failed_image_paths, 'a') as f:
            f.write('{}\n'.format(args.deploy_image_path))
        raise RuntimeError("Sentinel-2 image could not be loaded from: {}".format(args.deploy_image_path))

    # load model architecture
    architecture_collection = Architectures(args=args)
    net = architecture_collection(args.architecture)(num_outputs=1)

    net.cuda()  # move model to GPU

    # Load latest weights from checkpoint file (alternative load best val epoch from best_weights.pt)
    print('Loading model weights from latest checkpoint ...')
    checkpoint_path = Path(args.model_dir) / 'checkpoint.pt'
    checkpoint = torch.load(checkpoint_path)
    model_weights = checkpoint['model_state_dict']

    pred_dict = predict(model=net, args=args, model_weights=model_weights,
                        ds_pred=ds_pred, batch_size=args.deploy_batch_size, num_workers=args.num_workers_deploy,
                        train_target_mean=train_target_mean, train_target_std=train_target_std)

    # recompose predictions and variances
    recomposed_tiles = {}
    for k in pred_dict:
        print('recomposing {} ...'.format(k))
        recomposed_tiles[k] = ds_pred.recompose_patches(pred_dict[k], out_type=np.float32)
        print(recomposed_tiles[k].shape)

        if '.zip' in args.deploy_image_path:
            tif_path = os.path.join(args.deploy_dir, file_name + '_{}.tif'.format(k))
        else:
            tif_path = os.path.join(args.deploy_dir, file_name + '{}.tif'.format(k))

        save_array_as_geotif(out_path=tif_path,
                             array=recomposed_tiles[k],
                             tile_info=ds_pred.tile_info)

    if args.save_latlon_masks:
        # save lat mask
        tif_path = os.path.join(args.deploy_dir, file_name + '{}.tif'.format('lat'))
        save_array_as_geotif(out_path=tif_path,
                            array=ds_pred.lat_mask[ds_pred.border:-ds_pred.border, ds_pred.border:-ds_pred.border],
                            tile_info=ds_pred.tile_info)
        # save lon mask
        tif_path = os.path.join(args.deploy_dir, file_name + '{}.tif'.format('lon'))
        save_array_as_geotif(out_path=tif_path,
                            array=ds_pred.lon_mask[ds_pred.border:-ds_pred.border, ds_pred.border:-ds_pred.border],
                            tile_info=ds_pred.tile_info)

    if args.remove_image_after_pred:
        print('REMOVING IMAGE DATA: ', args.deploy_image_path)
        os.remove(args.deploy_image_path)

