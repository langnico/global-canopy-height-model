import argparse
import numpy as np
import json


def setup_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", default='./tmp/', help="output directory for the experiment")
    parser.add_argument("--h5_dir", default='/scratch2/data/global_vhm/GEDI_patches_CH_2020/h5_patches', help="path to directory with h5 datasets")
    parser.add_argument("--merged_h5_files", type=str2bool, nargs='?', const=True, default=False, help="if True: the h5_dir must contain merged h5 files REGION_train.h5, REGION_val.h5, REGION_test.h5.")
    parser.add_argument("--region_name", default='GLOBAL_GEDI', help="name of the region used if merged_h5_files is True")
    parser.add_argument("--input_lat_lon", type=str2bool, nargs='?', const=True, default=False, help="if True: lat lon masks are used as additional input channels.")
    parser.add_argument("--separate_lat_lon", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: lat lon input is not passed to the xception backbone, but only to the geo prior net.")
    parser.add_argument("--geo_shift", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: lat lon input is used to shift the predictions conditioned on the location.")
    parser.add_argument("--geo_scale", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: lat lon input is used to scale the predictions conditioned on the location.")
    parser.add_argument("--input_key", default='inputs', help="input key returned from custom torch dataset")
    parser.add_argument("--label_mean_key", default='labels_mean', help="target key (mean) returned from custom torch dataset")
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False, help="if True: some (costly) debug outputs/logs are computed ")
    parser.add_argument("--do_profile", type=str2bool, nargs='?', const=True, default=False, help="if True: creates torch.profile ")

    parser.add_argument("--channels", default=12, help="number of epochs to train", type=int)
    parser.add_argument("--patch_size", default=15, help="number of epochs to train", type=int)
    parser.add_argument("--long_skip", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: a long skip connection is used from 1x1 kernel features to final features.")

    parser.add_argument("--architecture", default='xceptionS2_08blocks_256', help="model architecture name",
                        choices=['xceptionS2_08blocks', 'xceptionS2_18blocks',  # 728 filters
                                 'xceptionS2_08blocks_256', 'xceptionS2_08blocks_512',
                                 'xceptionS2_18blocks_256', 'xceptionS2_18blocks_512',
                                 'linear_classifier', 'powerlaw_classifier', 'simple_fcn', 'simple_fcn_powerlaw'])
    parser.add_argument("--manual_init", type=str2bool, nargs='?', const=True, default=False, help="if True: re-initializes layer weights with custom init. strategy ")
    parser.add_argument("--return_variance", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: the network has two outputs a mean and a variance.")
    parser.add_argument("--max_pool_predictions", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: predictions are max pooled before supervision (to match GEDI footprint)")
    parser.add_argument("--max_pool_labels", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: labels are max pooled before supervision (to match GEDI footprint)")
    parser.add_argument("--loss_key", default='MSE', help="Loss name to optimize")
    parser.add_argument("--weight_key", default=None,
                        help="Key in the dict returned from the custom dataset class that is used to weight the loss")
    parser.add_argument("--eps", default=0, help="eps added to weights defined by weight_key (this may be set to a small positive number to not forget about the frequent samples)", type=float)

    parser.add_argument("--optimizer", default='ADAM', help="optimizer", choices=['ADAM', 'SGD'])
    parser.add_argument("--scheduler", default='MultiStepLR', help="learning rate scheduler", choices=['MultiStepLR', 'OneCycleLR'])
    parser.add_argument("--base_learning_rate", default=0.001, help="base learning rate", type=float)
    parser.add_argument("--l2_lambda", default=0, help="weight of l2 regularizer", type=float)
    parser.add_argument("--batch_size", default=64, help="number of samples per batch (iteration)", type=int)
    parser.add_argument("--num_workers", default=8, help="number of workers in dataloader", type=int)
    parser.add_argument("--model_weights_path", help="path to pre-trained model weights", type=str_or_none, default=None)
    parser.add_argument("--nb_epoch", default=50, help="number of epochs to train", type=int)
    parser.add_argument("--iterations_per_epoch", default=5000, help="number of iterations that define one epoch. if None: one epoch corresponds to the full dataset len(dl_train)", type=int)
    parser.add_argument("--max_grad_norm", default=None, help="max total norm for gradient norm clipping", type=str2none)
    parser.add_argument("--max_grad_value", default=None, help="max gradient value (+/-) for gradient value clipping", type=str2none)
    parser.add_argument("--custom_sampler", help="class name (str) of custom sampler type. Uses default random sampler if set to None.", choices=[None, 'SliceBatchSampler', 'BatchSampler'], type=str_or_none, default=None)
    parser.add_argument("--slice_step", default=1, help="If --custom_sampler='SliceBatchSampler': access every slice_step sample in the data array with slice(start, stop, slice_step)", type=int)
    parser.add_argument("--lr_milestones", default=[100, 200], nargs='+', type=int,
                        help="List of epoch indices at which the learning rate is dropped by factor 10. Must be increasing.")

    # fine-tune and re-weighting strategies
    parser.add_argument("--finetune_strategy", default=None,
                        help="Custom short name for setting the fine-tuning and re-weighting strategy. FT: Fine-tune, RT: re-train, ST: separate training, ALL: full network, L: last linear layers, Lm: last linear layer for mean output (freezes the layer for variance output), CB: class-balanced, SRCB: square root class-balanced, IB: instance-balanced (no reweighting)",
                        choices=['', None,
                                 'FT_ALL_CB', 'FT_L_CB', 'RT_L_CB',
                                 'FT_ALL_SRCB', 'FT_L_SRCB', 'RT_L_SRCB',
                                 'FT_Lm_SRCB', 'RT_Lm_SRCB',
                                 'RT_L_IB',
                                 'ST_geoshift_IB', 'ST_geoshiftscale_IB'],
                        type=str_or_none)
    parser.add_argument("--base_model_dir", default=None, help="Path to pretrained model directory. This directory will first be copied to out_dir in which the model is fine tuned.", type=str_or_none)
    parser.add_argument("--freeze_features", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: Only the last fully connected layer is optimized (for fine tuning)")
    parser.add_argument("--freeze_last_mean", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: Freezes the last mean regression layer. Used e.g. to only finetune the mean layer or to train the GeoPriorNet in a second stage to correct the residuals.")
    parser.add_argument("--freeze_last_var", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: Freezes the last variance regression layer. Used e.g. to only finetune the mean layer or to train the GeoPriorNet in a second stage to correct the residuals.")
    parser.add_argument("--reinit_last_layer", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: Re-initialize the last layer (i.e. linear regressor or linear classifier).")
    parser.add_argument("--class_balanced", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: Will re-weight the samples using inverse class frequency (i.e. bin frequency for regression).")
    parser.add_argument("--load_optimizer_state_dict", type=str2bool, nargs='?', const=True, default=True,
                        help="if True: loads existing optimizer_state_dict")

    parser.add_argument("--num_samples_statistics", default=1e6, type=float,
                        help="number of samples used to calculate training statistics")
    parser.add_argument("--data_stats_dir", help="path to dataset statistics (input, target mean and std)",
                        type=str_or_none, default=None)
    parser.add_argument("--normalize_targets", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: targets are normalized to mean=0 and std=1.")
    parser.add_argument("--do_train", type=str2bool, nargs='?', const=True, default=True,
                        help="if False: training will be skipped")

    parser.add_argument("--train_tiles", default=None,
                        help="List of Sentinel-2 tile names used for training.", nargs='+')
    parser.add_argument("--val_tiles", default=None,
                        help="List of Sentinel-2 tile names used for validation.", nargs='+')

    parser.add_argument("--use_cloud_free", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: Dataset returns only cloud free patches. Not needed if h5 patches were already filtered in h5. ")

    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2none(v):
    if v.lower() in ('none', '', 'nan', '0', '0.0'):
        return None
    else:
        return float(v)


def str_or_none(v):
    if v.lower() in ('none', '', 'nan', '0', '0.0'):
        return None
    else:
        return str(v)


class StoreAsArray(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super(StoreAsArray, self).__call__(parser, namespace, values, option_string)


def save_args_to_json(file_path, args):
    with open(file_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_args_from_json(file_path):
    with open(file_path, 'r') as f:
        args_dict = json.load(f)
    return args_dict


def set_finetune_strategy_params(args):

    # class-balanced inverse frequency (CB)
    if args.finetune_strategy == 'FT_ALL_CB':
        args.reinit_last_layer = False
        args.freeze_features = False
        args.freeze_last_mean = False
        args.freeze_last_var = False
        args.class_balanced = True
        args.weight_key = 'inv_freq'
        args.load_optimizer_state_dict = True

    elif args.finetune_strategy == 'FT_L_CB':
        args.reinit_last_layer = False
        args.freeze_features = True
        args.freeze_last_mean = False
        args.freeze_last_var = False
        args.class_balanced = True
        args.weight_key = 'inv_freq'
        args.load_optimizer_state_dict = True

    elif args.finetune_strategy == 'RT_L_CB':
        args.reinit_last_layer = True
        args.freeze_features = True
        args.freeze_last_mean = False
        args.freeze_last_var = False
        args.class_balanced = True
        args.weight_key = 'inv_freq'
        args.load_optimizer_state_dict = False

    # class-balanced inverse of the square root frequency (SRCB)
    elif args.finetune_strategy == 'FT_ALL_SRCB':
        args.reinit_last_layer = False
        args.freeze_features = False
        args.freeze_last_mean = False
        args.freeze_last_var = False
        args.class_balanced = True
        args.weight_key = 'inv_sqrt_freq'
        args.load_optimizer_state_dict = True

    elif args.finetune_strategy == 'FT_L_SRCB':
        args.reinit_last_layer = False
        args.freeze_features = True
        args.freeze_last_mean = False
        args.freeze_last_var = False
        args.class_balanced = True
        args.weight_key = 'inv_sqrt_freq'
        args.load_optimizer_state_dict = True

    elif args.finetune_strategy == 'RT_L_SRCB':
        args.reinit_last_layer = True
        args.freeze_features = True
        args.freeze_last_mean = False
        args.freeze_last_var = False
        args.class_balanced = True
        args.weight_key = 'inv_sqrt_freq'
        args.load_optimizer_state_dict = False

    elif args.finetune_strategy == 'RT_L_IB':
        args.reinit_last_layer = True
        args.freeze_features = True
        args.freeze_last_mean = False
        args.freeze_last_var = False
        args.class_balanced = False
        args.load_optimizer_state_dict = False

    elif args.finetune_strategy == 'FT_Lm_SRCB':
        args.reinit_last_layer = False
        args.freeze_features = True
        args.freeze_last_mean = False
        args.freeze_last_var = True
        args.class_balanced = True
        args.weight_key = 'inv_sqrt_freq'
        args.load_optimizer_state_dict = True

    elif args.finetune_strategy == 'RT_Lm_SRCB':
        args.reinit_last_layer = True
        args.freeze_features = True
        args.freeze_last_mean = False
        args.freeze_last_var = True
        args.class_balanced = True
        args.weight_key = 'inv_sqrt_freq'
        args.load_optimizer_state_dict = False

    elif args.finetune_strategy == 'ST_geoshift_IB':
        args.reinit_last_layer = False
        args.freeze_features = True
        args.freeze_last_mean = True
        args.freeze_last_var = True
        args.class_balanced = False

        args.input_lat_lon = True
        args.separate_lat_lon = True
        args.geo_shift = True
        args.geo_scale = False
        args.data_stats_dir = args.data_stats_dir.replace('latlon_False', 'latlon_True')
        args.load_optimizer_state_dict = False

    elif args.finetune_strategy == 'ST_geoshiftscale_IB':
        args.reinit_last_layer = False
        args.freeze_features = True
        args.freeze_last_mean = True
        args.freeze_last_var = True
        args.class_balanced = False

        args.input_lat_lon = True
        args.separate_lat_lon = True
        args.geo_shift = True
        args.geo_scale = True
        args.data_stats_dir = args.data_stats_dir.replace('latlon_False', 'latlon_True')
        args.load_optimizer_state_dict = False

    else:
        raise ValueError("This finetune strategy '{}' is not implemented.".format(args.finetune_strategy))

    return args

