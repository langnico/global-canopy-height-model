import os
import numpy as np
import json
import torch
import shutil
import wandb

from gchm.datasets.dataset_sentinel2 import make_concat_dataset
from gchm.trainer import Trainer
from gchm.models.architectures import Architectures
from gchm.utils.parser import setup_parser, save_args_to_json, set_finetune_strategy_params
from gchm.utils.loss import get_metric_lookup_dict, SampleWeightedLoss, ShrinkageLoss
from gchm.utils.preprocessing import compute_train_mean_std
from gchm.utils.transforms import Normalize, NormalizeVariance
from gchm.utils.h5_utils import load_paths_from_dicretory, filter_paths_by_tile_names


def run_test(dataset, model_weights, out_dir, trainer):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    test_metrics, test_dict, test_metric_string = trainer.test(model_weights=model_weights,
                                                               ds_test=dataset)
    # save results
    with open(os.path.join(out_dir, 'results.txt'), 'w') as f:
        f.write(test_metric_string)

    with open(os.path.join(out_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f)

    for k in test_dict:
        print(k, test_dict[k].shape)
        np.save(os.path.join(out_dir, '{}.npy'.format(k)), arr=test_dict[k])


if __name__ == "__main__":

    # load args set parameters
    parser = setup_parser()
    args, unknown = parser.parse_known_args()

    if args.finetune_strategy:
        args = set_finetune_strategy_params(args)
    
    if args.custom_sampler == 'BatchSampler':
        print('Ignore np.VisibleDeprecationWarning for fancy indexing with BatchSampler.')
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

    print('out_dir: ', args.out_dir)

    if args.input_lat_lon:
        args.channels = 15  # 12 sentinel2 bands + 3 channels (lat, sin(lon), cos(lon))

    # ----------------------------------------
    # copy base model directory with pre-trained model to out_dir
    if args.base_model_dir:
        print('Copying base_model_dir to out_dir...')
        if not os.path.exists(args.out_dir):
            shutil.copytree(src=args.base_model_dir, dst=args.out_dir, ignore=shutil.ignore_patterns('test', 'val',
                                                                                                     'FT_ALL_CB', 'FT_L_CB', 'RT_L_CB',
                                                                                                     'FT_ALL_SRCB', 'FT_L_SRCB', 'RT_L_SRCB',
                                                                                                     'FT_Lm_SRCB', 'RT_Lm_SRCB',
                                                                                                     'RT_L_IB',
                                                                                                     'ST_geoshift_IB', 'ST_geoshiftscale_IB'))
        else:
            print('base_model_dir exsits already')
            print('Layer has been re-initialized in a previous fine-tuning run. Setting args.reinit_last_layer=False.')
            args.reinit_last_layer = False
            args.load_optimizer_state_dict = True
    
    # create directory with subdirectory for tensorboard logs
    args.log_dir = os.path.join(args.out_dir, 'log')
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # init wandb for logging
    options = vars(args)
    run = wandb.init(
        project="gchm",
        config=options,
        reinit=True,
        mode="online"  # set to "online" "offline" "disabled"
    )

    if args.merged_h5_files:
        # use merged h5 files for each data split. Note: make_concat_dataset expects a list of paths.
        paths_h5_train = [os.path.join(args.h5_dir, '{}_train.h5'.format(args.region_name))]
        paths_h5_val = [os.path.join(args.h5_dir, '{}_val.h5'.format(args.region_name))]
    else:
        assert set(args.train_tiles).isdisjoint(set(args.val_tiles)), "train_tiles and val_tiles overlap."
        print('args.train_tiles', args.train_tiles)
        print('args.val_tiles', args.val_tiles)
        paths_h5 = load_paths_from_dicretory(dir=args.h5_dir)
        # split paths by tile names (spatial split)
        paths_h5_train = filter_paths_by_tile_names(paths=paths_h5, tile_names=args.train_tiles)
        paths_h5_val = filter_paths_by_tile_names(paths=paths_h5, tile_names=args.val_tiles)

    print('len(paths_h5_train): ', len(paths_h5_train))
    print('len(paths_h5_val): ', len(paths_h5_val))

    metrics_lookup = get_metric_lookup_dict()
    if not args.return_variance:
        del metrics_lookup['GNLL']
        del metrics_lookup['LNLL']

    if args.loss_key == 'shrinkage':
        metrics_lookup['shrinkage'] = ShrinkageLoss()

    # make raw train dataset to compute statistics for normalization (inputs, targets)
    ds_train_raw = make_concat_dataset(paths_h5=paths_h5_train)

    if args.data_stats_dir is None:
        args.data_stats_dir = args.out_dir

    # input statistics and normalization
    if not os.path.exists(os.path.join(args.data_stats_dir, 'train_input_mean.npy')):
        print('computing input statistics on training data...')
        train_input_mean, train_input_std = compute_train_mean_std(ds_train=ds_train_raw, data_key='inputs',
                                                                   num_samples=args.num_samples_statistics,
                                                                   num_workers=args.num_workers,
                                                                   batch_size=128)

    else:
        print('loading input statistics of training data...')
        train_input_mean = np.load(os.path.join(args.data_stats_dir, 'train_input_mean.npy'))
        train_input_std = np.load(os.path.join(args.data_stats_dir, 'train_input_std.npy'))

    print("train_input_mean: ", train_input_mean.shape, train_input_mean.dtype)
    print(train_input_mean)
    print("train_input_std: ", train_input_std.shape, train_input_std.dtype)
    print(train_input_std)

    # save statistics to out_dir
    np.save(os.path.join(args.out_dir, 'train_input_mean.npy'), train_input_mean)
    np.save(os.path.join(args.out_dir, 'train_input_std.npy'), train_input_std)

    input_transforms = Normalize(mean=train_input_mean, std=train_input_std)

    # target statistics and normalization
    if args.normalize_targets:
        if not os.path.exists(os.path.join(args.data_stats_dir, 'train_target_mean.npy')):
            print('computing target statistics on training data...')
            train_target_mean, train_target_std = compute_train_mean_std(ds_train=ds_train_raw, data_key='labels_mean',
                                                                         num_samples=args.num_samples_statistics,
                                                                         num_workers=args.num_workers,
                                                                         batch_size=128)

        else:
            print('loading target statistics of training data...')
            train_target_mean = np.load(os.path.join(args.data_stats_dir, 'train_target_mean.npy'))
            train_target_std = np.load(os.path.join(args.data_stats_dir, 'train_target_std.npy'))

        print("train_target_mean: ", train_target_mean.shape, train_target_mean.dtype)
        print(train_target_mean)
        print("train_target_std: ", train_target_std.shape, train_target_std.dtype)
        print(train_target_std)

        # save statistics to out_dir
        np.save(os.path.join(args.out_dir, 'train_target_mean.npy'), train_target_mean)
        np.save(os.path.join(args.out_dir, 'train_target_std.npy'), train_target_std)

        target_transforms = Normalize(mean=train_target_mean, std=train_target_std)
        target_var_transforms = NormalizeVariance(std=train_target_std)
    else:
        train_target_mean, train_target_std = 0, 1  # has no effect
        target_transforms, target_var_transforms = None, None

    # close all h5 files such that they can be opened again with normalization transforms
    del ds_train_raw

    # load train target distribution for class-balanced training/fine-tuning
    if args.class_balanced:
        path_bin_weights = os.path.join(args.data_stats_dir, 'train_target_distribution.npy')
    else:
        path_bin_weights = None
        
    # create datasets (train, val)
    if args.do_train:
        ds_train = make_concat_dataset(paths_h5=paths_h5_train,
                                       input_transforms=input_transforms,
                                       target_transforms=target_transforms,
                                       target_var_transforms=target_var_transforms,
                                       input_lat_lon=args.input_lat_lon,
                                       use_cloud_free=args.use_cloud_free,
                                       path_bin_weights=path_bin_weights,
                                       weight_key=args.weight_key)

        print('len(ds_train): ', len(ds_train))
    else:
        ds_train = None

    ds_val = make_concat_dataset(paths_h5=paths_h5_val,
                                 input_transforms=input_transforms,
                                 target_transforms=target_transforms,
                                 target_var_transforms=target_var_transforms,
                                 input_lat_lon=args.input_lat_lon,
                                 use_cloud_free=args.use_cloud_free,
                                 path_bin_weights=path_bin_weights,
                                 weight_key=args.weight_key)


    print('len(ds_val):   ', len(ds_val))

    # init sample weighted loss
    if args.weight_key:
        sample_weighted_loss = SampleWeightedLoss(loss_key=args.loss_key)
        print('USING SAMPLE WEIGHTED {}. Weighting samples with: {}'.format(args.loss_key, args.weight_key))
    else:
        sample_weighted_loss = None

    # load model architecture
    architecture_collection = Architectures(args=args)
    net = architecture_collection(args.architecture)(num_outputs=1)

    net.cuda()  # move model to GPU

    # save arguments
    save_args_to_json(file_path=os.path.join(args.out_dir, 'args.json'), args=args)


    # init trainer
    trainer = Trainer(model=net, args=args,
                      ds_train=ds_train, ds_val=ds_val,
                      metrics_lookup=metrics_lookup,
                      train_input_mean=train_input_mean, train_input_std=train_input_std,
                      train_target_mean=train_target_mean, train_target_std=train_target_std,
                      sample_weighted_loss=sample_weighted_loss)

    # --- train model ---
    if args.max_grad_norm:
        print('GRADIENT CLIPPING IS ON. Total grad norm is clipped to args.max_grad_norm: ', args.max_grad_norm)
    if args.max_grad_value:
        print('GRADIENT CLIPPING IS ON. Individual gradient values are clipped to args.max_grad_value: ', args.max_grad_value)
    if args.debug:
        print('DEBUG MODE. args.debug is true. Logging and printing additional debug outputs. Attention, tensorboard logs might become large in volume.')


    if args.do_train:
        training_metrics = trainer.train()
    else:
        print('SKIP TRAINING. args.do_train=False')

    # Load latest weights from checkpoint file (alternative load best val epoch from best_weights.pt)
    print('Loading model weights from latest checkpoint ...')
    checkpoint = torch.load(trainer.checkpoint_path)
    model_weights = checkpoint['model_state_dict']

    # --- test model ---
    print('testing on ds_val ...')
    run_test(dataset=ds_val,
             model_weights=model_weights,
             out_dir=os.path.join(args.out_dir, 'val'),
             trainer=trainer)




