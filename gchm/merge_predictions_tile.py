import numpy as np
from glob import glob
import os
import argparse

from gchm.utils.gdal_process import save_array_as_geotif, load_tif_as_array
from gchm.utils.parser import str2bool


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_name", help="tile name to collect and reduce predictions with mean")
    parser.add_argument("--out_dir", help="path to save reduced predictions per tile")
    parser.add_argument("--deploy_dir", help="path to directory with individual tile (date) predictions.tif")
    parser.add_argument("--out_type", default="float32", help="output data type for geotif. See choices.",  choices=["float32", "uint8", "uint16"])
    parser.add_argument("--nodata_value", default=None, type=float)
    parser.add_argument("--from_aws", type=str2bool, nargs='?', const=True, default=False,
                        help="if True: split tile_name according to pattern. e.g. 32TMT becomes 32_T_MT")
    parser.add_argument("--reduction", default='inv_var_mean', help="method to reduce multiple predictions per tile.",
                        choices=['mean', 'median', 'inv_var_mean'])
    return parser


if __name__ == "__main__":

    parser = setup_parser()
    args, unknown = parser.parse_known_args()

    print('reduction: ', args.reduction)
    std_final = None
    out_file_path = os.path.join(args.out_dir, '{}_pred.tif'.format(args.tile_name))
    out_file_path_std = os.path.join(args.out_dir, '{}_std.tif'.format(args.tile_name))

    if args.from_aws:
        search_pattern = list(args.tile_name)
        search_pattern.insert(2, '_')
        search_pattern.insert(4, '_')
        search_pattern = "".join(search_pattern)
    else:
        search_pattern = args.tile_name

    pred_paths = glob(os.path.join(args.deploy_dir, '*{}*_predictions.tif'.format(search_pattern)))
    num_paths = len(pred_paths)
    print('number of pred_paths', num_paths)
    print(pred_paths[0])
    
    log_dir = args.out_dir.replace('_merge', '_merge_logs')
    log_file_path = os.path.join(log_dir, '{}.txt'.format(args.tile_name))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # save the paths used for merging
    with open(log_file_path, 'w') as f:
        for p in pred_paths:
            f.write('{}\n'.format(os.path.basename(p)))

    if args.reduction == 'mean':
        # "map-reduce" mean and total variance
        pred_sum_weighted = None
        pred_squared_sum_weighted = None
        var_sum_weighted = None
        sum_weights = None

        for step, pred_path in enumerate(pred_paths):
            std_path = pred_path.replace('_predictions.tif', '_std.tif')
            print('step: {}/{}'.format(step, num_paths))
            print(pred_path)
            pred, tile_info = load_tif_as_array(pred_path)
            std, tile_info = load_tif_as_array(std_path)
            # equal weight on all predictions (to compute the mean)
            weights = np.ones_like(pred)
            print(pred.shape)
            if step == 0:
                pred_sum_weighted = pred * weights
                pred_squared_sum_weighted = (pred ** 2) * weights
                var_sum_weighted = (std ** 2) * weights
                sum_weights = weights
            else:
                pred_sum_weighted = np.nansum(np.stack((pred_sum_weighted, pred * weights)), axis=0)
                pred_squared_sum_weighted = np.nansum(np.stack((pred_squared_sum_weighted, (pred ** 2) * weights)),
                                                      axis=0)
                var_sum_weighted = np.nansum(np.stack((var_sum_weighted, (std ** 2) * weights)), axis=0)
                sum_weights = np.nansum(np.stack((sum_weights, weights)), axis=0)

        # normalize by the sum of weights
        with np.errstate(divide='ignore', invalid='ignore'):
            pred_final = pred_sum_weighted / sum_weights
            pred_squared_sum_weighted /= sum_weights
            var_sum_weighted /= sum_weights
        # compute the total variance --> resp. the standard deviation
        std_final = np.sqrt(var_sum_weighted + pred_squared_sum_weighted - pred_final ** 2)
        print('pred_final.shape: ', pred_final.shape)
        print('std_final.shape:  ', std_final.shape)

    if args.reduction == 'inv_var_mean':
        # "map-reduce" inverse variance weighted mean and total variance
        pred_sum_weighted = None
        pred_squared_sum_weighted = None
        var_sum_weighted = None
        sum_weights = None

        for step, pred_path in enumerate(pred_paths):
            std_path = pred_path.replace('_predictions.tif', '_std.tif')
            print('step: {}/{}'.format(step, num_paths))
            print(pred_path)
            pred, tile_info = load_tif_as_array(pred_path)
            std, tile_info = load_tif_as_array(std_path)
            weights = 1 / (std ** 2)
            print(pred.shape)
            if step == 0:
                pred_sum_weighted = pred * weights
                pred_squared_sum_weighted = (pred ** 2) * weights
                var_sum_weighted = (std ** 2) * weights
                sum_weights = weights
            else:
                pred_sum_weighted = np.nansum(np.stack((pred_sum_weighted, pred * weights)), axis=0)
                pred_squared_sum_weighted = np.nansum(np.stack((pred_squared_sum_weighted, (pred ** 2) * weights)), axis=0)
                var_sum_weighted = np.nansum(np.stack((var_sum_weighted, (std ** 2) * weights)), axis=0)
                sum_weights = np.nansum(np.stack((sum_weights, weights)), axis=0)

        # normalize by the sum of weights
        with np.errstate(divide='ignore', invalid='ignore'):
            pred_final = pred_sum_weighted / sum_weights
            pred_squared_sum_weighted /= sum_weights
            var_sum_weighted /= sum_weights
        # compute the total variance --> resp. the standard deviation
        std_final = np.sqrt(var_sum_weighted + pred_squared_sum_weighted - pred_final ** 2)
        print('pred_final.shape: ', pred_final.shape)
        print('std_final.shape:  ', std_final.shape)

    if args.reduction == 'median':
        # median
        pred_collect = None
        for step, pred_path in enumerate(pred_paths):
            print('step: {}/{}', step, num_paths)
            print(pred_path)
            pred, tile_info = load_tif_as_array(pred_path)
            
            if pred_collect is None:
                # init array
                pred_collect = np.full(shape=(len(pred_paths), pred.shape[0], pred.shape[1]), fill_value=np.nan, dtype=np.float16)

            pred_collect[step] = pred.astype(np.float16)

        # apply median along first axis ignore nodata value (like np.nanmean)
        print('pred_collect: ', pred_collect.shape, pred_collect.dtype)
        pred_median = np.nanmedian(a=pred_collect, axis=0, overwrite_input=True)
        print('pred_median.shape', pred_median.shape)

        pred_final = pred_median

    print('tile_info:')
    print(tile_info)

    if args.out_type in ["uint8", "uint16"]:
        # set compression
        compression = 'LZW'
        predictor = 2
        # set nan values to nodata_value
        args.nodata_value = int(args.nodata_value)
        mask_nan = np.isnan(pred_final)
        pred_final[mask_nan] = args.nodata_value
        # round to integer and cast to uint8 or uint16
        pred_final = np.rint(pred_final).astype(args.out_type)
        if std_final is not None:
            std_final[mask_nan] = args.nodata_value
            # round to integer and cast to uint8 or uint16
            std_final = np.rint(std_final).astype(args.out_type)
    else:
        # set compression for float32
        compression = 'DEFLATE'
        predictor = 2
    
    print('args.nodata_value: ', args.nodata_value)
    
    if args.reduction == "median":
        out_file_path = out_file_path.replace("_pred.tif", "_pred_median.tif")
    
    print('saving at {}'.format(out_file_path))
    save_array_as_geotif(out_path=out_file_path, array=pred_final, tile_info=tile_info,
                         out_type=args.out_type, dstnodata=args.nodata_value,
                         compress=compression, predictor=predictor)

    if std_final is not None:
        print('saving at {}'.format(out_file_path_std))
        save_array_as_geotif(out_path=out_file_path_std, array=std_final, tile_info=tile_info,
                             out_type=args.out_type, dstnodata=args.nodata_value,
                             compress=compression, predictor=predictor)

