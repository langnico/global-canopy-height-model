import os
import sys

from gchm.utils.h5_utils import merge_h5_datasets, load_paths_from_dicretory


if __name__ == "__main__":

    # set directory paths
    in_h5_dir_parts = sys.argv[1]
    out_h5_dir = sys.argv[2]

    num_samples_chunk = 64
    complib = 'blosc:zstd'  # blosc:lz4   blosc:zlib   blosc:zstd
    complevel = 5
    bitshuffle = 0
    shuffle = 1

    # total number of samples in merged file
    expectedrows_dict = {'train': 622311092,
                         'val':    69546665}

    # ---------------------------------

    print('out_h5_dir', out_h5_dir)
    if not os.path.exists(out_h5_dir):
        os.makedirs(out_h5_dir)

    # merge h5 files per split
    for split in ["val", "train"]:
        print("Merging files for split: {}".format(split))

        # get path to split subdirectory
        in_h5_dir_parts_split = os.path.join(in_h5_dir_parts, split)
        print("Loading h5 files from path: {}".format(in_h5_dir_parts_split))

        # load all h5 paths in subdirectory
        paths_h5_files = load_paths_from_dicretory(dir=in_h5_dir_parts_split)
        print("Number of h5 part files that are merged: {}".format(len(paths_h5_files)))

        # output path
        out_h5_path = os.path.join(out_h5_dir, "GLOBAL_GEDI_{}.h5".format(split))
        print("Writing to output h5 file: {}".format(out_h5_path))

        merge_h5_datasets(paths_h5_files=paths_h5_files,
                          out_h5_path=out_h5_path,
                          gt_attributes=('canopy_height', 'predictive_std'),
                          patch_size=15, channels=12,
                          ignore_datasets=('image_date', 'image_name'),
                          max_num_samples_per_tile=None,
                          expectedrows=expectedrows_dict[split],
                          complib=complib, complevel=complevel,
                          subgroups=None, num_samples_chunk=num_samples_chunk,
                          bitshuffle=bitshuffle, shuffle=shuffle)


