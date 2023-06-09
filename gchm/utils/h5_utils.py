import numpy as np
import tables
from tqdm import tqdm


def load_paths_from_dicretory(dir):
    file_names = glob.glob(dir + '/*.h5')
    paths_h5 = [os.path.abspath(f) for f in file_names]
    return paths_h5


def filter_paths_by_tile_names(paths, tile_names):
    valid_paths = []
    for p in paths:
        for tile_name in tile_names:
            if tile_name in p:
                valid_paths.append(p)
    return valid_paths


def init_hdf5_file(hdf5_path, patch_size, channels, projection, geotransform, gt_attributes, expectedrows=1000,
                   complib=None, complevel=0, subgroups=None, num_samples_chunk=64, bitshuffle=False, shuffle=True):

    def init_arrays(group):
        # init the storage:
        hdf5_file.create_earray(group, 'shot_number', tables.UInt64Atom(), shape=(0,), chunkshape=(num_samples_chunk,),
                                expectedrows=expectedrows)
        hdf5_file.create_earray(group, 'images', tables.UInt16Atom(), shape=img_shape, chunkshape=img_chunk_shape,
                                expectedrows=expectedrows, filters=filters)
        hdf5_file.create_earray(group, 'cloud', tables.UInt8Atom(), shape=label_shape, chunkshape=label_chunk_shape,
                                expectedrows=expectedrows, filters=filters)
        hdf5_file.create_earray(group, 'scl', tables.UInt8Atom(), shape=label_shape, chunkshape=label_chunk_shape,
                                expectedrows=expectedrows, filters=filters)

        for attribute in gt_attributes:
            hdf5_file.create_earray(group, attribute, tables.Float32Atom(), shape=label_shape,
                                    chunkshape=label_chunk_shape, expectedrows=expectedrows, filters=filters)

        # save patch top-left pixel location with respect to original tile.
        hdf5_file.create_earray(group, 'x_topleft', tables.UInt32Atom(), shape=(0,), chunkshape=(num_samples_chunk,),
                                expectedrows=expectedrows)
        hdf5_file.create_earray(group, 'y_topleft', tables.UInt32Atom(), shape=(0,), chunkshape=(num_samples_chunk,),
                                expectedrows=expectedrows)

        hdf5_file.create_earray(group, 'lat', tables.Float32Atom(), shape=label_shape, chunkshape=label_chunk_shape,
                                expectedrows=expectedrows, filters=filters)
        hdf5_file.create_earray(group, 'lon', tables.Float32Atom(), shape=label_shape, chunkshape=label_chunk_shape,
                                expectedrows=expectedrows, filters=filters)

        # save georeference for the original sentinel2 tile
        hdf5_file.create_group(group, 'georef', 'Georeference of the Sentinel-2 tile')

        if projection is not None:
            group.georef._v_attrs.projection = projection
        if geotransform is not None:
            group.georef._v_attrs.geotransform = geotransform

        # save image name from scihub 'title' (see products_df.title in crawl_patches)
        hdf5_file.create_earray(group, 'image_name', tables.StringAtom(itemsize=60), shape=(0,), chunkshape=(num_samples_chunk,),
                                expectedrows=expectedrows)
        hdf5_file.create_earray(group, 'image_date', tables.StringAtom(itemsize=10), shape=(0,), chunkshape=(num_samples_chunk,),
                                expectedrows=expectedrows)

    img_shape = (0, patch_size, patch_size, channels)
    label_shape = (0, patch_size, patch_size, 1)

    # most efficient for reading single patches
    img_chunk_shape = (num_samples_chunk, patch_size, patch_size, channels)
    label_chunk_shape = (num_samples_chunk, patch_size, patch_size, 1)

    # filter: compression complib can be: None, 'zlib', 'lzo', 'blosc'
    if complib:
        filters = tables.Filters(complevel=complevel, complib=complib, bitshuffle=bitshuffle, shuffle=shuffle)
    else:
        filters = tables.Filters(complevel=0)

    # open a hdf5 file and create earrays
    with tables.open_file(hdf5_path, mode='w') as hdf5_file:
        if subgroups is None:
            # init arrays in root group
            init_arrays(group=hdf5_file.root)
        else:
            # init arrays in subgroups (e.g. used for two-step pile shuffling)
            for subgroup in subgroups:
                hdf5_file.create_group(hdf5_file.root, subgroup, 'Pile')
                init_arrays(group=hdf5_file.root[subgroup])

    hdf5_file.close()


def write_patches_to_hdf(hdf5_path, band_arrays, image_date, image_name, latlon_patches=None, label_patches_dict=None):
    # the hdf5 file must already exist.
    with tables.open_file(hdf5_path, mode='r+') as hdf5_file:
        images_storage = hdf5_file.root.images
        lat_storage = hdf5_file.root.lat
        lon_storage = hdf5_file.root.lon
        shot_number_storage = hdf5_file.root.shot_number
        image_names_storage = hdf5_file.root.image_name
        image_dates_storage = hdf5_file.root.image_date
        x_topleft_storage = hdf5_file.root.x_topleft
        y_topleft_storage = hdf5_file.root.y_topleft
        cloud_storage = hdf5_file.root.cloud
        scl_storage = hdf5_file.root.scl

        print('images_storage before:', type(images_storage), images_storage.dtype, images_storage.shape)
        print('lat_storage before:', type(lat_storage), lat_storage.dtype, lat_storage.shape)
        print('lon_storage before:', type(lon_storage), lon_storage.dtype, lon_storage.shape)
        if label_patches_dict is not None:
            for attribute in label_patches_dict.keys():
                print('{}_storage before:'.format(attribute), type(hdf5_file.root[attribute]),
                      hdf5_file.root[attribute].dtype, hdf5_file.root[attribute].shape)

        count_skipped = 0
        for p_id in band_arrays:
            # Note: the shape of img, label must be 4-dim (1, patch, patch, channels)
            img_patch = np.expand_dims(sort_band_arrays(band_arrays=band_arrays[p_id]), axis=0)

            #print('DEBUG: img_patch.shape', img_patch.shape)
            #print('DEBUG: p_id:', p_id, 'x_topleft: {}, y_topleft: {}'.format(band_arrays[p_id]['x_topleft'], band_arrays[p_id]['y_topleft']))
            
            # skip the image patch if it contains any empty pixels (all bands equal zero)
            band_sum = np.sum(img_patch, axis=2)
            if (band_sum == 0).any():
                count_skipped += 1
                continue

            images_storage.append(img_patch)
            shot_number_storage.append(np.array([p_id]))

            cloud_patch = band_arrays[p_id]['CLD'][None, ..., None]  # expand first and last dimension
            cloud_storage.append(cloud_patch)

            scl_patch = band_arrays[p_id]['SCL'][None, ..., None]  # expand first and last dimension
            scl_storage.append(scl_patch)

            x_topleft_storage.append(np.array([band_arrays[p_id]['x_topleft']]))
            y_topleft_storage.append(np.array([band_arrays[p_id]['y_topleft']]))

            if label_patches_dict is not None:
                for attribute in label_patches_dict.keys():
                    label_patch = np.expand_dims(np.expand_dims(label_patches_dict[attribute][p_id], axis=-1), axis=0)
                    hdf5_file.root[attribute].append(label_patch)

            if latlon_patches is not None:
                lat_patch = np.expand_dims(np.expand_dims(latlon_patches[p_id]['lat'], axis=-1), axis=0)
                lon_patch = np.expand_dims(np.expand_dims(latlon_patches[p_id]['lon'], axis=-1), axis=0)
                lat_storage.append(lat_patch)
                lon_storage.append(lon_patch)

        image_dates_storage.append(np.array([image_date]))
        image_names_storage.append(np.array([image_name]))

        print('images_storage after:', type(images_storage), images_storage.dtype, images_storage.shape)
        print('lat_storage after:', type(lat_storage), lat_storage.dtype, lat_storage.shape)
        print('lon_storage after:', type(lon_storage), lon_storage.dtype, lon_storage.shape)
        if label_patches_dict is not None:
            for attribute in label_patches_dict.keys():
                print('{}_storage after:'.format(attribute), type(hdf5_file.root[attribute]),
                      hdf5_file.root[attribute].dtype, hdf5_file.root[attribute].shape)

        print('number of skipped patches: ', count_skipped)
    hdf5_file.close()


def merge_h5_datasets(paths_h5_files, out_h5_path,
                      gt_attributes=('canopy_height', 'predictive_std'),
                      patch_size=15, channels=12,
                      ignore_datasets=('image_date', 'image_name'),
                      max_num_samples_per_tile=None,
                      expectedrows=1000,
                      complib=None, complevel=0,
                      subgroups=None, num_samples_chunk=64, bitshuffle=False, shuffle=True):
    """
    Merge a list of h5 files with same f.root structure.
    E.g. merge all per tile h5 files used for training to a single h5 file.
    """

    init_hdf5_file(hdf5_path=out_h5_path, patch_size=patch_size, channels=channels,
                   projection=None, geotransform=None, gt_attributes=gt_attributes,
                   expectedrows=expectedrows, complib=complib, complevel=complevel,
                   subgroups=subgroups, num_samples_chunk=num_samples_chunk, bitshuffle=bitshuffle, shuffle=shuffle)

    with tables.open_file(out_h5_path, mode='r+') as f_out:
        dataset_names = list(f_out.root._v_leaves.keys())
        dataset_names = [n for n in dataset_names if n not in ignore_datasets]

        for step, h5_path in enumerate(tqdm(paths_h5_files, ncols=100, desc='merge')):
            with tables.open_file(h5_path, mode='r') as f_in:
                num_samples = len(f_in.root.images)
                if max_num_samples_per_tile is not None and (num_samples > max_num_samples_per_tile):
                    np.random.seed(1)
                    select_indices = np.random.permutation(num_samples)[:max_num_samples_per_tile]
                    for name in dataset_names:
                        data = f_in.root[name][:]
                        data = data[select_indices, ...]
                        f_out.root[name].append(data)
                else:
                    for name in dataset_names:
                        data = f_in.root[name][:]
                        f_out.root[name].append(data)

