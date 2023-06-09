import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import math

from gchm.utils.gdal_process import read_sentinel2_bands, create_latlon_mask, get_reference_band_ds_gdal


class Sentinel2Deploy(Dataset):
    """
    A custom Dataset to predict for a full Sentinel-2 image tile in the SAFE format.
    The Sentinel-2 tile is copped into slightly overlapping patches to apply the model.
    The resulting patches can then be recomposed to the original tile using recompose_patches().

    Args:
        path (str): Path of .zip file in SAFE format (or aws path to image data)
        input_transforms (callable): Optional transform to be applied on the sample input.
        input_lat_lon (bool): Option to use cyclic encoded lat lon as additional input channels.
        patch_size (int): Size of square patch of size
        border (int): Cropped patches will overlap by the amount of pixel set as border.
        from_aws (bool): Option to download the Sentinel-2 images from AWS S3.
    """
    def __init__(self, path, input_transforms=None, input_lat_lon=False, patch_size=128, border=8, from_aws=False):

        self.path = path
        self.from_aws = from_aws
        self.input_transforms = input_transforms
        self.input_lat_lon = input_lat_lon
        self.patch_size = patch_size
        self.border = border
        self.patch_size_no_border = self.patch_size - 2 * self.border
        self.image, self.tile_info, self.scl, self.cloud = read_sentinel2_bands(data_path=self.path, from_aws=self.from_aws, channels_last=True)
        self.image_shape_original = self.image.shape
        # pad the image with channels in last dimension
        self.image = np.pad(self.image, ((self.border, self.border), (self.border, self.border), (0, 0)), mode='symmetric')
        self.patch_coords_dict = self._get_patch_coords()
        self.scl_zero_canopy_height = np.array([5, 6])  # "not vegetated", "water"
        self.scl_exclude_labels = np.array([8, 9, 11, 6])  # CLOUD_MEDIUM_PROBABILITY, CLOUD_HIGH_PROBABILITY, SNOW, water
        self.scl = np.array(self.scl, dtype=np.uint8)
        # open a 10m reference band as gdal dataset
        self.ref_ds = get_reference_band_ds_gdal(path_file=self.path)
        # creat lat lon masks for entire images (10m resolution)
        self.lat_mask, self.lon_mask = create_latlon_mask(height=self.ref_ds.RasterYSize, width=self.ref_ds.RasterXSize,
                                                          refDataset=self.ref_ds)
        # pad lat lon mask to match the padded image
        self.lat_mask = np.pad(self.lat_mask, ((self.border, self.border), (self.border, self.border)), mode='symmetric')
        self.lon_mask = np.pad(self.lon_mask, ((self.border, self.border), (self.border, self.border)), mode='symmetric')

        print('self.image_shape_original: ', self.image_shape_original)
        print('after padding: self.image.shape: ', self.image.shape)
        print('after padding: self.lat_mask.shape: ', self.lat_mask.shape)
        print('after padding: self.lon_mask.shape: ', self.lon_mask.shape)

    def _get_patch_coords(self):
        img_rows, img_cols = self.image.shape[0:2]  # last dimension corresponds to channels

        print('img_rows, img_cols:', img_rows, img_cols)

        rows_tiles = int(math.ceil(img_rows / self.patch_size_no_border))
        cols_tiles = int(math.ceil(img_cols / self.patch_size_no_border))

        patch_coords_dict = {}
        patch_idx = 0
        for y in range(0, rows_tiles):
            y_coord = y * self.patch_size_no_border
            if y_coord > img_rows - self.patch_size:
                # move last patch up if it would exceed the image bottom
                y_coord = img_rows - self.patch_size
            for x in range(0, cols_tiles):
                x_coord = x * self.patch_size_no_border
                if x_coord > img_cols - self.patch_size:
                    # move last patch left if it would exceed the image right border
                    x_coord = img_cols - self.patch_size
                patch_coords_dict[patch_idx] = {'x_topleft': x_coord,
                                                'y_topleft': y_coord}
                patch_idx += 1

        print('number of patches: ', len(patch_coords_dict))
        return patch_coords_dict

    def __getitem__(self, index):

        y_topleft = self.patch_coords_dict[index]['y_topleft']
        x_topleft = self.patch_coords_dict[index]['x_topleft']

        patch = self.image[y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size, :]
        # cast to float32
        patch = patch.astype(np.float32)

        if self.input_lat_lon:
            lat = self.lat_mask[y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size][..., None]
            lon = self.lon_mask[y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size][..., None]
            lon_sin = np.sin(2 * np.pi * lon / 360)
            lon_cos = np.cos(2 * np.pi * lon / 360)
            inputs = np.concatenate((patch, lat, lon_sin, lon_cos), axis=-1)  # channels last
        else:
            inputs = patch

        if self.input_transforms:
            inputs = self.input_transforms(inputs)

        data_dict = {'inputs': inputs}

        for k in data_dict.keys():
            # move channels axis as pytorch expects the shape: [batch_size, channels, height, width].
            data_dict[k] = np.transpose(data_dict[k], axes=[2, 0, 1])
            # convert all numpy arrays to tensor
            data_dict[k] = torch.from_numpy(data_dict[k])

        return data_dict

    def __len__(self):
        return len(self.patch_coords_dict)

    def recompose_patches(self, patches, out_type=np.float32,
                          mask_empty=True, mask_negative=True,
                          mask_clouds=True, mask_with_scl=True, cloud_thresh_perc=5,
                          mask_tile_boundary=False):
        """ Recompose image patches or corresponding predictions to the full Sentinel-2 tile shape."""

        # init tile with channels first
        channels = patches.shape[1]
        height, width = self.image.shape[0:2]
        tile = np.full(shape=(channels, height, width), fill_value=np.nan, dtype=out_type)

        for index in range(len(patches)):
            y_topleft = self.patch_coords_dict[index]['y_topleft']
            x_topleft = self.patch_coords_dict[index]['x_topleft']

            tile[:, y_topleft+self.border:y_topleft + self.patch_size - self.border,
                 x_topleft+self.border:x_topleft + self.patch_size - self.border] \
                = patches[index, :,
                          self.border:self.patch_size - self.border,
                          self.border:self.patch_size - self.border]

        # remove padding to return original tile size
        tile = tile[:, self.border:-self.border, self.border:-self.border]

        # reduce first dimension if single band (e.g. predictions)
        tile = tile.squeeze()

        # masking
        tile_masked = tile
        if mask_empty:
            # pixels where all RGB values equal zero are empty (bands B02, B03, B04)
            # note self.image has shape: (height, width, channels)
            invalid_mask = np.sum(self.image[self.border:-self.border, self.border:-self.border, 1:4], axis=-1) == 0
            print('self.image.shape', self.image.shape)
            print('invalid_mask.shape', invalid_mask.shape)
            print('number of empty pixels:', np.sum(invalid_mask))
            # mask empty image pixels
            tile_masked[invalid_mask] = np.nan

        if mask_negative:
            # mask negative values in the recomposed tile (e.g. predictions)
            tile_masked[tile_masked < 0] = np.nan

        if mask_with_scl:
            # mask snow and cloud (medium and high density). In some cases the probability cloud mask might miss some clouds
            invalid_mask = np.logical_and(np.isin(self.scl, self.scl_exclude_labels), ~np.isnan(tile_masked))
            tile_masked[invalid_mask] = np.nan

            ## set not_vegetated and water class to zero canopy height
            #mask_zero_height = np.logical_and(np.isin(self.scl, self.scl_zero_canopy_height), ~np.isnan(tile_masked))
            #tile_masked[mask_zero_height] = 0

        if mask_clouds:
            tile_masked[self.cloud > cloud_thresh_perc] = np.nan

        if mask_tile_boundary:
            # top and bottom rows
            tile_masked[:self.border, :] = np.nan
            tile_masked[-self.border:, :] = np.nan
            # left and right columns
            tile_masked[:, :self.border] = np.nan
            tile_masked[:, -self.border:] = np.nan

        return tile_masked

