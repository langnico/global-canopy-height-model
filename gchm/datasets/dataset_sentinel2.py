import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import tables
import numpy as np
import glob

from gchm.utils.loss import get_inverse_bin_frequency_weights


def compute_cloud_free(clouds, cloud_thresh_perc=10):
    num_pixels_patch = clouds.shape[1] * clouds.shape[2]
    perc_cloudy_pixels = np.sum(clouds > cloud_thresh_perc, axis=(1, 2, 3))/num_pixels_patch *100
    cloud_free = perc_cloudy_pixels < cloud_thresh_perc
    return cloud_free


class Sentinel2PatchesH5(Dataset):
    """ Custom Dataset class for loading cropped image patches stored as pytables in h5 files.

    Args:
        path_h5 (str): Path to h5 file containing cropped image patches.
        input_transforms (callable): Optional transform to be applied on the sample input
        target_transforms (callable): Optional transform to be applied on the sample target mean
        target_var_transforms (callable): Optional transform to be applied on the sample target variance (if available)
        input_lat_lon (bool): Option to use cyclic encoded latitude and longitude as additional input channels.
        mask_with_scl (bool): Option to mask the predictions using the Sentinel-2 L2A scene classification.
        use_cloud_free (bool): Option to get cloud free pixels (not needed if patches are previously filtered in h5 file).
        path_bin_weights (str):
        weight_key (str): Key in the dict returned from this dataset class that is used to weight the loss
    """
    def __init__(self, path_h5, input_transforms=None, target_transforms=None, target_var_transforms=None,
                 input_lat_lon=False, mask_with_scl=True, use_cloud_free=False,
                 path_bin_weights=None, weight_key=None):

        self.path_h5 = path_h5
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms
        self.target_var_transforms = target_var_transforms
        self.input_lat_lon = input_lat_lon
        self.mask_with_scl = mask_with_scl
        self.scl_zero_canopy_height = np.array([5, 6])  # "not vegetated", "water"
        self.use_cloud_free = use_cloud_free
        if self.use_cloud_free:
            self.cloud_free_indices = self.get_cloud_free_indices()
        self.path_bin_weights = path_bin_weights
        self.weight_key = weight_key
        if self.path_bin_weights is not None:
            self.label_distribution = np.load(self.path_bin_weights, allow_pickle=True).item()  # load dict with bin_edges and bin_weights

    def _open_hdf5(self):
        self.h5_file = tables.open_file(self.path_h5, mode='r')

    def _set_datasets(self):
        """ Could be used to set attributes in __getitem__"""
        self.images = self.h5_file.root.images
        self.lat = self.h5_file.root.lat
        self.lon = self.h5_file.root.lon
        self.labels_mean = self.h5_file.root['canopy_height']
        self.labels_var = self.h5_file.root['predictive_std']
        self.scl = self.h5_file.root['scl']

    def __getitem__(self, index):
        
        if isinstance(index, list):
            index = sorted(index)

        # open the h5 file in the first iteration --> each worker has its own connection
        if not hasattr(self, 'h5_file'):
            self._open_hdf5()

        if self.use_cloud_free:
            index = self.cloud_free_indices[index]

        images = np.array(self.h5_file.root.images[index, ...], dtype=np.float32)
        if self.input_lat_lon:
            lat = np.array(self.h5_file.root.lat[index, ...], dtype=np.float32)  # degrees
            lon = np.array(self.h5_file.root.lon[index, ...], dtype=np.float32)  # degrees
            lon_sin = np.sin(2 * np.pi * lon / 360)
            lon_cos = np.cos(2 * np.pi * lon / 360)
            inputs = np.concatenate((images, lat, lon_sin, lon_cos), axis=-1)  # channels last
        else:
            inputs = images

        labels_mean = np.array(self.h5_file.root['canopy_height'][index, ...], dtype=np.float32)
        # square the predictive std to get the predictive variance
        labels_var = np.square(np.array(self.h5_file.root['predictive_std'][index, ...], dtype=np.float32))

        if self.mask_with_scl:
            scl = np.array(self.h5_file.root['scl'][index, ...], dtype=np.uint8)
            # set not_vegetated and water class to zero canopy height
            mask_zero_height = np.logical_and(np.isin(scl, self.scl_zero_canopy_height), ~np.isnan(labels_mean))
            labels_mean[mask_zero_height] = 0  # Note: currently we use the original variance for 0 heights

        # get sample weights based on the inverse bin frequency (analog to class-balanced weighting)
        if self.path_bin_weights is not None:
            # Note: this needs to be done before normalizing the labels
            sample_weights = get_inverse_bin_frequency_weights(labels_mean,
                                                               bin_edges=self.label_distribution['bin_edges'],
                                                               bin_weights=self.label_distribution[self.weight_key])

        if self.input_transforms:
            inputs = self.input_transforms(inputs)

        if self.target_transforms:
            labels_mean = self.target_transforms(labels_mean)
            labels_var = self.target_var_transforms(labels_var)

        data_dict = {'inputs': inputs,
                     'labels_mean': labels_mean,
                     'labels_var': labels_var,
                     'labels_inv_var': 1/labels_var}

        if self.path_bin_weights is not None:
            data_dict[self.weight_key] = sample_weights

        for k in data_dict.keys():
            # move channels axis as pytorch expects the shape: [batch_size, channels, height, width].
            data_dict[k] = np.moveaxis(data_dict[k], source=-1, destination=-3)
            # convert all numpy arrays to tensor
            data_dict[k] = torch.from_numpy(data_dict[k])

        return data_dict

    def __len__(self):
        if self.use_cloud_free:
            return len(self.cloud_free_indices)
        else:
            with tables.open_file(self.path_h5, mode='r') as f:
                return len(f.root.images)

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    def get_cloud_free_indices(self):
        with tables.open_file(self.path_h5, mode='r') as f:
            clouds = f.root.cloud[:]
            cloud_free = compute_cloud_free(clouds)
            cloud_free_indices = np.argwhere(cloud_free)
            # convert to one-dimensional array
            if len(cloud_free_indices.shape) == 2:
                cloud_free_indices = cloud_free_indices.squeeze()
            # convert zero-dimensional array (scalar) to one-dimensional array
            if np.ndim(cloud_free_indices) == 0:
                cloud_free_indices = cloud_free_indices[None]
            return cloud_free_indices


def make_concat_dataset(paths_h5, input_transforms=None, target_transforms=None, target_var_transforms=None,
                        input_lat_lon=False, use_cloud_free=False, path_bin_weights=None, weight_key=None):
    """
    Returns a concatenated dataset of the custom pytorch :class:`Sentine2PatchesH5` for multiple h5 files.

    Args:
        paths_h5 (list): list of absolute paths of h5 files with sentinel-2 patch data
        input_transforms: transforms to process input images (normalization, augmentation, etc.)
        target_transforms: transforms to process targets
        target_var_transforms: transforms to process the variance of targets

    Returns:
        concatenated :class:`Sentine2PatchesH5`
    """
    # create a custom pytorch dataset for each h5 file
    datasets = []

    for path_h5 in paths_h5:
        datasets.append(Sentinel2PatchesH5(path_h5=path_h5,
                                           input_transforms=input_transforms,
                                           target_transforms=target_transforms,
                                           target_var_transforms=target_var_transforms,
                                           input_lat_lon=input_lat_lon,
                                           use_cloud_free=use_cloud_free,
                                           path_bin_weights=path_bin_weights,
                                           weight_key=weight_key))

    if len(datasets) == 1:
        # return the custom dataset to work with a list of batched indices in "sampler"
        return datasets[0]
    else:
        return ConcatDataset(datasets)


