import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import tables
from torch.utils.data import SubsetRandomSampler

from gchm.utils.sampler import SliceBatchSampler, SubsetSequentialSampler


def compute_train_mean_std(ds_train, data_key, num_samples=None, num_workers=8, batch_size=128, return_distribution=False, bin_edges=None):
    """
    Compute training statistics (mean and standard deviation) per tensor channel (with channel axis=1).
    Can be used for multi-channel inputs and single channel targets
    Args:
        ds_train: torch dataset
        data_key: string name of torch tensor with shape (channels, height, width). (e.g. 'inputs', 'labels_mean')
        num_samples: max number of patches used to calculate the statistics
        num_workers: number of workers used in dataloader
        return_distribution: bool, if True: returns the number of samples per bin
        bin_edges: used to return the number of sampler per bin

    Returns:
        train_mean: numpy array with shape (channels,)
        train_std: numpy array with shape (channels,)
    """
    max_range = 1.7e308  # max range for float64

    if num_samples is None:
        shuffle = False
    else:
        shuffle = True

    slice_start_indices = range(0, len(ds_train), batch_size)
    print(slice_start_indices)

    if shuffle:
        sampler_starts = SubsetRandomSampler(slice_start_indices)
        print('len(sampler_starts)', len(sampler_starts))
    else:
        sampler_starts = SubsetSequentialSampler(slice_start_indices)

    dl_train = DataLoader(dataset=ds_train, batch_size=None,
                          sampler=SliceBatchSampler(sampler=sampler_starts,
                                                    batch_size=batch_size,
                                                    slice_step=1,
                                                    num_samples=len(ds_train),
                                                    drop_last=False),
                          num_workers=num_workers)

    if not num_samples:
        num_samples = len(ds_train)

    with torch.no_grad():
        # get the dimensions from the first sample
        for batch_idx, batch in enumerate(dl_train):
            first_sample = batch[data_key]
            _, channels, height, width = first_sample.shape
            break
        print('channels, height, width: ', channels, height, width)
    
        # init torch tensors
        sum_ = torch.zeros(channels, dtype=torch.float64)
        sum_sq_ = torch.zeros(channels, dtype=torch.float64)
        count_ = torch.zeros(1, dtype=torch.float64)  # assume equal count for each channel (image or labels)

        if return_distribution:
            if bin_edges is None:
                bin_edges = np.arange(0, 101, 1)
            bin_num_samples = np.zeros(len(bin_edges)-1, dtype=np.float64)

        for batch_idx, batch in enumerate(tqdm(dl_train, ncols=100, desc='stats')):  # for each training step

            # stop when num_samples are sampled
            if batch_idx >= int(num_samples/batch_size):
                print('Loaded {} samples to compute channel statistics.'.format(num_samples))
                break

            sum_ = sum_ + torch.nansum(batch[data_key], axis=(0, 2, 3))
            sum_sq_ = sum_sq_ + torch.nansum((batch[data_key] ** 2), axis=(0, 2, 3))
            count_ = count_ + torch.sum(~torch.isnan(batch[data_key][:, 0, ...]))  # use the first channel because we want the count per channel to compute image channel statistics

            if return_distribution:
                hist, bin_edges = np.histogram(batch[data_key], bins=bin_edges)
                bin_num_samples = bin_num_samples + hist

            # check overflow
            if torch.any(sum_sq_ >= max_range):
                print('sum_sq_: ', sum_sq_)
                raise OverflowError('sum_sq_ is too large')

        train_mean = sum_ / count_
        train_mean_sq_ = sum_sq_ / count_
        train_std = torch.sqrt(train_mean_sq_ - train_mean ** 2)

        # cast to numpy float32
        train_mean = train_mean.cpu().numpy().astype(np.float32)
        train_std = train_std.cpu().numpy().astype(np.float32)

    if return_distribution:
        return train_mean, train_std, bin_num_samples, bin_edges
    else:
        return train_mean, train_std


def compute_mean_std_earray(h5_file, earray_name):
    """
    TODO: update docs
    Compute training statistics (mean and standard deviation) per tensor channel (with channel axis=1).
    Can be used for multi-channel inputs and single channel targets
    Args:
        ds_train: torch dataset
        data_key: string name of torch tensor with shape (channels, height, width). (e.g. 'inputs', 'labels_mean')
        num_samples: max number of patches used to calculate the statistics
        num_workers: number of workers used in dataloader

    Returns:
        train_mean: numpy array with shape (channels,)
        train_std: numpy array with shape (channels,)
    """
    max_range = 1.7e308  # max range for float64

    f = tables.open_file(h5_file, 'r')
    e = f.root[earray_name]  # note channel is last

    num_samples, height, width, channels = e.shape
    print('e.shape', e.shape)
    print('num_samples, height, width, channels: ', num_samples, height, width, channels)

    # init torch tensors
    sum_ = np.zeros(channels, dtype=np.float64)
    sum_sq_ = np.zeros(channels, dtype=np.float64)
    count_ = np.zeros(1, dtype=np.float64)  # assume equal count for each channel (image or labels)

    for i, patch in enumerate(tqdm(e, ncols=100, desc='stats')):
        patch = patch[None, ...].astype(np.float32)
        sum_ = sum_ + np.nansum(patch, axis=(0, 1, 2))
        sum_sq_ = sum_sq_ + np.nansum((patch ** 2), axis=(0, 1, 2))
        count_ = count_ + np.sum(~np.isnan(patch[..., 0]))  # use the first channel because we want the count per channel to compute image channel statistics

        # check overflow
        if np.any(sum_sq_ >= max_range):
            print('sum_sq_: ', sum_sq_)
            raise OverflowError('sum_sq_ is too large')

    train_mean = sum_ / count_
    train_mean_sq_ = sum_sq_ / count_
    train_std = np.sqrt(train_mean_sq_ - train_mean ** 2)

    # cast to numpy float32
    train_mean = train_mean.astype(np.float32)
    train_std = train_std.astype(np.float32)

    return train_mean, train_std
