from torch.utils.data import Sampler
from typing import Iterator, List, Sequence


class SliceBatchSampler(Sampler):
    """ Wraps another sampler to yield a slice of a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        slice_step (int): slicing step to take every slice_step sample: slice(start_idx, stop_idx, slice_step)

    Example 1:
        # with random slice starts
        dl_train = DataLoader(dataset=self.ds_train, batch_size=None,
                              sampler=SliceBatchSampler(sampler=RandomSampler(self.ds_train),
                                                        batch_size=self.args.batch_size,
                                                        slice_step=self.args.slice_step,
                                                        num_samples=len(self.ds_train),
                                                        drop_last=False),
                              num_workers=self.args.num_workers, pin_memory=True)
    Example 2:
        # with sequential slice starts
        slice_start_indices = range(0, len(ds_test), batch_size)
        dl_test = DataLoader(dataset=ds_test, batch_size=None,
                             sampler=SliceBatchSampler(sampler=slice_start_indices,
                                                       batch_size=batch_size,
                                                       slice_step=self.args.slice_step,
                                                       num_samples=len(ds_test),
                                                       drop_last=False),
                             num_workers=self.args.num_workers, pin_memory=True)

    """

    def __init__(self, sampler: Sampler[int], batch_size: int, slice_step: int, num_samples: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))

        self.sampler = sampler
        self.batch_size = batch_size
        self.slice_step = slice_step
        self.num_samples = num_samples
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        for idx in self.sampler:
            start = idx
            stop = start + self.batch_size * self.slice_step
            index_slice = slice(start, stop, self.slice_step)
            if stop >= self.num_samples and self.drop_last:
                continue
            else:
                yield index_slice

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


class SubsetSequentialSampler(Sampler[int]):
    """Samples elements sequentially from a given list of indices.

    Args:
        indices (sequence): a sequence of indices
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self) -> int:
        return len(self.indices)

