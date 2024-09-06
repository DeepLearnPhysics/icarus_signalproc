"""Used to define which dataset entries to load at each iteration"""

import time

import numpy as np

from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

__all__ = ['SequentialBatchSampler', 'RandomSequenceBatchSampler',
           'BootstrapBatchSampler']


class AbstractBatchSampler(Sampler):
    """Abstract sampler class.

    Samplers that inherit from this class should work out of the box.
    Just define the __len__ and __iter__ functions. __init__ defines
    self.num_samples and self.batch_size as well as a self._random
    RNG, if needed.
    """

    def __init__(self, dataset, batch_size, seed=None, drop_last=True):
        """Check and store the values passed to the initializer,
        set the seeds appropriately.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to sampler from
        batch_size : int
            Number of samples to load per iteration
        seed : int, optional
            Seed to use for random sampling
        drop_last: bool, default True
            If `True`, drop the last batch to make the number of entries a
            multiple of the batch_size (if needed)
        """
        # Initialize parent class
        # TODO: `data_source` id deprecated, will need to remove with newer
        # version of pytorch (2.2.0 and above)
        super().__init__(data_source=None)

        # Initialize the random number generator with a seed
        if seed is None:
            seed = int(time.time())
        else:
            assert isinstance(seed, int), (
                    f"The sampler seed must be an integer, got: {seed}.")

        self._random = np.random.RandomState(seed=seed) # pylint: disable=E1101

        # Check that the batch_size is a sensible value
        self.batch_size = batch_size
        if batch_size < 1:
            raise ValueError(
                    "The `batch_size` must be a positive non-zero integer.")

        # Process the number of samples
        self.num_samples = len(dataset)
        self.drop_last = drop_last
        if self.num_samples < 1:
            raise ValueError(
                    "The dataset must have at least on entry to sample from.")
        if drop_last:
            if self.num_samples < self.batch_size:
                raise ValueError(
                        "The dataset does not have enough samples "
                        f"({self.num_samples}) to produce a complete batch "
                        f"({batch_size}).")
            self.num_samples -= self.num_samples%self.batch_size

    def __len__(self):
        """Provides the full length of the sampler.

        The length of the sampler can differ from the number of elements in
        the underlying dataset, if the last batch is smaller than the requested
        size and is dropped.

        Returns
        -------
        int
            Total number of entries to sample
        """
        return self.num_samples

    def __iter__(self):
        """Placeholder to be overridden by children classes."""
        raise NotImplementedError


class SequentialBatchSampler(AbstractBatchSampler):
    """Samples batches sequentially within the dataset."""
    name = 'sequential'

    def __iter__(self):
        """Iterates over sequential batches of data."""
        order = np.arange(self.num_samples, dtype=int)
        return iter(order)


class RandomSequenceBatchSampler(AbstractBatchSampler):
    """Samples sequential batches randomly within the dataset."""
    name = 'random_sequence'

    def __iter__(self):
        """Iterates over sequential batches of data randomly located
        in the dataset.
        """
        # Pick a general offset and produce sequence starts with respect to it.
        # Introducing this random offset ensures that the data is not
        # systematically batched in the same way
        offset = self._random.randint(0, self.batch_size)
        starts = np.arange(
                offset, offset+self.num_samples, self.batch_size, dtype=int)

        # Randomly pick the starts
        self._random.shuffle(starts)

        # Produce a sequence for each start
        batches = [np.arange(
            start, start + self.batch_size,
            dtype=int) for start in starts]
        indices = np.concatenate(batches)

        # Wrap indexes around if the offset is non-zero
        indices = indices%self.num_samples

        return iter(indices)


class BootstrapBatchSampler(AbstractBatchSampler):
    """Sampler used for bootstrap sampling of the entire dataset.

    This is particularly useful for training an ensemble of networks
    (bagging).
    """
    name = 'bootstrap'

    def __iter__(self):
        """Iterates over bootstrapped batches of data randomly picked
        from the dataset.
        """
        max_id  = self.num_samples + 1 - self.batch_size
        starts = np.arange(
                0, max_id, self.batch_size, dtype=int)
        bootstrap_indices = np.random.choice(np.arange(
            self.num_samples), self.num_samples)
        batches = [bootstrap_indices[np.arange(
            start, start+self.batch_size,
            dtype=int)] for start in starts]

        return iter(np.concatenate(batches))
