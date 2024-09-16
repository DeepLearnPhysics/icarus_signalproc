import numpy as np
from torcher.sampler import AbstractBatchSampler

class SGOverlayRandomSampler(AbstractBatchSampler):

    def __iter__(self):
        """Iterates over sequential batches of data."""
        self._dataset.shuffle()
        order = np.arange(len(self._dataset), dtype=int)
        return iter(order)

