from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray


class WeightedChoiceSampler:
    """
    A class for sampling from a weighted distribution using the cumulative distribution function (CDF).

    Usage:
    >>> sampler = WeightedChoiceSampler(weights=[0.2, 0.8])
    >>> samples = sampler.sample(size=10)
    """

    def __init__(self, weights: Optional[Union[List[float], NDArray]] = None):
        assert weights is not None and len(weights) > 0, "Weights must be provided"

        weights = np.asarray(weights, dtype=float)
        if np.any(weights < 0):
            raise AssertionError("Weights must be non-negative")
        total = float(weights.sum())
        if total == 0:
            raise AssertionError("Weights must be non-negative")

        weights /= total
        self.num_transforms = weights.size

        self.cdf = np.cumsum(weights)
        # Ensure the last element is exactly 1.0 due to potential floating point errors
        self.cdf[-1] = 1.0

    def sample(self, size: int = 1) -> np.ndarray:
        """
        Draw `size` indices according to the stored weight distribution.
        """
        random_vals = np.random.rand(size)
        return np.searchsorted(self.cdf, random_vals, side="right")
