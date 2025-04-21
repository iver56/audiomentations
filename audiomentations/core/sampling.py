import numpy as np


class WeightedChoiceSampler:
    """
    A class for sampling from a weighted distribution using the cumulative distribution function (CDF).

    Usage:
    >>> sampler = WeightedChoiceSampler(weights=[0.2, 0.8])
    >>> samples = sampler.sample(size=10)
    """
    def __init__(self, weights: list[float] = None):
        assert weights is not None and len(weights) > 0, "Weights must be provided"

        # ensure weights are normalized
        weights = np.array(weights)
        weights /= np.sum(weights)
        assert np.all(weights >= 0), "Weights must be non-negative"

        # initialize the sampler
        self.num_transforms = len(weights)
        self.cdf = np.cumsum(weights)
        # Ensure the last element is exactly 1.0 due to potential floating point errors
        self.cdf[-1] = 1.0

    # Modified sample method to take size
    def sample(self, size: int = 1) -> np.ndarray:
        random_vals = np.random.rand(size)
        # Use searchsorted with the array of random values
        return np.searchsorted(self.cdf, random_vals, side='right')
