from typing import Optional, Union, Sequence

import numpy as np
from numpy.typing import NDArray


class WeightedChoiceSampler:
    """
    Draw integer indices according to a (possibly weighted) discrete distribution.

    Parameters
    ----------
    weights : Sequence[float] | NDArray | None, default None
        Normalised or un-normalised, non-negative weights for each outcome.
        If *None*, a uniform distribution over `num_items` is used.
    num_items : int | None, default None
        Number of outcomes when `weights is None`. Ignored otherwise.

    Examples
    --------
    >>> # weighted
    >>> s = WeightedChoiceSampler(weights=[0.2, 0.8])
    >>> s.sample(size=5)
    array([1, 0, 1, 1, 1])

    >>> # uniformly among 10 choices
    >>> s = WeightedChoiceSampler(num_items=10)     # weights=None implied
    >>> s.sample(size=5)
    array([7, 2, 3, 0, 9])
    """

    def __init__(
        self,
        weights: Optional[Union[Sequence[float], NDArray]] = None,
        *,
        num_items: Optional[int] = None,
    ):
        if weights is None:
            if num_items is None or num_items <= 0:
                raise ValueError(
                    "If 'weights' is None you must supply a positive 'num_items'."
                )
            self._uniform = True
            self.num_items = int(num_items)
            self.cdf = None
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.ndim != 1:
                raise ValueError("weights must be one-dimensional")
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative")
            total = float(weights.sum())
            if total == 0:
                raise ValueError("Sum of weights must be > 0")

            weights /= total
            self.cdf = np.cumsum(weights)
            # Ensure the last element is exactly 1.0 due to potential floating point errors
            self.cdf[-1] = 1.0
            self._uniform = False
            self.num_items = weights.size

    def sample(self, size: int = 1) -> Union[NDArray[np.int32], NDArray[np.int64]]:
        """
        Draw `size` indices according to the stored weight distribution.
        """
        if size < 1:
            raise ValueError("size must be a positive int")

        if self._uniform:
            # O(1) per sample
            return np.random.randint(self.num_items, size=size)
        else:
            # O(log n) per sample
            random_vals = np.random.random(size)
            return np.searchsorted(self.cdf, random_vals, side="right")
