import random

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform


class BitCrush(BaseWaveformTransform):
    """
    Apply a bit crush effect to the audio by reducing the bit depth. In other words, it
    reduces the number of bits that can be used for representing each audio sample.
    This adds quantization noise, and affects dynamic range. This transform does not
    apply dithering.
    """

    supports_multichannel = True

    def __init__(self, min_bit_depth: int = 5, max_bit_depth: int = 10, p: float = 0.5):
        """
        :param min_bit_depth: The minimum bit depth the audio will be "converted" to
        :param max_bit_depth: The maximum bit depth the audio will be "converted" to
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_bit_depth = min_bit_depth
        self.max_bit_depth = max_bit_depth

        if min_bit_depth < 1:
            raise ValueError("min_bit_depth must be at least 1")

        if max_bit_depth > 32:
            raise ValueError("max_bit_depth must not be greater than 32")

        if min_bit_depth > max_bit_depth:
            raise ValueError("min_bit_depth must not be larger than max_bit_depth")

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["bit_depth"] = random.randint(
                self.min_bit_depth, self.max_bit_depth
            )

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        q = (2 ** self.parameters["bit_depth"] / 2) + 1
        return np.round(samples * q) / q
