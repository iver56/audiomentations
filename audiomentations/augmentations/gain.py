import random

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import convert_decibels_to_amplitude_ratio


class Gain(BaseWaveformTransform):
    """
    Multiply the audio by a random amplitude factor to reduce or increase the volume. This
    technique can help a model become somewhat invariant to the overall gain of the input audio.

    Warning: This transform can return samples outside the [-1, 1] range, which may lead to
    clipping or wrap distortion, depending on what you do with the audio in a later stage.
    See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping
    """

    supports_multichannel = True

    def __init__(
        self,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        p: float = 0.5,
    ):
        """
        :param min_gain_db: Minimum gain
        :param max_gain_db: Maximum gain
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        if min_gain_db > max_gain_db:
            raise ValueError("min_gain_db cannot be greater than max_gain_db")
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["amplitude_ratio"] = convert_decibels_to_amplitude_ratio(
                random.uniform(self.min_gain_db, self.max_gain_db)
            )

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        return samples * self.parameters["amplitude_ratio"]
