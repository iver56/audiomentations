import random
import warnings

import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    convert_decibels_to_amplitude_ratio,
)


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
        min_gain_in_db: float = None,
        max_gain_in_db: float = None,
        min_gain_db: float = None,
        max_gain_db: float = None,
        p: float = 0.5,
    ):
        """
        :param min_gain_in_db: Deprecated. Use min_gain_db instead
        :param max_gain_in_db: Deprecated. Use max_gain_db instead
        :param min_gain_db: Minimum gain
        :param max_gain_db: Maximum gain
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        if min_gain_db is not None and min_gain_in_db is not None:
            raise ValueError(
                "Passing both min_gain_db and min_gain_in_db is not supported. Use only"
                " min_gain_db."
            )
        elif min_gain_db is not None:
            self.min_gain_db = min_gain_db
        elif min_gain_in_db is not None:
            warnings.warn(
                "The min_gain_in_db parameter is deprecated. Use min_gain_db instead.",
                DeprecationWarning,
            )
            self.min_gain_db = min_gain_in_db
        else:
            self.min_gain_db = -12.0  # the default

        if max_gain_db is not None and max_gain_in_db is not None:
            raise ValueError(
                "Passing both max_gain_db and max_gain_in_db is not supported. Use only"
                " max_gain_db."
            )
        elif max_gain_db is not None:
            self.max_gain_db = max_gain_db
        elif max_gain_in_db is not None:
            warnings.warn(
                "The max_gain_in_db parameter is deprecated. Use max_gain_db instead.",
                DeprecationWarning,
            )
            self.max_gain_db = max_gain_in_db
        else:
            self.max_gain_db = 12.0  # the default

        assert self.min_gain_db <= self.max_gain_db

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["amplitude_ratio"] = convert_decibels_to_amplitude_ratio(
                random.uniform(self.min_gain_db, self.max_gain_db)
            )

    def apply(self, samples: np.ndarray, sample_rate: int):
        return samples * self.parameters["amplitude_ratio"]
