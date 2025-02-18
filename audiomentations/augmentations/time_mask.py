import random

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform


class TimeMask(BaseWaveformTransform):
    """
    Make a randomly chosen part of the audio silent.
    Inspired by https://arxiv.org/pdf/1904.08779.pdf
    """

    supports_multichannel = True

    def __init__(
        self,
        min_band_part: float = 0.0,
        max_band_part: float = 0.5,
        fade: bool = False,
        p: float = 0.5,
    ):
        """
        :param min_band_part: Minimum length of the silent part as a fraction of the
            total sound length. Must be between 0.0 and 1.0
        :param max_band_part: Maximum length of the silent part as a fraction of the
            total sound length. Must be between 0.0 and 1.0
        :param fade: When set to True, a linear fade-in and fade-out is added to the silent part.
            This can smooth out unwanted abrupt changes between consecutive samples, which might
            otherwise sound like transients/clicks/pops.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        if min_band_part < 0.0 or min_band_part > 1.0:
            raise ValueError("min_band_part must be between 0.0 and 1.0")
        if max_band_part < 0.0 or max_band_part > 1.0:
            raise ValueError("max_band_part must be between 0.0 and 1.0")
        if min_band_part > max_band_part:
            raise ValueError("min_band_part must not be greater than max_band_part")
        self.min_band_part = min_band_part
        self.max_band_part = max_band_part
        self.fade = fade

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            num_samples = samples.shape[-1]
            self.parameters["t"] = random.randint(
                int(num_samples * self.min_band_part),
                int(num_samples * self.max_band_part),
            )
            self.parameters["t0"] = random.randint(
                0, num_samples - self.parameters["t"]
            )

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        new_samples = samples.copy()
        t = self.parameters["t"]
        t0 = self.parameters["t0"]
        mask = np.zeros(t)
        if self.fade:
            fade_length = min(int(sample_rate * 0.01), int(t * 0.1))
            if fade_length:
                mask[0:fade_length] = np.linspace(1, 0, num=fade_length)
                mask[-fade_length:] = np.linspace(0, 1, num=fade_length)
        new_samples[..., t0 : t0 + t] *= mask
        return new_samples
