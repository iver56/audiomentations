import random

import librosa
import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform


class Resample(BaseWaveformTransform):
    """
    Resample signal using librosa.core.resample

    To do downsampling only, set both minimum and maximum sampling rate lower than
    original sampling rate. Conversely, to perform upsampling only, set both rates higher than the original sampling
    rate.
    """

    supports_multichannel = True

    def __init__(
        self, min_sample_rate: int = 8000, max_sample_rate: int = 44100, p: float = 0.5
    ):
        """
        :param min_sample_rate: The minimum sample rate
        :param max_sample_rate: The maximum sample rate
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_sample_rate <= max_sample_rate
        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["target_sample_rate"] = random.randint(
                self.min_sample_rate, self.max_sample_rate
            )

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        samples = librosa.core.resample(
            samples,
            orig_sr=sample_rate,
            target_sr=self.parameters["target_sample_rate"],
        )
        return samples
