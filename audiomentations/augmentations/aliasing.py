import random

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from audiomentations.core.transforms_interface import BaseWaveformTransform


class Aliasing(BaseWaveformTransform):
    """
    Apply an aliasing effect to the audio by downsampling to a lower 
    sample rate without filtering and upsampling after that.
    """

    supports_multichannel = True

    def __init__(self, min_sample_rate: int = 8000, max_sample_rate: int = 32000, p: float = 0.5):
        """
        :param min_sample_rate: The minimum sample rate used during an aliasing
        :param max_sample_rate: The maximum sample rate used during an aliasing
        :param p: The probability of applying this transform
        """
        super().__init__(p)
    
        if min_sample_rate > max_sample_rate:
            raise ValueError("min_sample_rate must not be larger than max_sample_rate")
        
        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["new_sample_rate"] = random.randint(
                self.min_sample_rate, self.max_sample_rate
            )

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        n = len(samples)
        dwn_n = round(n * float(self.parameters["new_sample_rate"]) / sample_rate)
        dwn_samples = signal.resample(samples, dwn_n)
        return signal.resample(dwn_samples, n)
