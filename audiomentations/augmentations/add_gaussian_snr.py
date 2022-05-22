import random

import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    calculate_desired_noise_rms,
    calculate_rms,
)


class AddGaussianSNR(BaseWaveformTransform):
    """
    Add gaussian noise to the input. A random Signal to Noise Ratio (SNR) will be picked
    uniformly in the decibel scale. This aligns with human hearing, which is more
    logarithmic than linear.
    """

    supports_multichannel = True

    def __init__(
        self, min_snr_in_db=5, max_snr_in_db=40.0, p=0.5
    ):
        """
        :param min_snr_in_db: Minimum signal-to-noise ratio in db. A lower number means more noise.
        :param max_snr_in_db: Maximum signal-to-noise ratio in db. A greater number means less noise.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            # Pick SNR in decibel scale
            snr = random.uniform(self.min_snr_in_db, self.max_snr_in_db)

            clean_rms = calculate_rms(samples)
            noise_rms = calculate_desired_noise_rms(clean_rms=clean_rms, snr=snr)

            # In gaussian noise, the RMS gets roughly equal to the std
            self.parameters["noise_std"] = noise_rms

    def apply(self, samples, sample_rate):
        noise = np.random.normal(
            0.0, self.parameters["noise_std"], size=samples.shape
        ).astype(np.float32)
        return samples + noise
