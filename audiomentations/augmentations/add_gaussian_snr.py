import random
import warnings

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
        self,
        min_snr_in_db: float = None,
        max_snr_in_db: float = None,
        min_snr_db: float = None,
        max_snr_db: float = None,
        p: float = 0.5,
    ):
        """
        :param min_snr_in_db: Deprecated. Use min_snr_db instead.
        :param max_snr_in_db: Deprecated. Use max_snr_db instead.
        :param min_snr_db: Minimum signal-to-noise ratio in dB. A lower number means more noise.
        :param max_snr_db: Maximum signal-to-noise ratio in dB. A greater number means less noise.
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        if min_snr_db is not None and min_snr_in_db is not None:
            raise ValueError(
                "Passing both min_snr_db and min_snr_in_db is not supported. Use only"
                " min_snr_db."
            )
        elif min_snr_db is not None:
            self.min_snr_db = min_snr_db
        elif min_snr_in_db is not None:
            warnings.warn(
                "The min_snr_in_db parameter is deprecated. Use min_snr_db instead.",
                DeprecationWarning,
            )
            self.min_snr_db = min_snr_in_db
        else:
            self.min_snr_db = 5.0  # the default

        if max_snr_db is not None and max_snr_in_db is not None:
            raise ValueError(
                "Passing both max_snr_db and max_snr_in_db is not supported. Use only"
                " max_snr_db."
            )
        elif max_snr_db is not None:
            self.max_snr_db = max_snr_db
        elif max_snr_in_db is not None:
            warnings.warn(
                "The max_snr_in_db parameter is deprecated. Use max_snr_db instead.",
                DeprecationWarning,
            )
            self.max_snr_db = max_snr_in_db
        else:
            self.max_snr_db = 40.0  # the default

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            # Pick SNR in decibel scale
            snr = random.uniform(self.min_snr_db, self.max_snr_db)

            clean_rms = calculate_rms(samples)
            noise_rms = calculate_desired_noise_rms(clean_rms=clean_rms, snr=snr)

            # In gaussian noise, the RMS gets roughly equal to the std
            self.parameters["noise_std"] = noise_rms

    def apply(self, samples: np.ndarray, sample_rate: int):
        noise = np.random.normal(
            0.0, self.parameters["noise_std"], size=samples.shape
        ).astype(np.float32)
        return samples + noise
