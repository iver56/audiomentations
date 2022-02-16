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
    Add gaussian noise to the samples with random Signal to Noise Ratio (SNR).

    Note that old versions of audiomentations (0.16.0 and below) used parameters
    min_SNR and max_SNR, which had inverse (wrong) characteristics. The use of these
    parameters is discouraged, and one should use min_snr_in_db and max_snr_in_db
    instead now.

    Note also that if you use the new parameters, a random SNR will be picked uniformly
    in the decibel scale instead of a uniform amplitude ratio. This aligns
    with human hearing, which is more logarithmic than linear.
    """

    supports_multichannel = True

    def __init__(
        self, min_SNR=None, max_SNR=None, min_snr_in_db=None, max_snr_in_db=None, p=0.5
    ):
        """
        :param min_SNR: Minimum signal-to-noise ratio (legacy). A lower number means less noise.
        :param max_SNR: Maximum signal-to-noise ratio (legacy). A greater number means more noise.
        :param min_snr_in_db: Minimum signal-to-noise ratio in db. A lower number means more noise.
        :param max_snr_in_db: Maximum signal-to-noise ratio in db. A greater number means less noise.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if min_snr_in_db is None and max_snr_in_db is None:
            # Apply legacy defaults
            if min_SNR is None:
                min_SNR = 0.001
            if max_SNR is None:
                max_SNR = 1.0
        else:
            if min_SNR is not None or max_SNR is not None:
                raise Exception(
                    "Error regarding AddGaussianSNR: Set min_snr_in_db"
                    " and max_snr_in_db to None to keep using min_SNR and"
                    " max_SNR parameters (legacy) instead. We highly recommend to use"
                    " min_snr_in_db and max_snr_in_db parameters instead. To migrate"
                    " from legacy parameters to new parameters,"
                    " use the following conversion formulas: \n"
                    "min_snr_in_db = -20 * math.log10(max_SNR)\n"
                    "max_snr_in_db = -20 * math.log10(min_SNR)"
                )
        self.min_SNR = min_SNR
        self.max_SNR = max_SNR

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:

            if self.min_SNR is not None and self.max_SNR is not None:
                if self.min_snr_in_db is not None and self.max_snr_in_db is not None:
                    raise Exception(
                        "Error regarding AddGaussianSNR: Set min_snr_in_db"
                        " and max_snr_in_db to None to keep using min_SNR and"
                        " max_SNR parameters (legacy) instead. We highly recommend to use"
                        " min_snr_in_db and max_snr_in_db parameters instead. To migrate"
                        " from legacy parameters to new parameters,"
                        " use the following conversion formulas: \n"
                        "min_snr_in_db = -20 * math.log10(max_SNR)\n"
                        "max_snr_in_db = -20 * math.log10(min_SNR)"
                    )
                else:
                    warnings.warn(
                        "You use legacy min_SNR and max_SNR parameters in AddGaussianSNR."
                        " We highly recommend to use min_snr_in_db and max_snr_in_db parameters instead."
                        " To migrate from legacy parameters to new parameters,"
                        " use the following conversion formulas: \n"
                        "min_snr_in_db = -20 * math.log10(max_SNR)\n"
                        "max_snr_in_db = -20 * math.log10(min_SNR)"
                    )
                    min_snr = self.min_SNR
                    max_snr = self.max_SNR
                    std = np.std(samples)
                    self.parameters["noise_std"] = random.uniform(
                        min_snr * std, max_snr * std
                    )
            else:
                # Pick snr in decibel scale
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
