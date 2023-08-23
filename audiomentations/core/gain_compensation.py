from typing import Callable

import numpy as np
import sys

from audiomentations.core.utils import (
    calculate_rms,
    convert_decibels_to_amplitude_ratio,
)


class GainCompensation:
    """
    Gain up or down the audio after the given transform (or set of transforms) has
    processed the audio. `GainCompensation` can be useful for compensating for any gain
    differences introduced by a (set of) transform(s), like `ApplyImpulseResponse`,
    `ApplyBackgroundNoise`, `Clip` and many others. `GainCompensation` ensures that the
    processed audio's RMS (Root Mean Square) or LUFS (Loudness Units Full Scale) matches
    the original.
    """

    def __init__(self, transform: Callable[[np.ndarray, int], np.ndarray], method: str):
        """
        :param transform: A callable to be applied. It should input
            samples (ndarray), sample_rate (int) and optionally some user-defined
            keyword arguments.
        :param method: "same_rms" or "same_lufs".
            * "same_rms" is fast to compute
            * "same_lufs" is slower, but is more aligned with human's perceptual loudness
        """
        self.transform = transform
        self.method = method
        if self.method not in ("same_rms", "same_lufs"):
            raise ValueError('method must be set to "same_rms" or "same_lufs"')

    def method_same_rms(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        rms_before = calculate_rms(samples)
        samples = self.transform(samples, sample_rate)
        rms_after = calculate_rms(samples)
        gain_factor = rms_before / rms_after
        samples *= gain_factor
        return samples

    def method_same_lufs(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            import pyloudnorm
        except ImportError:
            print(
                (
                    "Failed to import pyloudnorm. Maybe it is not installed? "
                    "To install the optional pyloudnorm dependency of audiomentations,"
                    " do `pip install audiomentations[extras]` or simply "
                    " `pip install pyloudnorm`"
                ),
                file=sys.stderr,
            )
            raise

        meter = pyloudnorm.Meter(sample_rate)  # create BS.1770 meter
        # transpose because pyloudnorm expects shape like (smp, chn), not (chn, smp)
        lufs_before = meter.integrated_loudness(samples.transpose())
        samples = self.transform(samples, sample_rate)
        lufs_after = meter.integrated_loudness(samples.transpose())
        gain_db = lufs_before - lufs_after
        samples *= convert_decibels_to_amplitude_ratio(gain_db)
        return samples

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.method == "same_rms":
            return self.method_same_rms(samples, sample_rate)
        elif self.method == "same_lufs":
            return self.method_same_lufs(samples, sample_rate)
        else:
            raise Exception()
