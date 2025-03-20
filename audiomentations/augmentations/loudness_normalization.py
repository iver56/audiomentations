import random
import sys

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform


class LoudnessNormalization(BaseWaveformTransform):
    """
    Apply a constant amount of gain to match a specific loudness (in LUFS). This is an
    implementation of ITU-R BS.1770-4.

    For an explanation on LUFS, see https://en.wikipedia.org/wiki/LUFS

    See also the following web pages for more info on audio loudness normalization:
        https://github.com/csteinmetz1/pyloudnorm
        https://en.wikipedia.org/wiki/Audio_normalization

    Warning: This transform can return samples outside the [-1, 1] range, which may lead to
    clipping or wrap distortion, depending on what you do with the audio in a later stage.
    See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping
    """

    supports_multichannel = True

    def __init__(
        self,
        min_lufs: float = -31.0,
        max_lufs: float = -13.0,
        p: float = 0.5,
    ):
        super().__init__(p)

        if min_lufs > max_lufs:
            raise ValueError("min_lufs must not be greater than max_lufs")

        self.min_lufs = min_lufs
        self.max_lufs = max_lufs

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
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

        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            meter = pyloudnorm.Meter(sample_rate)  # create BS.1770 meter
            # Transpose because pyloudnorm expects shape like (smp, chn), not (chn, smp)
            self.parameters["loudness"] = float(
                meter.integrated_loudness(samples.transpose())
            )
            self.parameters["lufs"] = float(
                random.uniform(self.min_lufs, self.max_lufs)
            )

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
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

        # Guard against digital silence
        if self.parameters["loudness"] > float("-inf"):
            # Normalize loudness
            delta_loudness = self.parameters["lufs"] - self.parameters["loudness"]
            gain = np.power(10.0, delta_loudness / 20.0, dtype=np.float32)
            return gain * samples

        return samples
