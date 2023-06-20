import random
import warnings

import numpy as np
import sys

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
        min_lufs_in_db: float = None,
        max_lufs_in_db: float = None,
        min_lufs: float = None,
        max_lufs: float = None,
        p: float = 0.5,
    ):
        super().__init__(p)

        if min_lufs is not None and min_lufs_in_db is not None:
            raise ValueError(
                "Passing both min_lufs and min_lufs_in_db is not supported. Use only"
                " min_lufs."
            )
        elif min_lufs is not None:
            self.min_lufs = min_lufs
        elif min_lufs_in_db is not None:
            warnings.warn(
                "The min_lufs_in_db parameter is deprecated. Use min_lufs instead.",
                DeprecationWarning,
            )
            self.min_lufs = min_lufs_in_db
        else:
            self.min_lufs = -31.0  # the default

        if max_lufs is not None and max_lufs_in_db is not None:
            raise ValueError(
                "Passing both max_lufs and max_lufs_in_db is not supported. Use only"
                " max_lufs."
            )
        elif max_lufs is not None:
            self.max_lufs = max_lufs
        elif max_lufs_in_db is not None:
            warnings.warn(
                "The max_lufs_in_db parameter is deprecated. Use max_lufs instead.",
                DeprecationWarning,
            )
            self.max_lufs = max_lufs_in_db
        else:
            self.max_lufs = -13.0  # the default

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
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
            # transpose because pyloudnorm expects shape like (smp, chn), not (chn, smp)
            self.parameters["loudness"] = meter.integrated_loudness(samples.transpose())
            self.parameters["lufs"] = float(
                random.uniform(self.min_lufs, self.max_lufs)
            )

    def apply(self, samples: np.ndarray, sample_rate: int):
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
            # transpose because pyloudnorm expects shape like (smp, chn), not (chn, smp)
            return pyloudnorm.normalize.loudness(
                samples.transpose(),
                self.parameters["loudness"],
                self.parameters["lufs"],
            ).transpose()
        else:
            return samples
