import random

import sys

from audiomentations.core.transforms_interface import BaseWaveformTransform


class LoudnessNormalization(BaseWaveformTransform):
    """
    Apply a constant amount of gain to match a specific loudness. This is an implementation of
    ITU-R BS.1770-4.
    See also:
        https://github.com/csteinmetz1/pyloudnorm
        https://en.wikipedia.org/wiki/Audio_normalization

    Warning: This transform can return samples outside the [-1, 1] range, which may lead to
    clipping or wrap distortion, depending on what you do with the audio in a later stage.
    See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping
    """

    supports_multichannel = True

    def __init__(self, min_lufs_in_db=-31, max_lufs_in_db=-13, p=0.5):
        super().__init__(p)
        # For an explanation on LUFS, see https://en.wikipedia.org/wiki/LUFS
        self.min_lufs_in_db = min_lufs_in_db
        self.max_lufs_in_db = max_lufs_in_db

    def randomize_parameters(self, samples, sample_rate):
        try:
            import pyloudnorm
        except ImportError:
            print(
                "Failed to import pyloudnorm. Maybe it is not installed? "
                "To install the optional pyloudnorm dependency of audiomentations,"
                " do `pip install audiomentations[extras]` or simply "
                " `pip install pyloudnorm`",
                file=sys.stderr,
            )
            raise

        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            meter = pyloudnorm.Meter(sample_rate)  # create BS.1770 meter
            # transpose because pyloudnorm expects shape like (smp, chn), not (chn, smp)
            self.parameters["loudness"] = meter.integrated_loudness(samples.transpose())
            self.parameters["lufs_in_db"] = float(
                random.uniform(self.min_lufs_in_db, self.max_lufs_in_db)
            )

    def apply(self, samples, sample_rate):
        try:
            import pyloudnorm
        except ImportError:
            print(
                "Failed to import pyloudnorm. Maybe it is not installed? "
                "To install the optional pyloudnorm dependency of audiomentations,"
                " do `pip install audiomentations[extras]` or simply "
                " `pip install pyloudnorm`",
                file=sys.stderr,
            )
            raise

        # Guard against digital silence
        if self.parameters["loudness"] > float("-inf"):
            # transpose because pyloudnorm expects shape like (smp, chn), not (chn, smp)
            return pyloudnorm.normalize.loudness(
                samples.transpose(),
                self.parameters["loudness"],
                self.parameters["lufs_in_db"],
            ).transpose()
        else:
            return samples
