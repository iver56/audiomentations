import warnings

import librosa

from audiomentations.core.transforms_interface import BaseWaveformTransform


class Trim(BaseWaveformTransform):
    """
    Trim leading and trailing silence from an audio signal using librosa.effects.trim
    """

    supports_multichannel = True

    def __init__(self, top_db=20, p=None):
        if p is None:
            p = 1.0
            warnings.warn(
                "The default value of p in Trim will change from 1.0 to 0.5 in a future"
                " version of audiomentations. Please specify p explicitly to make your"
                " code more future-proof and to get rid of this warning."
            )

        super().__init__(p)
        self.top_db = top_db

    def apply(self, samples, sample_rate):
        samples, lens = librosa.effects.trim(samples, top_db=self.top_db)
        return samples
