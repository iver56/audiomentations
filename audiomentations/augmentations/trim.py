import librosa
import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform


class Trim(BaseWaveformTransform):
    """
    Trim leading and trailing silence from an audio signal using librosa.effects.trim
    """

    supports_multichannel = True

    def __init__(self, top_db: float = 30.0, p: float = 0.5):
        """
        :param top_db: The threshold (in Decibels) below reference to consider as silence
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.top_db = top_db

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        samples, _ = librosa.effects.trim(samples, top_db=self.top_db)
        return samples
