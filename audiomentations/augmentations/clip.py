import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform


class Clip(BaseWaveformTransform):
    """
    Clip audio by specified values. e.g. set a_min=-1.0 and a_max=1.0 to ensure that no
    samples in the audio exceed that extent. This can be relevant for avoiding integer
    overflow or underflow (which results in unintended wrap distortion that can sound
    horrible) when exporting to e.g. 16-bit PCM wav.

    Another way of ensuring that all values stay between -1.0 and 1.0 is to apply
    PeakNormalization.

    This transform is different from ClippingDistortion in that it takes fixed values
    for clipping instead of clipping a random percentile of the samples. Arguably, this
    transform is not very useful for data augmentation. Instead, think of it as a very
    cheap and harsh limiter (for samples that exceed the allotted extent) that can
    sometimes be useful at the end of a data augmentation pipeline.
    """

    supports_multichannel = True

    def __init__(self, a_min: float = -1.0, a_max: float = 1.0, p: float = 0.5):
        """
        :param a_min: float, minimum value for clipping
        :param a_max: float, maximum value for clipping
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert a_min < a_max
        self.a_min = a_min
        self.a_max = a_max

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        return np.clip(samples, self.a_min, self.a_max)
