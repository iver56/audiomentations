import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform


class Reverse(BaseWaveformTransform):
    """
    Reverse the audio. Also known as time inversion. Inversion of an audio track along its time
    axis relates to the random flip of an image, which is an augmentation technique that is
    widely used in the visual domain. This can be relevant in the context of audio
    classification. It was successfully applied in the paper
    AudioCLIP: Extending CLIP to Image, Text and Audio
    https://arxiv.org/pdf/2106.13043.pdf
    """

    supports_multichannel = True

    def __init__(self, p=0.5):
        """
        :param p: The probability of applying this transform
        """
        super().__init__(p)

    def apply(self, samples, sample_rate):
        if len(samples.shape) > 1:
            return np.fliplr(samples)
        else:
            return np.flipud(samples)
