import random

import librosa
import torch
import torchaudio
import numpy as np

from audiomentations.core.utils import find_time_shift
from audiomentations.core.transforms_interface import BaseWaveformTransform


class ApplyMP3Codec(BaseWaveformTransform):
    """
    Apply MP3 Codec. 
    Mp3 encode and decode the audio signal. May cause time shift issues.
    """

    supports_multichannel = True

    def __init__(self,
                 min_bitrate=8,
                 max_bitrate=320,
                 p=0.5):
        """
        :param min_bitrate, int, minimum bitrate (in `kbps`)
        :param max_bitrate, int, maximum bitrate (in `kbps`)
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        assert self.min_bitrate < self.max_bitrate

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters['bitrate'] = random.randint(
                self.min_bitrate, self.max_bitrate
            )

    def apply(self, samples, sample_rate):
        samples_torch = torch.tensor(samples.astype(np.float32))

        if len(samples.shape) == 1:
            samples_torch = samples_torch.unsqueeze(0)

        compressed_samples = torchaudio.functional.apply_codec(
            samples_torch,
            sample_rate, 
            format='mp3',
            compression=self.parameters['bitrate']
        )

        # the decoded audio may have more samples than the original due to mp3 codec characteristics.
        # to alight the decoded audio with original, first use convolution to find time shift.
        shift = find_time_shift(compressed_samples[0].numpy(), samples_torch[0].numpy())

        assert shift > 0
        compressed_samples = compressed_samples[:, shift:]
        assert compressed_samples.shape[-1] >= samples_torch.shape[-1]
        compressed_samples = compressed_samples[:, :samples_torch.shape[-1]]

        if len(samples.shape) == 1:
            compressed_samples = compressed_samples[0]

        compressed_samples = compressed_samples.numpy()

        assert compressed_samples.shape == samples.shape
        return compressed_samples