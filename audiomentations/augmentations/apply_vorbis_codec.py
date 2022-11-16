import random

import librosa
import numpy as np
import torch
import torchaudio

from audiomentations.core.transforms_interface import BaseWaveformTransform


class ApplyVorbisCodec(BaseWaveformTransform):
    """
    Apply OGG/Vorbis Codec. 
    OGG/Vorbis encode and decode the audio signal.
    """

    supports_multichannel = True

    def __init__(self,
                 min_compression=-1,
                 max_compression=10,
                 p=0.5):
        """
        :param min_compression, int, minimum compression. This corresponds to ``-C`` option of ``sox`` command.
        :param max_compression, int, maximum compression. This corresponds to ``-C`` option of ``sox`` command.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_compression = min_compression
        self.max_compression = max_compression
        assert self.min_compression < self.max_compression

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters['compression'] = random.randint(
                self.min_compression, self.max_compression
            )

    def apply(self, samples, sample_rate):
        samples_torch = torch.tensor(samples.astype(np.float32))

        if len(samples.shape) == 1:
            samples_torch = samples_torch.unsqueeze(0)

        compressed_samples = torchaudio.functional.apply_codec(
            samples_torch,
            sample_rate, 
            format='ogg',
            compression=self.parameters['compression']
        )

        if len(samples.shape) == 1:
            compressed_samples = compressed_samples[0]

        compressed_samples = compressed_samples.numpy()

        assert compressed_samples.shape == samples.shape
        return compressed_samples