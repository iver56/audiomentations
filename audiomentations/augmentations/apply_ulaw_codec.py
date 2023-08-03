import random

import librosa
import numpy as np
import torch
import torchaudio

from audiomentations.core.transforms_interface import BaseWaveformTransform

    
class ApplyULawCodec(BaseWaveformTransform):
    """
    Apply MU-Law/U-Law Codec. 
    ULAW encode and decode the audio signal.
    """

    supports_multichannel = True

    def __init__(self,
                 p=0.5):
        """
        :param p: The probability of applying this transform
        """
        super().__init__(p)


    def apply(self, samples, sample_rate):
        samples_torch = torch.tensor(samples.astype(np.float32))

        if len(samples.shape) == 1:
            samples_torch = samples_torch.unsqueeze(0)

        compressed_samples = torchaudio.functional.apply_codec(
            samples_torch,
            sample_rate, 
            format='wav',
            encoding='ULAW',
            bits_per_sample=8
        )

        if len(samples.shape) == 1:
            compressed_samples = compressed_samples[0]

        compressed_samples = compressed_samples.numpy()

        assert compressed_samples.shape == samples.shape
        return compressed_samples