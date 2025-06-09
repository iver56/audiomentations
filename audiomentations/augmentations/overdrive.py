import random

import numpy as np
import torch
import torchaudio

from audiomentations.core.transforms_interface import BaseWaveformTransform


class Overdrive(BaseWaveformTransform):
    """
    Apply a constant amount of gain, so that highest signal level present in the sound becomes
    0 dBFS, i.e. the loudest level allowed if all samples must be between -1 and 1. Also known
    as peak normalization.
    """

    supports_multichannel = True

    def __init__(self,
                 min_gain=10,
                 max_gain=60,
                 min_colour=0,
                 max_colour=100,
                 p=0.5):
        super().__init__(p)
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.min_colour = min_colour
        self.max_colour = max_colour
        assert self.min_gain <= self.max_gain
        assert self.min_colour <= self.max_colour

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters['gain'] = random.randint(self.min_gain, self.max_gain)
            self.parameters['colour'] = random.randint(self.min_colour, self.max_colour)

    def apply(self, samples, sample_rate):
        samples_torch = torch.tensor(samples.astype(np.float32))

        if len(samples.shape) == 1:
            samples_torch = samples_torch.unsqueeze(0)

        distorted_samples = torchaudio.functional.overdrive(
            samples_torch,
            gain=self.parameters['gain'],
            colour=self.parameters['colour']
        )

        if len(samples.shape) == 1:
            distorted_samples = distorted_samples[0]

        distorted_samples = distorted_samples.numpy()

        assert distorted_samples.shape == samples.shape

        return distorted_samples