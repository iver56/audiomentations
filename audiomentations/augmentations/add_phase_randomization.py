import random

import numpy as np
from audiomentations.core.transforms_interface import BaseWaveformTransform


class AddRandomizedPhaseShiftNoise(BaseWaveformTransform):

    def __init__(self, p=0.5):
        """
        :param p:
        """
        super().__init__(p)

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            phase_shift = random.uniform(0, np.pi)
            self.parameters["phase_shift"] = phase_shift

    def apply(self, samples):
        fourier = np.fft.rfft(samples)
        random_phases = np.exp(np.random.uniform(0, self.parameters["phase_shift"], int(len(samples) / 2 + 1)) * 1.0j)
        fourier_randomized = fourier * random_phases
        new_samples = np.fft.irfft(fourier_randomized)

        return new_samples
