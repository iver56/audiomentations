import random

import numpy as np
from audiomentations.core.transforms_interface import BaseWaveformTransform


class AddRandomizedPhaseShiftNoise(BaseWaveformTransform):

    def __init__(self, p=0.5, min_phase_shift=0, max_phase_shift=np.pi):
        """
        :param p:
        """
        super().__init__(p)
        self.min_phase_shift = min_phase_shift
        self.max_phase_shift = max_phase_shift

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            phase_shift = random.uniform(self.min_phase_shift, self.max_phase_shift)
            self.parameters["phase_shift"] = phase_shift

    def apply(self, samples):
        fourier = np.fft.rfft(samples)
        random_phases = np.exp(np.random.uniform(0, self.parameters["phase_shift"], int(len(samples) / 2 + 1)) * 1.0j)
        fourier_randomized = fourier * random_phases
        new_samples = np.fft.irfft(fourier_randomized)

        return new_samples
