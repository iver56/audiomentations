import random

import numpy as np

from audiomentations.core.transforms_interface import BasicTransform


class AddGaussianNoise(BasicTransform):
    def __init__(self, min_amplitude=0.001, max_amplitude=0.006, p=0.5):
        super().__init__(p)
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def apply(self, samples, sample_rate):
        noise = np.random.randn(len(samples))
        amplitude = random.uniform(self.min_amplitude, self.max_amplitude)
        samples = samples + amplitude * noise
        return samples
