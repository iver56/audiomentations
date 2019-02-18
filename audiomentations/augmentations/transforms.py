import random

import librosa
import numpy as np

from audiomentations.core.transforms_interface import BasicTransform


class AddGaussianNoise(BasicTransform):
    def __init__(self, min_amplitude=0.001, max_amplitude=0.015, p=0.5):
        super().__init__(p)
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def apply(self, samples, sample_rate):
        noise = np.random.randn(len(samples))
        amplitude = random.uniform(self.min_amplitude, self.max_amplitude)
        samples = samples + amplitude * noise
        return samples


class TimeStretch(BasicTransform):
    """
    Note: This transform potentially changes the length of the sound
    """
    def __init__(self, min_rate=0.8, max_rate=1.25, p=0.5):
        super().__init__(p)
        assert min_rate > 0.1
        assert max_rate < 10
        assert min_rate <= max_rate
        self.min_rate = min_rate
        self.max_rate = max_rate

    def apply(self, samples, sample_rate):
        """
        If `rate > 1`, then the signal is sped up.
        If `rate < 1`, then the signal is slowed down.
        """
        rate = random.uniform(self.min_rate, self.max_rate)
        return librosa.effects.time_stretch(samples, rate)
