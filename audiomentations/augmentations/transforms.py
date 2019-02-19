import random

import librosa
import numpy as np

from audiomentations.core.transforms_interface import BasicTransform


class AddGaussianNoise(BasicTransform):
    """Add gaussian noise to the samples"""
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
    """Time stretch the signal without changing the pitch"""
    def __init__(self, min_rate=0.8, max_rate=1.25, leave_length_unchanged=True, p=0.5):
        super().__init__(p)
        assert min_rate > 0.1
        assert max_rate < 10
        assert min_rate <= max_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.leave_length_unchanged = leave_length_unchanged

    def apply(self, samples, sample_rate):
        """
        If `rate > 1`, then the signal is sped up.
        If `rate < 1`, then the signal is slowed down.
        """
        rate = random.uniform(self.min_rate, self.max_rate)
        time_stretched_samples = librosa.effects.time_stretch(samples, rate)
        if self.leave_length_unchanged:
            # Apply zero padding if the time stretched audio is not long enough to fill the
            # whole space, or crop the time stretched audio if it ended up too long.
            padded_samples = np.zeros(shape=samples.shape, dtype=samples.dtype)
            window = time_stretched_samples[: samples.shape[0]]
            actual_window_length = len(window)  # may be smaller than samples.shape[0]
            padded_samples[:actual_window_length] = window
            time_stretched_samples = padded_samples
        return time_stretched_samples


class PitchShift(BasicTransform):
    """Pitch shift the sound up or down without changing the tempo"""
    def __init__(self, min_semitones=-4, max_semitones=4, p=0.5):
        super().__init__(p)
        assert min_semitones >= -12
        assert max_semitones <= 12
        assert min_semitones <= max_semitones
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def apply(self, samples, sample_rate):
        num_semitones = random.uniform(self.min_semitones, self.max_semitones)
        pitch_shifted_samples = librosa.effects.pitch_shift(
            samples, sample_rate, n_steps=num_semitones
        )
        return pitch_shifted_samples
