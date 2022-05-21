import random

import librosa
import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform


class SampleRateChange(BaseWaveformTransform):
    """Time stretch the signal with matching change of the pitch"""

    supports_multichannel = True

    def __init__(self, min_rate=0.8, max_rate=1.25, leave_length_unchanged=True, p=0.5):
        super().__init__(p)
        assert min_rate > 0.1
        assert max_rate < 10
        assert min_rate <= max_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.leave_length_unchanged = leave_length_unchanged

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            """
            If rate > 1, then the signal is sped up.
            If rate < 1, then the signal is slowed down.
            """
            self.parameters["rate"] = random.uniform(self.min_rate, self.max_rate)

    def apply(self, samples, sample_rate):
        augmented_samples = librosa.core.resample(
            samples,
            orig_sr=sample_rate,
            target_sr=self.parameters["rate"]*sample_rate,
        )
        if self.leave_length_unchanged:
            # Apply zero padding if the time stretched audio is not long enough to fill the
            # whole space, or crop the time stretched audio if it ended up too long.
            padded_samples = np.zeros(shape=samples.shape, dtype=samples.dtype)
            window = augmented_samples[..., : samples.shape[-1]]
            actual_window_length = window.shape[
                -1
            ]  # may be smaller than samples.shape[-1]
            padded_samples[..., :actual_window_length] = window
            augmented_samples = padded_samples
        return augmented_samples
