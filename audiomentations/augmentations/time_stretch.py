import random

import librosa
import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform


class TimeStretch(BaseWaveformTransform):
    """Time stretch the signal without changing the pitch"""

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
        try:
            time_stretched_samples = librosa.effects.time_stretch(
                samples, rate=self.parameters["rate"]
            )
        except librosa.util.exceptions.ParameterError:
            # In librosa<0.9.0 time_stretch doesn't natively support multichannel audio.
            # Here we use a workaround that simply loops over the channels instead.
            # TODO: Remove this workaround when we remove support for librosa<0.9.0
            time_stretched_channels = []
            for i in range(samples.shape[0]):
                time_stretched_samples = librosa.effects.time_stretch(
                    samples[i], rate=self.parameters["rate"]
                )
                time_stretched_channels.append(time_stretched_samples)
            time_stretched_samples = np.array(
                time_stretched_channels, dtype=samples.dtype
            )

        if self.leave_length_unchanged:
            # Apply zero padding if the time stretched audio is not long enough to fill the
            # whole space, or crop the time stretched audio if it ended up too long.
            padded_samples = np.zeros(shape=samples.shape, dtype=samples.dtype)
            window = time_stretched_samples[..., : samples.shape[-1]]
            actual_window_length = window.shape[
                -1
            ]  # may be smaller than samples.shape[-1]
            padded_samples[..., :actual_window_length] = window
            time_stretched_samples = padded_samples
        return time_stretched_samples
