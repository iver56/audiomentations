import random

import librosa
import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform


class PitchShift(BaseWaveformTransform):
    """Pitch shift the sound up or down without changing the tempo"""

    supports_multichannel = True

    def __init__(self, min_semitones=-4, max_semitones=4, p=0.5):
        super().__init__(p)
        assert min_semitones >= -12
        assert max_semitones <= 12
        assert min_semitones <= max_semitones
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["num_semitones"] = random.uniform(
                self.min_semitones, self.max_semitones
            )

    def apply(self, samples, sample_rate):
        if samples.ndim == 2:
            # librosa's pitch_shift function doesn't natively support multichannel audio.
            # Here we use a workaround that simply loops over the channels. It's not perfect.
            # TODO: When librosa has closed the following issue, we can remove our workaround:
            # https://github.com/librosa/librosa/issues/1085
            pitch_shifted_samples = np.copy(samples)
            for i in range(samples.shape[0]):
                pitch_shifted_samples[i] = librosa.effects.pitch_shift(
                    pitch_shifted_samples[i],
                    sample_rate,
                    n_steps=self.parameters["num_semitones"],
                )
        else:
            pitch_shifted_samples = librosa.effects.pitch_shift(
                samples, sample_rate, n_steps=self.parameters["num_semitones"]
            )
        return pitch_shifted_samples
