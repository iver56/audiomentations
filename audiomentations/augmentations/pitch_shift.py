import random
import warnings

import librosa
import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform


class PitchShift(BaseWaveformTransform):
    """Pitch shift the sound up or down without changing the tempo"""

    supports_multichannel = True

    def __init__(
        self, min_semitones: float = -4.0, max_semitones: float = 4, res_type: str = "soxr_hq", p: float = 0.5
    ):
        """
        :param min_semitones: Minimum semitones to shift. Negative number means shift down.
        :param max_semitones: Maximum semitones to shift. Positive number means shift up.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_semitones >= -12
        assert max_semitones <= 12
        assert min_semitones <= max_semitones
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.res_type = res_type

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["num_semitones"] = random.uniform(
                self.min_semitones, self.max_semitones
            )

    def apply(self, samples: np.ndarray, sample_rate: int):
        try:
            pitch_shifted_samples = librosa.effects.pitch_shift(
                samples, sr=sample_rate, n_steps=self.parameters["num_semitones"]
            )
        except librosa.util.exceptions.ParameterError:
            warnings.warn(
                "Warning: You are probably using an old version of librosa. Upgrade"
                " librosa to 0.9.0 or later for better performance when applying"
                " PitchShift to stereo audio."
            )
            # In librosa<0.9.0 pitch_shift doesn't natively support multichannel audio.
            # Here we use a workaround that simply loops over the channels instead.
            # TODO: Remove this workaround when we remove support for librosa<0.9.0
            pitch_shifted_samples = np.copy(samples)
            for i in range(samples.shape[0]):
                pitch_shifted_samples[i] = librosa.effects.pitch_shift(
                    pitch_shifted_samples[i],
                    sr=sample_rate,
                    n_steps=self.parameters["num_semitones"],
                    res_type=self.res_type,
                )

        return pitch_shifted_samples
