from typing import Literal

import python_stretch
import random
import warnings

import librosa
import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform


class PitchShift(BaseWaveformTransform):
    """Pitch shift the sound up or down without changing the tempo"""

    supports_multichannel = True

    def __init__(
        self,
        min_semitones: float = -4.0,
        max_semitones: float = 4.0,
        method: Literal[
            "librosa_phase_vocoder", "signalsmith_stretch"
        ] = "signalsmith_stretch",
        p: float = 0.5,
    ):
        """
        :param min_semitones: Minimum semitones to shift. A negative number means shift down.
        :param max_semitones: Maximum semitones to shift. A positive number means shift up.
        :param method:
            "librosa_phase_vocoder": slow, low quality, supports any number of channels
            "signalsmith_stretch" (default): fast, high quality, only supports mono and stereo
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        if min_semitones < -24:
            raise ValueError("min_semitones must be >= -24")
        if max_semitones > 24:
            raise ValueError("max_semitones must be <= 24")
        if min_semitones > max_semitones:
            raise ValueError("min_semitones must not be greater than max_semitones")
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        if method not in ("librosa_phase_vocoder", "signalsmith_stretch"):
            raise ValueError(
                'method must be set to either "librosa_phase_vocoder" or'
                ' "signalsmith_stretch"'
            )
        self.method = method

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["num_semitones"] = random.uniform(
                self.min_semitones, self.max_semitones
            )

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        if self.method == "signalsmith_stretch":
            original_ndim = samples.ndim
            if original_ndim == 1:
                samples = samples[np.newaxis, :]

            if (
                original_ndim == 2
                and samples.shape[0] > 1
                and not samples.flags.c_contiguous
            ):
                samples = np.ascontiguousarray(samples)

            stretch = python_stretch.Signalsmith.Stretch()
            stretch.preset(samples.shape[0], sample_rate)
            stretch.setTransposeSemitones(self.parameters["num_semitones"])
            samples = stretch.process(samples)
            if samples.ndim > original_ndim:
                samples = samples[0]
            return samples

        try:
            resample_type = (
                "kaiser_best" if librosa.__version__.startswith("0.8.") else "soxr_hq"
            )
            pitch_shifted_samples = librosa.effects.pitch_shift(
                samples,
                sr=sample_rate,
                n_steps=self.parameters["num_semitones"],
                res_type=resample_type,
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
                )

        return pitch_shifted_samples
