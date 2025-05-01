import random
import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import get_crossfade_mask_pair

# 0.00025 s corresponds to 2 samples at 8 kHz
DURATION_EPSILON = 0.00025


class TimeMask(BaseWaveformTransform):
    """
    Make a randomly chosen part of the audio silent (time-masking augmentation).

    The silent part can optionally be faded in/out to avoid hard transients.

    Inspired by SpecAugment, A Simple Data Augmentation Method for Automatic Speech Recognition https://arxiv.org/pdf/1904.08779.pdf
    """

    supports_multichannel = True

    def __init__(
        self,
        min_band_part: float = 0.01,
        max_band_part: float = 0.2,
        fade_duration: float = 0.005,
        p: float = 0.5,
    ):
        """
        :param min_band_part: Minimum length of the silent part as a fraction of the
            total sound length. Must be between 0.0 and 1.0
        :param max_band_part: Maximum length of the silent part as a fraction of the
            total sound length. Must be between 0.0 and 1.0
        :param fade_duration: Duration (in seconds) of the fade-in and fade-out applied
            at the edges of the silent region to smooth transitions and avoid abrupt
            changes, which can otherwise produce impulses or clicks in the audio.
            If you need hard edges or clicks, set this to `0.0` to disable fading.
            Positive values must be at least 0.00025 seconds.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        if min_band_part < 0.0 or min_band_part > 1.0:
            raise ValueError("min_band_part must be between 0.0 and 1.0")
        if max_band_part < 0.0 or max_band_part > 1.0:
            raise ValueError("max_band_part must be between 0.0 and 1.0")
        if min_band_part > max_band_part:
            raise ValueError("min_band_part must not be greater than max_band_part")

        if fade_duration < 0:
            raise ValueError("fade_duration must be non-negative")
        if 0 < fade_duration < DURATION_EPSILON:
            raise ValueError(
                "When fade_duration is set to a positive number, it must be >="
                f" {DURATION_EPSILON}"
            )

        self.min_band_part = min_band_part
        self.max_band_part = max_band_part
        self.fade_duration = fade_duration

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            num_samples = samples.shape[-1]
            self.parameters["t"] = random.randint(
                int(num_samples * self.min_band_part),
                int(num_samples * self.max_band_part),
            )
            self.parameters["t0"] = random.randint(
                0, num_samples - self.parameters["t"]
            )

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        new_samples = samples.copy()
        t = self.parameters["t"]
        t0 = self.parameters["t0"]
        mask = np.zeros(t, dtype=np.float32)

        if self.fade_duration > 0:
            fade_length = int(round(sample_rate * self.fade_duration))
            # Don't let the fade be longer than half of the silent region
            fade_length = min(fade_length, t // 2)

            if fade_length >= 2:
                fade_in, fade_out = get_crossfade_mask_pair(
                    fade_length, equal_energy=False
                )
                mask[:fade_length] = fade_out
                mask[-fade_length:] = fade_in
        new_samples[..., t0 : t0 + t] *= mask
        return new_samples
