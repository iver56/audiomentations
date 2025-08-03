import random
import numpy as np
from typing import Literal
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import get_crossfade_mask_pair

# 0.00025 s corresponds to 2 samples at 8 kHz
DURATION_EPSILON = 0.00025


class TimeMask(BaseWaveformTransform):
    """
    Make a chosen part of the audio silent (time-masking augmentation).

    The silent part can optionally be faded in/out to avoid hard transients.

    Inspired by *SpecAugment: A Simple Data Augmentation Method for Automatic
    Speech Recognition* (https://arxiv.org/pdf/1904.08779.pdf)
    """

    supports_multichannel = True

    def __init__(
        self,
        min_band_part: float = 0.01,
        max_band_part: float = 0.2,
        fade_duration: float = 0.005,
        mask_location: Literal["start", "end", "random"] = "random",
        p: float = 0.5,
        **kwargs,
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
        :param mask_location: Where to place the silent region.
            * "start": silence begins at index 0
            * "end": silence ends at the last sample
            * "random" (default): silence starts at a random position
        :param p: The probability of applying this transform
        """
        if "fade" in kwargs:
            raise TypeError(
                "The 'fade' parameter was removed in v0.41.0 and is no longer supported."
                " Please use the 'fade_duration' parameter (float, seconds) instead."
                " To disable fading (equivalent to fade=False), set fade_duration=0.0."
                " To enable fading (equivalent to fade=True), set fade_duration to a"
                " positive value (e.g., the default 0.005 seconds)."
            )

        super().__init__(p)

        if not (0.0 <= min_band_part <= 1.0):
            raise ValueError("min_band_part must be between 0.0 and 1.0")
        if not (0.0 <= max_band_part <= 1.0):
            raise ValueError("max_band_part must be between 0.0 and 1.0")
        if min_band_part > max_band_part:
            raise ValueError("min_band_part must not be greater than max_band_part")

        if fade_duration < 0:
            raise ValueError("fade_duration must be non-negative")
        if 0 < fade_duration < DURATION_EPSILON:
            raise ValueError(
                f"When fade_duration is positive it must be >= {DURATION_EPSILON}"
            )

        if mask_location not in {"start", "end", "random"}:
            raise ValueError('mask_location must be "start", "end" or "random"')

        self.min_band_part = min_band_part
        self.max_band_part = max_band_part
        self.fade_duration = fade_duration
        self.mask_location = mask_location

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if not self.parameters["should_apply"]:
            return

        num_samples = samples.shape[-1]

        # Mask length
        t = random.randint(
            int(num_samples * self.min_band_part),
            int(num_samples * self.max_band_part),
        )

        # Start index based on mask_location
        loc = self.mask_location
        if loc == "start":
            t0 = 0
        elif loc == "end":
            t0 = num_samples - t
        else:  # "random"
            t0 = random.randint(0, num_samples - t)

        self.parameters.update({"t": t, "t0": t0})

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        new_samples = samples.copy()
        t: int = self.parameters["t"]
        t0: int = self.parameters["t0"]

        fade_len = 0
        if self.fade_duration > 0.0:
            fade_len = int(round(sample_rate * self.fade_duration))
            fade_len = min(fade_len, t // 2)

        if fade_len >= 2:
            fade_in, fade_out = get_crossfade_mask_pair(fade_len, equal_energy=False)

            left = slice(t0, t0 + fade_len)
            mid = slice(t0 + fade_len, t0 + t - fade_len)
            right = slice(t0 + t - fade_len, t0 + t)

            new_samples[..., left] *= fade_out
            if mid.start < mid.stop:
                new_samples[..., mid] = 0.0
            new_samples[..., right] *= fade_in
        else:
            new_samples[..., t0 : t0 + t] = 0.0

        return new_samples
