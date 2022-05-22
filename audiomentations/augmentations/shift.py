import random

import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform


class Shift(BaseWaveformTransform):
    """
    Shift the samples forwards or backwards, with or without rollover
    """

    supports_multichannel = True

    def __init__(
        self,
        min_fraction=-0.5,
        max_fraction=0.5,
        rollover=True,
        fade=False,
        fade_duration=0.01,
        p=0.5,
    ):
        """
        :param min_fraction: float, fraction of total sound length
        :param max_fraction: float, fraction of total sound length
        :param rollover: When set to True, samples that roll beyond the first or last position
            are re-introduced at the last or first. When set to False, samples that roll beyond
            the first or last position are discarded. In other words, rollover=False results in
            an empty space (with zeroes).
        :param fade: When set to True, there will be a short fade in and/or out at the "stitch"
            (that was the start or the end of the audio before the shift). This can smooth out an
            unwanted abrupt change between two consecutive samples (which sounds like a
            transient/click/pop).
        :param fade_duration: If `fade=True`, then this is the duration of the fade in seconds.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_fraction >= -1
        assert max_fraction <= 1
        assert type(fade_duration) in [int, float] or not fade
        assert fade_duration > 0 or not fade
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.rollover = rollover
        self.fade = fade
        self.fade_duration = fade_duration

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["num_places_to_shift"] = int(
                round(
                    random.uniform(self.min_fraction, self.max_fraction)
                    * samples.shape[-1]
                )
            )

    def apply(self, samples, sample_rate):
        num_places_to_shift = self.parameters["num_places_to_shift"]
        shifted_samples = np.roll(samples, num_places_to_shift, axis=-1)

        if not self.rollover:
            if num_places_to_shift > 0:
                shifted_samples[..., :num_places_to_shift] = 0.0
            elif num_places_to_shift < 0:
                shifted_samples[..., num_places_to_shift:] = 0.0

        if self.fade:
            fade_length = int(sample_rate * self.fade_duration)

            fade_in = np.linspace(0, 1, num=fade_length)
            fade_out = np.linspace(1, 0, num=fade_length)

            if num_places_to_shift > 0:

                fade_in_start = num_places_to_shift
                fade_in_end = min(
                    num_places_to_shift + fade_length, shifted_samples.shape[-1]
                )
                fade_in_length = fade_in_end - fade_in_start

                shifted_samples[
                ...,
                fade_in_start:fade_in_end,
                ] *= fade_in[:fade_in_length]

                if self.rollover:

                    fade_out_start = max(num_places_to_shift - fade_length, 0)
                    fade_out_end = num_places_to_shift
                    fade_out_length = fade_out_end - fade_out_start

                    shifted_samples[..., fade_out_start:fade_out_end] *= fade_out[
                                                                         -fade_out_length:
                                                                         ]

            elif num_places_to_shift < 0:

                positive_num_places_to_shift = (
                    shifted_samples.shape[-1] + num_places_to_shift
                )

                fade_out_start = max(positive_num_places_to_shift - fade_length, 0)
                fade_out_end = positive_num_places_to_shift
                fade_out_length = fade_out_end - fade_out_start

                shifted_samples[..., fade_out_start:fade_out_end] *= fade_out[
                                                                     -fade_out_length:
                                                                     ]

                if self.rollover:
                    fade_in_start = positive_num_places_to_shift
                    fade_in_end = min(
                        positive_num_places_to_shift + fade_length,
                        shifted_samples.shape[-1],
                        )
                    fade_in_length = fade_in_end - fade_in_start
                    shifted_samples[
                    ...,
                    fade_in_start:fade_in_end,
                    ] *= fade_in[:fade_in_length]

        return shifted_samples
