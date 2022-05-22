import random

import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform


class Padding(BaseWaveformTransform):
    """
    Apply padding to the audio signal - take a fraction of the end or the start of the
    audio and replace that part with padding. This can be useful for preparing ML models
    with constant input length for padded inputs.
    """

    supports_multichannel = True

    def __init__(
        self,
        mode="silence",
        min_fraction=0.01,
        max_fraction=0.7,
        pad_section="end",
        p=0.5,
    ):
        """
        :param mode: Padding mode. Must be one of "silence", "wrap", "reflect"
        :param min_fraction: Minimum fraction of the signal duration to be padded
        :param max_fraction: Maximum fraction of the signal duration to be padded
        :param pad_section: Which part of the signal should be replaced with padding:
            "start" or "end"
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        assert mode in ("silence", "wrap", "reflect")
        self.mode = mode

        assert max_fraction <= 1.0
        assert min_fraction >= 0
        assert min_fraction <= max_fraction
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction

        assert pad_section in ("start", "end")
        self.pad_section = pad_section

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            input_length = samples.shape[-1]
            self.parameters["padding_length"] = random.randint(
                int(round(self.min_fraction * input_length)),
                int(round(self.max_fraction * input_length)),
            )

    def apply(self, samples, sample_rate):
        padding_length = self.parameters["padding_length"]
        if padding_length == 0:
            return samples

        untouched_length = samples.shape[-1] - padding_length

        if self.mode == "silence":
            samples = np.copy(samples)
            if self.pad_section == "start":
                samples[..., :padding_length] = 0.0
            else:
                samples[..., -padding_length:] = 0.0
        else:
            if samples.ndim == 1:
                if self.pad_section == "start":
                    pad_width = (padding_length, 0)
                else:
                    pad_width = (0, padding_length)
            else:
                if self.pad_section == "start":
                    pad_width = ((0, 0), (padding_length, 0))
                else:
                    pad_width = ((0, 0), (0, padding_length))

            if self.pad_section == "start":
                samples = samples[..., -untouched_length:]
            else:
                samples = samples[..., :untouched_length]

            samples = np.pad(samples, pad_width, self.mode)

        return samples
