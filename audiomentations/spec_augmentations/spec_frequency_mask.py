import random

import numpy as np

from audiomentations.core.transforms_interface import BaseSpectrogramTransform


class SpecFrequencyMask(BaseSpectrogramTransform):
    """
    Mask a set of frequencies in a spectrogram, à la Google AI SpecAugment. This type of data
    augmentation has proved to make speech recognition models more robust.

    The masked frequencies can be replaced with either the mean of the original values or a
    given constant (e.g. zero).
    """

    supports_multichannel = True

    def __init__(
        self,
        min_mask_fraction: float = 0.03,
        max_mask_fraction: float = 0.25,
        fill_mode: str = "constant",
        fill_constant: float = 0.0,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.min_mask_fraction = min_mask_fraction
        self.max_mask_fraction = max_mask_fraction
        assert fill_mode in ("mean", "constant")
        self.fill_mode = fill_mode
        self.fill_constant = fill_constant

    def randomize_parameters(self, magnitude_spectrogram):
        super().randomize_parameters(magnitude_spectrogram)
        if self.parameters["should_apply"]:
            num_frequency_bins = magnitude_spectrogram.shape[0]
            min_frequencies_to_mask = int(
                round(self.min_mask_fraction * num_frequency_bins)
            )
            max_frequencies_to_mask = int(
                round(self.max_mask_fraction * num_frequency_bins)
            )
            num_frequencies_to_mask = random.randint(
                min_frequencies_to_mask, max_frequencies_to_mask
            )
            self.parameters["start_frequency_index"] = random.randint(
                0, num_frequency_bins - num_frequencies_to_mask
            )
            self.parameters["end_frequency_index"] = (
                self.parameters["start_frequency_index"] + num_frequencies_to_mask
            )

    def apply(self, magnitude_spectrogram):
        if self.fill_mode == "mean":
            fill_value = np.mean(
                magnitude_spectrogram[
                self.parameters["start_frequency_index"] : self.parameters[
                    "end_frequency_index"
                ]
                ]
            )
        else:
            # self.fill_mode == "constant"
            fill_value = self.fill_constant
        magnitude_spectrogram = magnitude_spectrogram.copy()
        magnitude_spectrogram[
        self.parameters["start_frequency_index"] : self.parameters[
            "end_frequency_index"
        ]
        ] = fill_value
        return magnitude_spectrogram

