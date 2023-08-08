import random
from typing import Optional, Callable

import math
import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import convert_decibels_to_amplitude_ratio


class RepeatPart(BaseWaveformTransform):
    """
    Select a part of the audio and repeat that part a number of times
    """

    supports_multichannel = True

    def __init__(
        self,
        min_repeats: int = 1,
        max_repeats: int = 3,
        min_part_duration: float = 0.25,
        max_part_duration: float = 1.2,
        mode: str = "insert",
        crossfade: bool = True,
        crossfade_duration: float = 0.005,
        part_transform: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        p: float = 0.5,
    ):
        """
        TODO: docstring goes here
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        assert min_repeats >= 1
        assert max_repeats >= min_repeats
        self.min_repeats = min_repeats
        self.max_repeats = max_repeats
        assert min_part_duration >= 0.0
        assert max_part_duration >= min_part_duration
        self.min_part_duration = min_part_duration
        self.max_part_duration = max_part_duration
        assert mode in ("insert", "replace")
        self.mode = mode
        self.crossfade = crossfade
        self.crossfade_duration = crossfade_duration
        self.part_transform = part_transform  # TODO: Actually implement the use of this

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["part_num_samples"] = random.randint(
                int(self.min_part_duration * sample_rate),
                int(self.max_part_duration * sample_rate),
            )
            if self.parameters["part_num_samples"] > samples.shape[-1]:
                # The input sound is not long enough for applying the transform in this case
                self.parameters["should_apply"] = False
                return

            self.parameters["repeats"] = random.randint(
                self.min_repeats, self.max_repeats
            )
            self.parameters["part_start_index"] = random.randint(
                0, samples.shape[-1] - self.parameters["part_duration_samples"]
            )

    def apply(self, samples: np.ndarray, sample_rate: int):
        part = samples[
            ...,
            self.parameters["part_start_index"] : self.parameters["part_start_index"]
            + self.parameters["part_num_samples"],
        ]
        if self.crossfade:
            # TODO: Maybe do sqrt-based crossfade, like I learned in the first semester of mustek?
            pass
        else:
            if self.mode == "insert":
                parts = np.tile(part, self.parameters["repeats"])

                result_length = samples.shape[-1] + parts.shape[-1]
                if samples.ndim == 1:
                    result_shape = (result_length,)
                else:
                    result_shape = (samples.shape[0], result_length)

                repeats_start_index = (
                    self.parameters["part_start_index"]
                    + self.parameters["part_num_samples"]
                )
                repeats_end_index = repeats_start_index + parts.shape[-1]

                result_placeholder = np.zeros(shape=result_shape, dtype=np.float32)
                if self.parameters["part_start_index"] > 0:
                    result_placeholder[..., :repeats_start_index] = samples[
                        ..., :repeats_start_index
                    ]

                result_placeholder[
                    ...,
                    repeats_start_index:repeats_end_index,
                ] = parts

                result_placeholder[..., repeats_end_index:] = samples[
                    ..., -(result_length - repeats_end_index):
                ]
                return result_placeholder
            else:
                repeated_part_start_index = (
                    self.parameters["part_start_index"]
                    + self.parameters["part_num_samples"]
                )
                max_repeated_part_length = samples.shape[-1] - repeated_part_start_index
                if self.parameters["repeats"] > 1:
                    # Limit the number of tiles to what fits inside the samples
                    repeats = min(
                        self.parameters["repeats"],
                        int(math.ceil(max_repeated_part_length / part.shape[-1])),
                    )
                    parts = np.tile(part, repeats)
                else:
                    parts = part

                if parts.shape[-1] > max_repeated_part_length:
                    parts = parts[..., :max_repeated_part_length]

                samples[
                    ...,
                    repeated_part_start_index : repeated_part_start_index
                    + parts.shape[-1],
                ] = parts
                return samples
