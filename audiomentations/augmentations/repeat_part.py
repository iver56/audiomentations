import random
from typing import Optional, Callable

import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform


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
        self.part_transform = (
            part_transform  # TODO: implement freeze transforms for the part_transform
        )

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
                0, samples.shape[-1] - self.parameters["part_num_samples"]
            )

    @staticmethod
    def transform_parts(parts: np.ndarray, part_length: int, part_transform):
        for i in range(0, parts.shape[-1], part_length):
            parts[..., i : i + part_length] = part_transform(
                parts[..., i : i + part_length]
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
            repeats_start_index = (
                self.parameters["part_start_index"]
                + self.parameters["part_num_samples"]
            )

            last_end_index = repeats_start_index
            parts = []
            for i in range(self.parameters["repeats"]):
                start_idx = last_end_index

                part_array = part

                if self.part_transform:
                    part_array = self.part_transform(np.copy(part_array), sample_rate)

                last_end_index = start_idx + part_array.shape[-1]

                stop = False
                if self.mode == "replace" and last_end_index > samples.shape[-1]:
                    limited_part_length = samples.shape[-1] - start_idx
                    last_end_index = start_idx + limited_part_length
                    part_array = part_array[..., :limited_part_length]
                    stop = True

                parts.append(
                    {"array": part_array, "start_idx": start_idx, "end_idx": last_end_index}
                )
                if stop:
                    break

            repeats_length = parts[-1]["end_idx"] - parts[0]["start_idx"]
            result_length = samples.shape[-1]
            if self.mode == "insert":
                result_length += repeats_length

            if samples.ndim == 1:
                result_shape = (result_length,)
            else:
                result_shape = (samples.shape[0], result_length)

            result_placeholder = np.zeros(shape=result_shape, dtype=np.float32)
            result_placeholder[..., :repeats_start_index] = samples[
                ..., :repeats_start_index
            ]

            for part in parts:
                result_placeholder[..., part["start_idx"] : part["end_idx"]] = part[
                    "array"
                ]  # needs to be += instead of = when crossfading is enabled
            del parts

            if self.mode == "insert":
                result_placeholder[..., last_end_index:] = samples[
                    ..., -(result_length - last_end_index) :
                ]
            else:
                if last_end_index < result_length:
                    result_placeholder[..., last_end_index:] = samples[..., last_end_index:]
            return result_placeholder

    def freeze_parameters(self):
        super().freeze_parameters()
        if hasattr(self.part_transform, "freeze_parameters"):
            self.part_transform.freeze_parameters()

    def unfreeze_parameters(self):
        super().unfreeze_parameters()
        if hasattr(self.part_transform, "unfreeze_parameters"):
            self.part_transform.unfreeze_parameters()
