import random
import warnings
from typing import Optional, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import get_crossfade_mask_pair, get_crossfade_length

# 0.00025 seconds corresponds to 2 samples at 8000 Hz
DURATION_EPSILON = 0.00025


class RepeatPart(BaseWaveformTransform):
    """
    Select a subsection (or "part") of the audio and repeat that part a number of times.
    This can be useful when simulating scenarios where a short audio snippet gets
    repeated, for example:

    * Repetitions of some musical note or sound in a rhythmical way
    * A person stutters or says the same word (with variations) multiple times in a row
    * A mechanical noise with periodic repetitions
    * A "skip in the record" or a "stuck needle" effect, reminiscent of vinyl records or
        CDs when they repeatedly play a short section due to a scratch or other
        imperfection.
    * Digital audio glitches, such as a buffer underrun in video games,
        where the current audio frame gets looped continuously due to system overloads
        or a software crash.

    Note that the length of inputs you give it must be compatible with the part
    duration range and crossfade duration. If you give it an input audio array that is
    too short, a `UserWarning` will be raised and no operation is applied to the signal.
    """

    supports_multichannel = True

    def __init__(
        self,
        min_repeats: int = 1,
        max_repeats: int = 3,
        min_part_duration: float = 0.25,
        max_part_duration: float = 1.2,
        mode: Literal["insert", "replace"] = "insert",
        crossfade_duration: float = 0.005,
        part_transform: Optional[
            Callable[[NDArray[np.float32], int], NDArray[np.float32]]
        ] = None,
        p: float = 0.5,
    ):
        """
        :param min_repeats: Minimum number of times a selected audio segment should be
            repeated in addition to the original. For instance, if the selected number
            of repeats is 1, the selected segment will be followed by one repeat.
        :param max_repeats: Maximum number of times a selected audio segment can be
            repeated in addition to the original.
        :param min_part_duration: Minimum duration (in seconds) of the audio segment
            that can be selected for repetition.
        :param max_part_duration: Maximum duration (in seconds) of the audio segment
            that can be selected for repetition.
        :param mode: This parameter has two options:
            "insert": Insert the repeat(s), making the array longer. After the last
                repeat there will be the last part of the original audio, offset in time
                compared to the input array.
            "replace": Have the repeats replace (as in overwrite) the original audio.
                Any remaining part at the end (if not overwritten by repeats) will be
                left untouched without offset. The length of the output array is the
                same as the input array.
        :param crossfade_duration: Duration (in seconds) for crossfading between repeated
            parts as well as potentially from the original audio to the repetitions and back.
            The crossfades will be equal-energy or equal-gain depending on the audio and/or the
            chosen parameters of the transform. The crossfading feature can be used to smooth
            transitions and avoid abrupt changes, which can lead to impulses/clicks in the audio.
            If you know what you're doing, and impulses/clicks are desired for your use case,
            you can disable the crossfading by setting this value to `0.0`.
        :param part_transform: An optional callable (audiomentations transform) that
            gets applied individually to each repeat. This can be used to make each
            repeat slightly different from the previous one. Note that a part_transform
            that makes the part shorter is only supported if the transformed part is at
            least two times the crossfade duration.
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        if min_repeats < 1:
            raise ValueError("min_repeats must be >= 1")
        if max_repeats < min_repeats:
            raise ValueError("max_repeats must be >= min_repeats")
        self.min_repeats = min_repeats
        self.max_repeats = max_repeats
        if min_part_duration < DURATION_EPSILON:
            raise ValueError(f"min_part_duration must be >= {DURATION_EPSILON}")
        if max_part_duration < min_part_duration:
            raise ValueError("max_part_duration must be >= min_part_duration")
        self.min_part_duration = min_part_duration
        self.max_part_duration = max_part_duration
        if mode not in ("insert", "replace"):
            raise ValueError('mode must be set to either "insert" or "replace"')
        self.mode = mode

        if crossfade_duration == 0.0:
            self.crossfade = False
        elif crossfade_duration < 0.0:
            raise ValueError("crossfade_duration must not be negative")
        elif crossfade_duration < DURATION_EPSILON:
            raise ValueError(
                "When crossfade_duration is set to a positive number, it must be >="
                f" {DURATION_EPSILON}"
            )
        else:
            self.crossfade = True
        if crossfade_duration > (min_part_duration / 2):
            raise ValueError(
                "crossfade_duration must be <= 0.5 * min_part_duration. You can fix this"
                " error by increasing min_part_duration or by decreasing"
                " crossfade_duration."
            )
        self.crossfade_duration = crossfade_duration
        self.part_transform = part_transform

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["part_num_samples"] = random.randint(
                int(self.min_part_duration * sample_rate),
                int(self.max_part_duration * sample_rate),
            )

            crossfade_length = get_crossfade_length(
                sample_rate, self.crossfade_duration
            )
            half_crossfade_length = crossfade_length // 2
            if (
                half_crossfade_length
                >= samples.shape[-1] - self.parameters["part_num_samples"]
            ):
                warnings.warn(
                    "The input sound is not long enough for applying the RepeatPart"
                    " transform with the current parameters."
                )
                self.parameters["should_apply"] = False
                return

            self.parameters["part_start_index"] = random.randint(
                half_crossfade_length,
                samples.shape[-1] - self.parameters["part_num_samples"],
            )
            self.parameters["repeats"] = random.randint(
                self.min_repeats, self.max_repeats
            )

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        crossfade_length = 0
        half_crossfade_length = 0
        equal_energy_fade_in_mask = None
        equal_energy_fade_out_mask = None
        last_crossfade_type = "equal_energy"
        if self.crossfade:
            crossfade_length = get_crossfade_length(
                sample_rate, self.crossfade_duration
            )
            half_crossfade_length = crossfade_length // 2
            equal_energy_fade_in_mask, equal_energy_fade_out_mask = (
                get_crossfade_mask_pair(crossfade_length)
            )

        if self.crossfade:
            part = samples[
                ...,
                self.parameters["part_start_index"]
                - half_crossfade_length : self.parameters["part_start_index"]
                + self.parameters["part_num_samples"]
                + half_crossfade_length,
            ]
        else:
            part = samples[
                ...,
                self.parameters["part_start_index"] : self.parameters[
                    "part_start_index"
                ]
                + self.parameters["part_num_samples"],
            ]

        repeats_start_index = (
            self.parameters["part_start_index"] + self.parameters["part_num_samples"]
        )

        last_repetition_end_index = repeats_start_index
        parts = []
        for i in range(self.parameters["repeats"]):
            start_idx = last_repetition_end_index
            if self.crossfade:
                if i == 0:
                    start_idx -= half_crossfade_length
                else:
                    start_idx -= crossfade_length

            part_array = np.copy(part)

            if self.part_transform:
                part_array = self.part_transform(part_array, sample_rate)
                if self.crossfade and part_array.shape[-1] < 2 * crossfade_length:
                    raise ValueError(
                        "Applying a part_transform that makes a part shorter than 2 *"
                        " crossfade_duration is not supported"
                    )

            last_repetition_end_index = start_idx + part_array.shape[-1]

            if self.crossfade:
                part_array[..., :crossfade_length] *= equal_energy_fade_in_mask

                fade_out_crossfade_type = "equal_energy"
                is_last_part = i == self.parameters["repeats"] - 1
                if is_last_part and self.mode == "insert":
                    if self.part_transform is None:
                        fade_out_crossfade_type = "equal_gain"
                    else:
                        # If the part was transformed, check if it still correlates with
                        # the original signal in the seam. If it does, we use equal-gain
                        # crossfading instead of equal-energy.
                        correlation_coefficient = np.corrcoef(
                            part_array[..., -crossfade_length:],
                            samples[
                                ...,
                                self.parameters["part_start_index"]
                                + self.parameters["part_num_samples"]
                                - half_crossfade_length : self.parameters[
                                    "part_start_index"
                                ]
                                + self.parameters["part_num_samples"]
                                + half_crossfade_length,
                            ],
                        )[0, 1]
                        if abs(correlation_coefficient) > 0.5:
                            fade_out_crossfade_type = "equal_gain"
                if is_last_part:
                    last_crossfade_type = fade_out_crossfade_type
                if fade_out_crossfade_type == "equal_energy":
                    part_array[..., -crossfade_length:] *= equal_energy_fade_out_mask
                else:
                    _, equal_gain_fade_out_mask = get_crossfade_mask_pair(
                        crossfade_length, equal_energy=False
                    )
                    part_array[..., -crossfade_length:] *= equal_gain_fade_out_mask

            parts.append(
                {
                    "array": part_array,
                    "start_idx": start_idx,
                    "end_idx": last_repetition_end_index,
                }
            )
            if self.mode == "replace" and last_repetition_end_index > samples.shape[-1]:
                break

        result_length = samples.shape[-1]
        if self.mode == "insert":
            result_length += (
                parts[-1]["end_idx"] - parts[0]["start_idx"] - crossfade_length
            )

        if samples.ndim == 1:
            result_shape = (result_length,)
        else:
            result_shape = (samples.shape[0], result_length)

        result_placeholder = np.zeros(shape=result_shape, dtype=np.float32)

        if self.crossfade:
            # Fade out the signal where the first repetition takes over
            result_placeholder[..., : repeats_start_index - half_crossfade_length] = (
                samples[..., : repeats_start_index - half_crossfade_length]
            )
            result_placeholder[
                ...,
                repeats_start_index - half_crossfade_length : repeats_start_index
                + half_crossfade_length,
            ] = (
                equal_energy_fade_out_mask
                * samples[
                    ...,
                    repeats_start_index - half_crossfade_length : repeats_start_index
                    + half_crossfade_length,
                ]
            )
        else:
            result_placeholder[..., :repeats_start_index] = samples[
                ..., :repeats_start_index
            ]

        # Add all repetitions except the last one
        for part in parts[:-1]:
            result_placeholder[..., part["start_idx"] : part["end_idx"]] += part[
                "array"
            ]

        # Add the (potentially truncated) last repetition
        if self.mode == "replace" and parts[-1]["end_idx"] > samples.shape[-1]:
            truncated_part_length = samples.shape[-1] - parts[-1]["start_idx"]
            truncated_part = parts[-1]["array"][..., :truncated_part_length]
            result_placeholder[
                ...,
                parts[-1]["start_idx"] : parts[-1]["start_idx"] + truncated_part_length,
            ] += truncated_part
        else:
            result_placeholder[..., parts[-1]["start_idx"] : parts[-1]["end_idx"]] += (
                parts[-1]["array"]
            )

        del parts

        if self.mode == "insert":
            if self.crossfade:
                if last_crossfade_type == "equal_energy":
                    fade_in_mask = equal_energy_fade_in_mask
                else:
                    fade_in_mask, _ = get_crossfade_mask_pair(
                        crossfade_length, equal_energy=False
                    )
                result_placeholder[
                    ...,
                    last_repetition_end_index
                    - crossfade_length : last_repetition_end_index,
                ] += (
                    fade_in_mask
                    * samples[
                        ...,
                        -(result_length - last_repetition_end_index)
                        - crossfade_length : -(
                            result_length - last_repetition_end_index
                        ),
                    ]
                )

            result_placeholder[..., last_repetition_end_index:] = samples[
                ..., -(result_length - last_repetition_end_index) :
            ]
        else:
            if self.crossfade:
                last_crossfade_start_idx = last_repetition_end_index - crossfade_length
                if last_crossfade_start_idx < samples.shape[-1]:
                    last_crossfade_length = crossfade_length
                    if last_repetition_end_index > samples.shape[-1]:
                        # Truncate crossfade
                        last_crossfade_length = (
                            samples.shape[-1] - last_crossfade_start_idx
                        )

                    result_placeholder[
                        ...,
                        last_crossfade_start_idx : last_crossfade_start_idx
                        + last_crossfade_length,
                    ] += (
                        equal_energy_fade_in_mask[:last_crossfade_length]
                        * samples[
                            ...,
                            last_crossfade_start_idx : last_crossfade_start_idx
                            + last_crossfade_length,
                        ]
                    )

            if last_repetition_end_index < result_length:
                result_placeholder[..., last_repetition_end_index:] = samples[
                    ..., last_repetition_end_index:
                ]
        return result_placeholder

    def freeze_parameters(self):
        super().freeze_parameters()
        if hasattr(self.part_transform, "freeze_parameters"):
            self.part_transform.freeze_parameters()

    def unfreeze_parameters(self):
        super().unfreeze_parameters()
        if hasattr(self.part_transform, "unfreeze_parameters"):
            self.part_transform.unfreeze_parameters()
