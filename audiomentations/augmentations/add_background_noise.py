import functools
import random
import warnings
from pathlib import Path
from typing import Optional, List, Callable, Union, Literal

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.audio_loading_utils import load_sound_file
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    calculate_desired_noise_rms,
    calculate_rms,
    convert_decibels_to_amplitude_ratio,
    find_audio_files_in_paths,
)


class AddBackgroundNoise(BaseWaveformTransform):
    """Mix in another sound, e.g. a background noise. Useful if your original sound is clean and
    you want to simulate an environment where background noise is present.
    Can also be used for mixup, as in https://arxiv.org/pdf/1710.09412.pdf
    A folder of (background noise) sounds to be mixed in must be specified. These sounds should
    ideally be at least as long as the input sounds to be transformed. Otherwise, the background
    sound will be repeated, which may sound unnatural.
    Note that the gain of the added noise is relative to the signal level in the input if the parameter noise_rms
    is set to "relative" (default option). This implies that if the input is completely silent, no noise will be added.
    Here are some examples of datasets that can be downloaded and used as background noise:
    * https://github.com/karolpiczak/ESC-50#download
    * https://github.com/microsoft/DNS-Challenge/
    """

    def __init__(
        self,
        sounds_path: Union[List[Path], List[str], Path, str],
        min_snr_db: float = 3.0,
        max_snr_db: float = 30.0,
        noise_rms: Literal["relative", "absolute"] = "relative",
        min_absolute_rms_db: float = -45.0,
        max_absolute_rms_db: float = -15.0,
        noise_transform: Optional[
            Callable[[NDArray[np.float32], int], NDArray[np.float32]]
        ] = None,
        p: float = 0.5,
        lru_cache_size: int = 2,
    ):
        """
        :param sounds_path: A path or list of paths to audio file(s) and/or folder(s) with
            audio files. Can be str or Path instance(s). The audio files given here are
            supposed to be background noises.
        :param min_snr_db: Minimum signal-to-noise ratio in dB. Is only used if noise_rms is set to "relative"
        :param max_snr_db: Maximum signal-to-noise ratio in dB. Is only used if noise_rms is set to "relative"
        :param noise_rms: Defines how the background noise will be added to the audio input. If the chosen
            option is "relative", the RMS of the added noise will be proportional to the RMS of
            the input sound. If the chosen option is "absolute", the background noise will have
            an RMS independent of the RMS of the input audio file. The default option is "relative".
        :param min_absolute_rms_db: Is only used if noise_rms is set to "absolute". It is
            the minimum RMS value in dB that the added noise can take. The lower the RMS is,
            the lower the added sound will be. Default: -45.0
        :param max_absolute_rms_db: Is only used if noise_rms is set to "absolute". It is
            the maximum RMS value in dB that the added noise can take. Note that this value
            can not exceed 0. Default: -15.0
        :param noise_transform: A callable waveform transform (or composition of transforms) that
            gets applied to the noise before it gets mixed in. The callable is expected
            to input audio waveform (numpy array) and sample rate (int).
        :param p: The probability of applying this transform
        :param lru_cache_size: Maximum size of the LRU cache for storing noise files in memory
        """
        super().__init__(p)
        self.sounds_path = sounds_path
        self.sound_file_paths = find_audio_files_in_paths(self.sounds_path)
        self.sound_file_paths = [str(p) for p in self.sound_file_paths]

        assert len(self.sound_file_paths) > 0

        if min_snr_db > max_snr_db:
            raise ValueError("min_snr_db must not be greater than max_snr_db")
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        if min_absolute_rms_db > max_absolute_rms_db:
            raise ValueError(
                "min_absolute_rms_db must not be greater than max_absolute_rms_db"
            )
        if max_absolute_rms_db > 0:
            raise ValueError("max_absolute_rms_db must not be greater than 0")
        self.min_absolute_rms_db = min_absolute_rms_db
        self.max_absolute_rms_db = max_absolute_rms_db

        self.noise_rms = noise_rms
        self.lru_cache_size = lru_cache_size
        self._load_sound = functools.lru_cache(maxsize=self.lru_cache_size)(
            AddBackgroundNoise._load_sound
        )
        self.noise_transform = noise_transform

    @staticmethod
    def _load_sound(file_path, sample_rate):
        return load_sound_file(file_path, sample_rate)

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["snr_db"] = random.uniform(self.min_snr_db, self.max_snr_db)
            self.parameters["rms_db"] = random.uniform(
                self.min_absolute_rms_db, self.max_absolute_rms_db
            )
            self.parameters["noise_file_path"] = random.choice(self.sound_file_paths)

            num_samples = len(samples)
            noise_sound, _ = self._load_sound(
                self.parameters["noise_file_path"], sample_rate
            )

            num_noise_samples = len(noise_sound)
            min_noise_offset = 0
            max_noise_offset = max(0, num_noise_samples - num_samples - 1)
            self.parameters["noise_start_index"] = random.randint(
                min_noise_offset, max_noise_offset
            )
            self.parameters["noise_end_index"] = (
                self.parameters["noise_start_index"] + num_samples
            )

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        noise_sound, _ = self._load_sound(
            self.parameters["noise_file_path"], sample_rate
        )
        noise_sound = noise_sound[
            self.parameters["noise_start_index"] : self.parameters["noise_end_index"]
        ]

        if self.noise_transform:
            noise_sound = self.noise_transform(noise_sound, sample_rate)

        noise_rms = calculate_rms(noise_sound)
        if noise_rms < 1e-9:
            warnings.warn(
                "The file {} is too silent to be added as noise. Returning the input"
                " unchanged.".format(self.parameters["noise_file_path"])
            )
            return samples

        clean_rms = calculate_rms(samples)

        if self.noise_rms == "relative":
            desired_noise_rms = calculate_desired_noise_rms(
                clean_rms, self.parameters["snr_db"]
            )

            # Adjust the noise to match the desired noise RMS
            noise_sound = noise_sound * (desired_noise_rms / noise_rms)

        if self.noise_rms == "absolute":
            desired_noise_rms_db = self.parameters["rms_db"]
            desired_noise_rms_amp = convert_decibels_to_amplitude_ratio(
                desired_noise_rms_db
            )
            gain = desired_noise_rms_amp / noise_rms
            noise_sound = noise_sound * gain

        # Repeat the sound if it shorter than the input sound
        num_samples = len(samples)
        while len(noise_sound) < num_samples:
            noise_sound = np.concatenate((noise_sound, noise_sound))

        if len(noise_sound) > num_samples:
            noise_sound = noise_sound[0:num_samples]

        # Return a mix of the input sound and the background noise sound
        return samples + noise_sound

    def __getstate__(self):
        state = self.__dict__.copy()
        warnings.warn(
            "Warning: the LRU cache of AddBackgroundNoise gets discarded when pickling"
            " it. E.g. this means the cache will not be used when using"
            " AddBackgroundNoise together with multiprocessing on Windows"
        )
        del state["_load_sound"]
        return state
