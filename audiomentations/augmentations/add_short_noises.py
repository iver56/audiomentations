import functools
import random
import warnings
from pathlib import Path
from typing import Optional, List, Union, Callable

import numpy as np

from audiomentations.core.audio_loading_utils import load_sound_file
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    calculate_desired_noise_rms,
    calculate_rms,
    calculate_rms_without_silence,
    convert_decibels_to_amplitude_ratio,
    find_audio_files_in_paths,
)


class AddShortNoises(BaseWaveformTransform):
    """Mix in various (bursts of overlapping) sounds with random pauses between. Useful if your
    original sound is clean and you want to simulate an environment where short noises sometimes
    occur.
    A folder of (noise) sounds to be mixed in must be specified.
    """

    def __init__(
        self,
        sounds_path: Union[List[Path], List[str], Path, str],
        min_snr_in_db: float = 0.0,
        max_snr_in_db: float = 24.0,
        min_time_between_sounds: float = 4.0,
        max_time_between_sounds: float = 16.0,
        noise_rms: str = "relative",
        min_absolute_noise_rms_db: float = -50.0,
        max_absolute_noise_rms_db: float = -20,
        add_all_noises_with_same_level: bool = False,
        include_silence_in_noise_rms_estimation: bool = True,
        burst_probability: float = 0.22,
        min_pause_factor_during_burst: float = 0.1,
        max_pause_factor_during_burst: float = 1.1,
        min_fade_in_time: float = 0.005,
        max_fade_in_time: float = 0.08,
        min_fade_out_time: float = 0.01,
        max_fade_out_time: float = 0.1,
        signal_gain_in_db_during_noise: float = 0.0,
        noise_transform: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        p: float = 0.5,
        lru_cache_size: Optional[int] = 64,
    ):
        """
        :param sounds_path: A path or list of paths to audio file(s) and/or folder(s) with
            audio files. Can be str or Path instance(s). The audio files given here are
            supposed to be (short) noises.
        :param min_snr_in_db: Minimum signal-to-noise ratio in dB. A lower value means the added
            sounds/noises will be louder.
        :param max_snr_in_db: Maximum signal-to-noise ratio in dB. A lower value means the added
            sounds/noises will be louder.
        :param min_time_between_sounds: Minimum pause time between the added sounds/noises
        :param max_time_between_sounds: Maximum pause time between the added sounds/noises
        :param noise_rms: To choose in ["relative", "absolute", "relative_to_whole_input"].
            Defines how the noises will be added to the audio input.
            "relative": the RMS value of the added noise will be proportional to the RMS value of
            the input sound calculated only for the samples where the noise is added.
            "absolute": the added noises will have a RMS independent of the RMS of the input audio
            file.
            "relative_to_whole_input": the RMS of the added noises will be
            proportional to the RMS of the whole input sound.
        :param min_absolute_noise_rms_db: Is only used if noise_rms is set to "absolute". It is
            the minimum RMS value in dB that the added noise can take. The lower the RMS is, the
            lower will the added sound be.
        :param max_absolute_noise_rms_db: Is only used if noise_rms is set to "absolute". It is
            the maximum RMS value in dB that the added noise can take. Note that this value
            can not exceed 0.
        : param add_all_noises_with_same_level: add all the short noises with the same snr.
            The latter will be included between min_snr_in_db and max_snr_in_db. If
            noise_rms == "absolute", the RMS is used instead of the snr.This snr value
            will change each time the parameters of the transform are randomized.
        :param include_silence_in_noise_rms_estimation: A boolean. It chooses how the RMS of
            the noises to be added will be calculated. If this option is set to False, the silence
            in the noise files will be removed before the RMS calculation. It is useful for
            non-stationary noises where silent periods occur.
        :param burst_probability: The probability of adding an extra sound/noise that overlaps
        :param min_pause_factor_during_burst: Min value of how far into the current sound (as
            fraction) the burst sound should start playing. The value must be greater than 0.
        :param max_pause_factor_during_burst: Max value of how far into the current sound (as
            fraction) the burst sound should start playing. The value must be greater than 0.
        :param min_fade_in_time: Min sound/noise fade in time in seconds. Use a value larger
            than 0 to avoid a "click" at the start of the sound/noise.
        :param max_fade_in_time: Min sound/noise fade out time in seconds. Use a value larger
            than 0 to avoid a "click" at the start of the sound/noise.
        :param min_fade_out_time: Min sound/noise fade out time in seconds. Use a value larger
            than 0 to avoid a "click" at the end of the sound/noise.
        :param max_fade_out_time: Max sound/noise fade out time in seconds. Use a value larger
            than 0 to avoid a "click" at the end of the sound/noise.
        :param signal_gain_in_db_during_noise: Gain applied to the signal during a short noise.
            When fading the signal to the custom gain, the same fade times are used as
            for the noise, so it's essentially cross-fading. The default value (0.0) means
            the signal will not be gained. If set to a very low value, e.g. -100.0, this
            feature could be used for completely replacing the signal with the noise.
            This could be relevant in some use cases, for example:
            * replace the signal with another signal of a similar class (e.g. replace some
                speech with a cough)
            * simulate an ECG off-lead condition (electrodes are temporarily disconnected)
        :param noise_transform: A callable waveform transform (or composition of transforms) that
            gets applied to noises before they get mixed in.
        :param p: The probability of applying this transform
        :param lru_cache_size: Maximum size of the LRU cache for storing noise files in memory
        """
        super().__init__(p)
        self.sound_file_paths = find_audio_files_in_paths(sounds_path)
        self.sound_file_paths = [str(p) for p in self.sound_file_paths]
        assert len(self.sound_file_paths) > 0
        assert min_snr_in_db <= max_snr_in_db
        assert min_time_between_sounds <= max_time_between_sounds
        assert 0.0 < burst_probability <= 1.0
        if burst_probability == 1.0:
            assert (
                min_pause_factor_during_burst > 0.0
            )  # or else an infinite loop will occur
        assert 0.0 < min_pause_factor_during_burst <= 1.0
        assert max_pause_factor_during_burst > 0.0
        assert max_pause_factor_during_burst >= min_pause_factor_during_burst
        assert min_fade_in_time >= 0.0
        assert max_fade_in_time >= 0.0
        assert min_fade_in_time <= max_fade_in_time
        assert min_fade_out_time >= 0.0
        assert max_fade_out_time >= 0.0
        assert min_fade_out_time <= max_fade_out_time
        assert min_absolute_noise_rms_db <= max_absolute_noise_rms_db < 0
        assert type(include_silence_in_noise_rms_estimation) == bool
        assert noise_rms in ["relative", "absolute", "relative_to_whole_input"]

        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.min_time_between_sounds = min_time_between_sounds
        self.max_time_between_sounds = max_time_between_sounds
        self.burst_probability = burst_probability
        self.min_pause_factor_during_burst = min_pause_factor_during_burst
        self.max_pause_factor_during_burst = max_pause_factor_during_burst
        self.min_fade_in_time = min_fade_in_time
        self.max_fade_in_time = max_fade_in_time
        self.min_fade_out_time = min_fade_out_time
        self.max_fade_out_time = max_fade_out_time
        self.noise_rms = noise_rms
        self.min_absolute_noise_rms_db = min_absolute_noise_rms_db
        self.max_absolute_noise_rms_db = max_absolute_noise_rms_db
        self.include_silence_in_noise_rms_estimation = (
            include_silence_in_noise_rms_estimation
        )
        self.add_all_noises_with_same_level = add_all_noises_with_same_level
        self.signal_gain_in_db_during_noise = signal_gain_in_db_during_noise
        self.noise_transform = noise_transform
        self._load_sound = functools.lru_cache(maxsize=lru_cache_size)(
            AddShortNoises.__load_sound
        )

    @staticmethod
    def __load_sound(file_path, sample_rate):
        return load_sound_file(file_path, sample_rate)

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            input_sound_duration = len(samples) / sample_rate

            current_time = 0
            global_offset = random.uniform(
                -self.max_time_between_sounds, self.max_time_between_sounds
            )
            current_time += global_offset
            sounds = []

            snr_in_db = random.uniform(self.min_snr_in_db, self.max_snr_in_db)
            rms_in_db = random.uniform(
                self.min_absolute_noise_rms_db, self.max_absolute_noise_rms_db
            )

            while current_time < input_sound_duration:
                sound_file_path = random.choice(self.sound_file_paths)
                sound, _ = self.__load_sound(sound_file_path, sample_rate)
                sound_duration = len(sound) / sample_rate

                # Ensure that the fade time is not longer than the duration of the sound
                fade_in_time = min(
                    sound_duration,
                    random.uniform(self.min_fade_in_time, self.max_fade_in_time),
                )
                fade_out_time = min(
                    sound_duration,
                    random.uniform(self.min_fade_out_time, self.max_fade_out_time),
                )

                if not self.add_all_noises_with_same_level:
                    snr_in_db = random.uniform(self.min_snr_in_db, self.max_snr_in_db)
                    rms_in_db = random.uniform(
                        self.min_absolute_noise_rms_db, self.max_absolute_noise_rms_db
                    )

                sounds.append(
                    {
                        "fade_in_time": fade_in_time,
                        "start": current_time,
                        "end": current_time + sound_duration,
                        "fade_out_time": fade_out_time,
                        "file_path": sound_file_path,
                        "snr_in_db": snr_in_db,
                        "rms_in_db": rms_in_db,
                    }
                )

                # burst mode - add overlapping sounds
                while (
                    random.random() < self.burst_probability
                    and current_time < input_sound_duration
                ):
                    pause_factor = random.uniform(
                        self.min_pause_factor_during_burst,
                        self.max_pause_factor_during_burst,
                    )
                    pause_time = pause_factor * sound_duration
                    current_time = sounds[-1]["start"] + pause_time

                    if current_time >= input_sound_duration:
                        break

                    sound_file_path = random.choice(self.sound_file_paths)
                    sound, _ = self.__load_sound(sound_file_path, sample_rate)
                    sound_duration = len(sound) / sample_rate

                    fade_in_time = min(
                        sound_duration,
                        random.uniform(self.min_fade_in_time, self.max_fade_in_time),
                    )
                    fade_out_time = min(
                        sound_duration,
                        random.uniform(self.min_fade_out_time, self.max_fade_out_time),
                    )

                    if not self.add_all_noises_with_same_level:
                        snr_in_db = random.uniform(
                            self.min_snr_in_db, self.max_snr_in_db
                        )
                        rms_in_db = random.uniform(
                            self.min_absolute_noise_rms_db,
                            self.max_absolute_noise_rms_db,
                        )

                    sounds.append(
                        {
                            "fade_in_time": fade_in_time,
                            "start": current_time,
                            "end": current_time + sound_duration,
                            "fade_out_time": fade_out_time,
                            "file_path": sound_file_path,
                            "snr_in_db": snr_in_db,
                            "rms_in_db": rms_in_db,
                        }
                    )

                # wait until the last sound is done
                current_time += sound_duration

                # then add a pause
                pause_duration = random.uniform(
                    self.min_time_between_sounds, self.max_time_between_sounds
                )
                current_time += pause_duration

            self.parameters["sounds"] = sounds

    def apply(self, samples, sample_rate):
        num_samples = samples.shape[-1]
        noise_placeholder = np.zeros_like(samples)

        signal_mask = None
        gain_signal = self.signal_gain_in_db_during_noise != 0.0
        if gain_signal:
            signal_mask = np.zeros(shape=(num_samples,), dtype=np.float32)

        for sound_params in self.parameters["sounds"]:
            if sound_params["end"] < 0:
                # Skip a sound if it ended before the start of the input sound
                continue

            noise_samples, _ = self.__load_sound(sound_params["file_path"], sample_rate)

            if self.noise_transform:
                noise_samples = self.noise_transform(noise_samples, sample_rate)

            # Apply fade in and fade out
            noise_gain = np.ones_like(noise_samples)
            fade_in_time_in_samples = int(sound_params["fade_in_time"] * sample_rate)
            fade_in_mask = np.linspace(0.0, 1.0, num=fade_in_time_in_samples)
            fade_out_time_in_samples = int(sound_params["fade_out_time"] * sample_rate)
            fade_out_mask = np.linspace(1.0, 0.0, num=fade_out_time_in_samples)
            noise_gain[: fade_in_mask.shape[0]] = fade_in_mask
            noise_gain[-fade_out_mask.shape[0] :] = np.minimum(
                noise_gain[-fade_out_mask.shape[0] :], fade_out_mask
            )

            start_sample_index = int(sound_params["start"] * sample_rate)
            end_sample_index = start_sample_index + len(noise_samples)

            if start_sample_index < 0:
                # crop noise_samples: shave off a chunk in the beginning
                num_samples_to_shave_off = abs(start_sample_index)
                noise_samples = noise_samples[num_samples_to_shave_off:]
                noise_gain = noise_gain[num_samples_to_shave_off:]
                start_sample_index = 0

            if end_sample_index > num_samples:
                # crop noise_samples: shave off a chunk in the end
                num_samples_to_shave_off = end_sample_index - num_samples
                end_index = len(noise_samples) - num_samples_to_shave_off
                noise_samples = noise_samples[:end_index]
                noise_gain = noise_gain[:end_index]
                end_sample_index = num_samples

            # Gain here describes just the gain from the fade in and fade out.
            noise_samples = noise_samples * noise_gain

            if self.noise_rms == "relative_to_whole_input":
                clean_rms = calculate_rms(samples)
            else:
                clean_rms = calculate_rms(samples[start_sample_index:end_sample_index])

            if self.include_silence_in_noise_rms_estimation:
                noise_rms = calculate_rms(noise_samples)
            else:
                noise_rms = calculate_rms_without_silence(noise_samples, sample_rate)

            if noise_rms > 0:
                if self.noise_rms in ["relative", "relative_to_whole_input"]:
                    desired_noise_rms = calculate_desired_noise_rms(
                        clean_rms, sound_params["snr_in_db"]
                    )

                    # Adjust the noise to match the desired noise RMS
                    noise_samples = noise_samples * (desired_noise_rms / noise_rms)
                elif self.noise_rms == "absolute":
                    desired_noise_rms_db = sound_params["rms_in_db"]
                    desired_noise_rms_amp = convert_decibels_to_amplitude_ratio(
                        desired_noise_rms_db
                    )
                    gain = desired_noise_rms_amp / noise_rms
                    noise_samples = noise_samples * gain

                noise_placeholder[start_sample_index:end_sample_index] += noise_samples
                if gain_signal:
                    signal_mask[start_sample_index:end_sample_index] = np.maximum(
                        signal_mask[start_sample_index:end_sample_index],
                        noise_gain,
                    )

        if gain_signal:
            # Gain the original signal before mixing in the noises
            signal_mask *= self.signal_gain_in_db_during_noise
            signal_mask = convert_decibels_to_amplitude_ratio(signal_mask)
            return samples * signal_mask + noise_placeholder
        else:
            # Return a mix of the input sound and the added sounds
            return samples + noise_placeholder

    def __getstate__(self):
        state = self.__dict__.copy()
        warnings.warn(
            "Warning: the LRU cache of AddShortNoises gets discarded when pickling it."
            " E.g. this means the cache will not be used when using AddShortNoises together"
            " with multiprocessing on Windows"
        )
        del state["_load_sound"]
        return state
