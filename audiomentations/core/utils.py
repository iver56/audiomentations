import math
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Union, Tuple, Any

import numpy as np
import numpy_minmax
import numpy_rms
from numpy.typing import NDArray

SUPPORTED_EXTENSIONS = (
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
)

def format_args(args_dict: dict[str, Any]) -> str:
    formatted_args = []
    for k, v in args_dict.items():
        v_formatted = f"'{v}'" if isinstance(v, str) else str(v)
        formatted_args.append(f"{k}={v_formatted}")
    return ", ".join(formatted_args)

def find_audio_files(
    root_path,
    filename_endings=SUPPORTED_EXTENSIONS,
    traverse_subdirectories=True,
    follow_symlinks=True,
):
    """Return a list of paths to all audio files with the given extension(s) in a directory.
    Also traverses subdirectories by default.
    """
    file_paths = []

    for root, dirs, filenames in os.walk(root_path, followlinks=follow_symlinks):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)

            if filename.lower().endswith(filename_endings):
                file_paths.append(Path(file_path))
        if not traverse_subdirectories:
            # prevent descending into subfolders
            break

    return file_paths


def find_audio_files_in_paths(
    paths: Union[List[Path], List[str], Path, str],
    filename_endings=SUPPORTED_EXTENSIONS,
    traverse_subdirectories=True,
    follow_symlinks=True,
):
    """Return a list of paths to all audio files with the given extension(s) contained in the list or in its directories.
    Also traverses subdirectories by default.
    """

    file_paths = []

    if isinstance(paths, (list, tuple, set)):
        paths = list(paths)
    else:
        paths = [paths]

    for p in paths:
        if str(p).lower().endswith(SUPPORTED_EXTENSIONS):
            file_path = Path(os.path.abspath(p))
            file_paths.append(file_path)
        elif os.path.isdir(p):
            file_paths += find_audio_files(
                p,
                filename_endings=filename_endings,
                traverse_subdirectories=traverse_subdirectories,
                follow_symlinks=follow_symlinks,
            )
    return file_paths


def calculate_rms(samples):
    """Given a numpy array of audio samples, return its Root Mean Square (RMS)."""
    return np.mean(numpy_rms.rms(samples))


def calculate_rms_without_silence(samples: NDArray[np.float32], sample_rate: int):
    """
    This function returns the rms of a given noise whose silent periods have been removed. This ensures
    that the rms of the noise is not underestimated. Is most useful for short non-stationary noises.
    """

    window = int(0.025 * sample_rate)

    if samples.shape[-1] < window:
        return calculate_rms(samples)

    rms_all_windows = np.zeros(samples.shape[-1] // window)
    current_time = 0

    while current_time < samples.shape[-1] - window:
        rms_all_windows[current_time // window] += calculate_rms(
            samples[current_time : current_time + window]
        )
        current_time += window

    rms_threshold = np.max(rms_all_windows) / 25

    # The segments with a too low rms are identified and discarded
    rms_all_windows = rms_all_windows[rms_all_windows > rms_threshold]
    if rms_all_windows.shape[-1] > 0:
        # Beware that each window must have the same number of samples so that this calculation of the rms is valid.
        return calculate_rms(rms_all_windows)
    else:
        # Handle edge case: No windows remain. This can happen if there was just one window before discarding.
        return calculate_rms(samples)


def calculate_desired_noise_rms(clean_rms, snr):
    """
    Given the Root Mean Square (RMS) of a clean sound and a desired signal-to-noise ratio (SNR),
    calculate the desired RMS of a noise sound to be mixed in.
    Based on https://github.com/Sato-Kunihiko/audio-SNR/blob/8d2c933b6c0afe6f1203251f4877e7a1068a6130/create_mixed_audio_file.py#L20
    :param clean_rms: Root Mean Square (RMS) - a value between 0.0 and 1.0
    :param snr: Signal-to-Noise (SNR) Ratio in dB - typically somewhere between -20 and 60
    :return:
    """
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms


def convert_decibels_to_amplitude_ratio(decibels):
    return 10 ** (decibels / 20)


def is_waveform_multichannel(samples):
    """
    Return bool that answers the question: Is the given ndarray a multichannel waveform or not?
    :param samples: numpy ndarray
    :return:
    """
    return len(samples.shape) > 1


def is_spectrogram_multichannel(spectrogram):
    """
    Return bool that answers the question: Is the given ndarray a multichannel spectrogram?
    :param samples: numpy ndarray
    :return:
    """
    return len(spectrogram.shape) > 2 and spectrogram.shape[-1] > 1


def convert_float_samples_to_int16(y):
    """Convert floating-point numpy array of audio samples to int16."""
    if not issubclass(y.dtype.type, np.floating):
        raise ValueError("input samples not floating-point")
    return (y * np.iinfo(np.int16).max).astype(np.int16)


def convert_frequency_to_mel(f: float) -> float:
    """
    Convert f hertz to mels
    https://en.wikipedia.org/wiki/Mel_scale#Formula
    """
    return 2595.0 * math.log10(1.0 + f / 700.0)


def convert_mel_to_frequency(m: Union[float, np.array]) -> Union[float, np.array]:
    """
    Convert m mels to hertz
    https://en.wikipedia.org/wiki/Mel_scale#History_and_other_formulas
    """
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def a_weighting_frequency_envelope(n_fft, sample_rate):
    """
    Return the A-weighting frequency envelope for the given FFT size and sample rate.

    See the wikipedia article here:
    https://en.wikipedia.org/wiki/A-weighting#A
    """

    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    weighting = (
        (12194**2 * freqs**4)
        / (
            (freqs**2 + 20.6**2)
            * np.sqrt((freqs**2 + 107.7**2) * (freqs**2 + 737.9**2))
            * (freqs**2 + 12194**2)
        )
    ) + 2.00
    return weighting


@lru_cache(maxsize=8)
def get_crossfade_mask_pair(
    length: int, equal_energy=True
) -> Tuple[np.array, np.array]:
    """
    Equal-gain or equal-energy (within ~1%) cross-fade mask pair with
    smooth start and end.
    https://signalsmith-audio.co.uk/writing/2021/cheap-energy-crossfade/
    """
    x = np.linspace(0, 1, length, dtype=np.float32)
    x2 = 1 - x
    a = x * x2
    k = 1.4186 if equal_energy else -0.70912
    b = a * (1 + k * a)
    c = b + x
    d = b + x2
    fade_in = c * c
    fade_out = d * d
    return fade_in, fade_out


def get_max_abs_amplitude(samples: NDArray):
    min_amplitude, max_amplitude = numpy_minmax.minmax(samples)
    max_abs_amplitude = max(abs(min_amplitude), abs(max_amplitude))
    return max_abs_amplitude
