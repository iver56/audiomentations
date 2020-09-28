import os
from pathlib import Path

import numpy as np

AUDIO_FILENAME_ENDINGS = (".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav")


def get_file_paths(
    root_path,
    filename_endings=AUDIO_FILENAME_ENDINGS,
    traverse_subdirectories=True,
):
    """Return a list of paths to all files with the given in a directory
    Also traverses subdirectories by default.
    """
    file_paths = []

    for root, dirs, filenames in os.walk(root_path):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)

            if filename.endswith(filename_endings):
                file_paths.append(Path(file_path))
        if not traverse_subdirectories:
            # prevent descending into subfolders
            break

    return file_paths


def calculate_rms(samples):
    """Given a numpy array of audio samples, return its Root Mean Square (RMS)."""
    return np.sqrt(np.mean(np.square(samples), axis=-1))


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
    noise_rms = clean_rms / (10 ** a)
    return noise_rms


def convert_decibels_to_amplitude_ratio(decibels):
    return 10 ** (decibels / 20)


def is_multichannel(samples):
    """

    :param samples:
    :return:
    """
    return len(samples.shape) > 1


def convert_float_samples_to_int16(y):
    """Convert floating-point numpy array of audio samples to int16."""
    if not issubclass(y.dtype.type, np.floating):
        raise ValueError("input samples not floating-point")
    return (y * np.iinfo(np.int16).max).astype(np.int16)
