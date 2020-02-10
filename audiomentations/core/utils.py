import math
import os


def read_dir(path):
    ir_list = os.listdir(path)
    return [os.path.join(path, ir) for ir in ir_list]


def calculate_rms(samples):
    """Given a numpy array of audio samples, return its RMS power level."""
    chunk = pow(abs(samples), 2)
    return math.sqrt(chunk.mean())


def calculate_desired_noise_rms(clean_rms, snr):
    """
    Given the Root Mean Square (RMS) of a clean sound and a desired signal-to-noise ratio (SNR),
    calculate the desired RMS of a noise sound to be mixed in.

    Based on https://github.com/Sato-Kunihiko/audio-SNR/blob/8d2c933b6c0afe6f1203251f4877e7a1068a6130/create_mixed_audio_file.py#L20
    :param clean_rms: Root Mean Square (RMS) - a value between 0.0 and 1.0
    :param snr: Signal-to-Noise (SNR) Ratio in dB - typically somewhere between 0 and 80
    :return:
    """
    a = float(snr) / 20
    noise_rms = clean_rms / (10 ** a)
    return noise_rms
