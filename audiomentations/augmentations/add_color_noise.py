import random
from typing import Optional

import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import calculate_desired_noise_rms, calculate_rms

import scipy.signal as sp

NOISE_COLOR_DECAYS = {
    "pink": 1.0,
    "brown": 2.0,
    "brownian": 2.0,
    "red": 2.0,
    "blue": -1.0,
    "azure": -1.0,
    "violet": -2.0,
    "white": 0.0,
}


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


def generate_decaying_white_noise(
    n_samples, f_decay, sample_rate, apply_a_weighting=False, n_fft=64
):
    """
    Generates a white noise signal decaying by 1/f^beta.

    Note that you can get away with low n_fft (e.g. 128 points) values
    if you are not using a_weighting, but keep it higher otherwise.
    """
    # Create n_samples of white noise
    sig = np.random.randn(n_samples)

    if f_decay == 0.0:
        # No decay, return white noise
        return sig.astype(np.float32)

    # Compute the decay in fft domain (ignore phase)
    f = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    f[0] = 1
    decay = np.ones(n_fft // 2 + 1, dtype=complex)

    # Decay is in PSD, for magnitude, take sqrt and add random phase
    decay = np.sqrt(1 / f**f_decay) * np.exp(
        1j * np.random.uniform(0, 2 * np.pi, len(decay))
    )

    # Optionally apply a-weighting
    if apply_a_weighting:
        weighting = a_weighting_frequency_envelope(n_fft, sample_rate)
        decay *= weighting

    # Compute the impulse response of the decay
    decay_ir = np.fft.irfft(decay)

    # Convolve the white noise with the decay impulse response
    fsig = sp.oaconvolve(sig, decay_ir, "same")

    # Normalize to unit energy
    fsig /= np.sqrt(np.mean(fsig**2))

    return fsig.astype(np.float32)


class AddColorNoise(BaseWaveformTransform):
    """
    Adds noise to the input samples with a decaying frequency spectrum resulting in "color" noise.
    For more see the Wikipedia article here: https://en.wikipedia.org/wiki/Colors_of_noise
    """

    supports_multichannel = True

    def __init__(
        self,
        min_snr_in_db: float = 5.0,
        max_snr_in_db: float = 40.0,
        min_f_decay: float = -2.0,
        max_f_decay: float = 2.0,
        p_apply_a_weighting: float = 0.0,
        p: float = 0.5,
        n_fft: int = 128,
    ):
        """
        :param min_snr_in_db: Minimum signal-to-noise ratio in dB. A lower number means more noise.
        :param max_snr_in_db: Maximum signal-to-noise ratio in dB. A greater number means less noise.
        :param min_f_decay: Minimum frequency decay in dB.
        :param max_f_decay: Maximum frequency decay in dB.
        :param p: The probability of applying this transform
        :param p_apply_a_weighting: The probability of applying A-weighting to the noise.
        :param n_fft: Noise is 'colorized' by applying a decay in the frequency domain. The decay envelope
                      is generated in the fft domain. This parameter controls the number of fft points. For
                      unweighted colored noise, you should use low n_fft (e.g. 128 points) values, but
                      you should probably increase it higher otherwise.

        f_decay is the decay applied to the spectral density of the noise. The power spectral density per
        unit of bandwidth becomes proportional to 1/f^f_decay. For example, a decay of 0 means that the
        spectral density is flat, equivalent to applying white noise. Below are common noise colors and
        their corresponding f_decay values:

        Color          f_decay
        ----------------------
        white              0.0
        pink               1.0
        brown(ian)/red     2.0
        blue/azure        -1.0
        violet            -2.0

        Additionally, p_apply_a_weighting gives the probability the noise to be weighted by a psychoacoustic
        equal loudness curve, which results in grey-noise when f_decay = 0.0.
        """
        super().__init__(p)
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.min_f_decay = min_f_decay
        self.max_f_decay = max_f_decay
        self.p_apply_a_weighting = p_apply_a_weighting
        self.n_fft = n_fft

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            # Pick SNR in decibel scale
            snr = random.uniform(self.min_snr_in_db, self.max_snr_in_db)

            # Pick f_decay
            f_decay = random.uniform(self.min_f_decay, self.max_f_decay)

            # Pick whether to apply A-weighting
            apply_a_weighting = random.random() < self.p_apply_a_weighting

            # Calculate desired noise rms
            clean_rms = calculate_rms(samples)
            desired_noise_rms = calculate_desired_noise_rms(
                clean_rms=clean_rms, snr=snr
            )

            # Set the parameters
            self.parameters["desired_noise_rms"] = desired_noise_rms
            self.parameters["f_decay"] = f_decay
            self.parameters["apply_a_weighting"] = apply_a_weighting

    def apply(self, samples: np.ndarray, sample_rate: int):
        desired_noise_rms = self.parameters["desired_noise_rms"]

        noise_with_unit_rms = generate_decaying_white_noise(
            n_samples=len(samples),
            f_decay=self.parameters["f_decay"],
            sample_rate=sample_rate,
            apply_a_weighting=self.parameters["apply_a_weighting"],
            n_fft=self.n_fft,
        )

        return samples + noise_with_unit_rms * desired_noise_rms
