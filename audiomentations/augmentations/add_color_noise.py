import random

import numpy as np
import scipy.signal as sp
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import a_weighting_frequency_envelope
from audiomentations.core.utils import calculate_desired_noise_rms, calculate_rms

NOISE_COLOR_DECAYS = {
    "pink": -1.0 * 10 * np.log10(2),
    "brown": -2.0 * 10 * np.log10(2),
    "brownian": -2.0 * 10 * np.log10(2),
    "red": -2.0 * 10 * np.log10(2),
    "blue": 1.0 * 10 * np.log10(2),
    "azure": 1.0 * 10 * np.log10(2),
    "violet": 2.0 * 10 * np.log10(2),
    "white": 0.0 * 10 * np.log10(2),
}


def decay_to_beta(decay: float) -> float:
    """
    Converts a decay given in dB/octave to the beta exponent
    in the PSD logarithmic decay function:

    logarithmic_decay = \sqrt{\frac{1}{f^{\beta}}}

    where logarithmic_decay is the rate of change of the PSD
    over frequency (in log-space).
    """

    return decay / (-10.0) / np.log10(2.0)


def generate_decaying_white_noise(
    size,
    beta,
    sample_rate,
    apply_a_weighting=False,
    n_fft=64,
    in_db_per_octave=True,
):
    """
    Generates a white noise signal decaying linearly by 1/f^beta
    (when in_db_per_octave==False) or changing (decaying or increasing)
    linearly by beta dB/octave (when in_db_per_octave==False).

    The values for beta are given below:

    | Colour         |  in_db_per_octave=False | in_db_per_octave=True |
    |----------------+-------------------------+-----------------------|
    | pink           |                     1.0 |                 -3.01 |
    | brown/brownian |                     2.0 |                 -6.02 |
    | red            |                     2.0 |                 -6.02 |
    | blue           |                    -1.0 |                  3.01 |
    | azure          |                    -1.0 |                  3.01 |
    | violet         |                    -2.0 |                  6.02 |
    | white          |                     0.0 |                   0.0 |

    Note that you can get away with low n_fft (e.g. 128 points) values
    if you are not using a_weighting, but keep it higher otherwise.
    """

    sig = np.random.normal(0, 1, size=size)

    if beta == 0.0 and not apply_a_weighting:
        # No decay, return white noise
        return sig.astype(np.float32)

    # Compute the decay in fft domain (ignore phase)
    f = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    f[0] = 1
    decay = np.ones(n_fft // 2 + 1, dtype=complex)

    # If beta is given as decay in db/octave, convert it to a
    # decay exponent.
    if in_db_per_octave:
        beta = decay_to_beta(beta)

    # Decay is in PSD, for magnitude, take sqrt and add random phase
    decay = np.sqrt(1 / f**beta) * np.exp(
        1j * np.random.uniform(0, 2 * np.pi, len(decay))
    )

    # Optionally apply a-weighting
    if apply_a_weighting:
        weighting = a_weighting_frequency_envelope(n_fft, sample_rate)
        decay *= weighting

    # Compute the impulse response of the decay
    decay_ir = np.fft.irfft(decay)

    # Convolve the white noise with the decay impulse response

    # Calculate number of channels
    if len(sig.shape) > 1:
        n_channels = sig.shape[0]
        fsig = np.zeros_like(sig).astype(np.float32)
        for chn_idx in range(n_channels):
            fsig[chn_idx] = sp.oaconvolve(sig[chn_idx], decay_ir, "same")
    else:
        n_channels = 1
        fsig = sp.oaconvolve(sig, decay_ir, "same")

    # Normalize to unit energy
    fsig /= np.sqrt(np.mean(fsig**2))

    return fsig.astype(np.float32)


class AddColorNoise(BaseWaveformTransform):
    """
    Adds noise to the input samples with a decaying frequency spectrum resulting in "color" noise.
    For info, more see the Wikipedia article here: https://en.wikipedia.org/wiki/Colors_of_noise
    """

    supports_multichannel = True

    def __init__(
        self,
        min_snr_db: float = 5.0,
        max_snr_db: float = 40.0,
        min_f_decay: float = -6.0,
        max_f_decay: float = 6.0,
        p_apply_a_weighting: float = 0.0,
        p: float = 0.5,
        n_fft: int = 128,
    ):
        """
        :param min_snr_db: Minimum signal-to-noise ratio in dB. A lower number means more noise.
        :param max_snr_db: Maximum signal-to-noise ratio in dB. A greater number means less noise.
        :param min_f_decay: Minimum frequency decay in dB per octave.
        :param max_f_decay: Maximum frequency decay in dB per octave.
        :param p: The probability of applying this transform
        :param p_apply_a_weighting: The probability of applying A-weighting to the noise. Useful for
                                    generating "grey" noise. See here: https://en.wikipedia.org/wiki/Grey_noise
        :param n_fft: Noise is 'colorized' by applying a decay in the frequency domain. The decay envelope
                      is generated in the fft domain. This parameter controls the number of fft points. For
                      unweighted colored noise, you should use low n_fft (e.g. 128 points) values, but
                      you should probably increase it higher otherwise.

        f_decay is the decay applied to the spectral density of the noise. The power spectral density per
        unit of bandwidth becomes proportional to 1/f^f_decay. For example, a decay of 0 means that the
        spectral density is flat, equivalent to applying white noise. Below are common noise colors and
        their corresponding f_decay values:

        | Colour   | f_decay (dB/octave)   |
        |----------+-----------------------|
        | pink     |                 -3.01 |
        | brown    |                 -6.02 |
        | Brownian |                 -6.02 |
        | red      |                 -6.02 |
        | blue     |                  3.01 |
        | azure    |                  3.01 |
        | violet   |                  6.02 |
        | white    |                   0.0 |
        For information about those values, see here: https://en.wikipedia.org/wiki/Colors_of_noise

        Additionally, p_apply_a_weighting gives the probability the noise to be weighted by a psychoacoustic
        equal loudness curve, which results in grey-noise when f_decay = 0.0.
        """
        super().__init__(p)
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.min_f_decay = min_f_decay
        self.max_f_decay = max_f_decay
        self.p_apply_a_weighting = p_apply_a_weighting
        self.n_fft = n_fft

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            # Pick SNR in Decibel scale
            snr = random.uniform(self.min_snr_db, self.max_snr_db)

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
            self.parameters["desired_noise_rms"] = float(desired_noise_rms)
            self.parameters["f_decay"] = f_decay
            self.parameters["apply_a_weighting"] = apply_a_weighting

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        desired_noise_rms = self.parameters["desired_noise_rms"]

        if samples.ndim == 1:
            n_channels = 1
        else:
            n_channels = samples.shape[0]

        noise_with_unit_rms = generate_decaying_white_noise(
            size=samples.shape,
            beta=self.parameters["f_decay"],
            sample_rate=sample_rate,
            apply_a_weighting=self.parameters["apply_a_weighting"],
            n_fft=self.n_fft,
        )

        if n_channels > 1:
            return samples + noise_with_unit_rms * desired_noise_rms
        else:
            return samples + noise_with_unit_rms * desired_noise_rms
