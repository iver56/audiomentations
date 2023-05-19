from typing import Optional
from audiomentations.core.utils import a_weighting_frequency_envelope
import pytest

import numpy as np

from audiomentations import AddColorNoise, NOISE_COLOR_DECAYS


def calculate_decay_rate(noise: np.ndarray, n_fft=8192):
    """Calculates the decay rate per octave of a noise PSD"""

    pxx = np.square(np.abs(np.fft.rfft(noise, n=n_fft)))
    pxx = pxx / np.sum(pxx)
    f = np.arange(len(pxx)).astype(np.float32)
    f[0] = 1e-10

    x = np.log10(f)

    # Linear regression using polyfit
    m, _ = np.polyfit(x[1:], 10 * np.log10(pxx[1:]), 1)

    return -m * np.log10(2)


class TestAddColorNoise:
    def test_add_colored_noise_defaults(self):
        np.random.seed(42)
        samples_in = np.random.normal(0, 1, size=1024).astype(np.float32)
        augmenter = AddColorNoise(p=1.0)
        rms_before = np.sqrt(np.mean(samples_in**2))
        samples_out = augmenter(samples=samples_in, sample_rate=16000)
        rms_after = np.sqrt(np.mean(samples_out**2))
        assert samples_out.shape == samples_in.shape
        assert samples_out.dtype == np.float32
        assert not (float(rms_after) == pytest.approx(0.0))
        assert rms_after != rms_before

    def test_add_colored_noise_defaults_stereo(self):
        np.random.seed(42)
        samples_in = np.random.normal(0, 1, size=(2, 1024)).astype(np.float32)
        augmenter = AddColorNoise(p=1.0)
        rms_before = np.sqrt(np.mean(samples_in**2))
        samples_out = augmenter(samples=samples_in, sample_rate=16000)
        rms_after = np.sqrt(np.mean(samples_out**2))

        assert samples_out.shape == samples_in.shape
        assert samples_out.dtype == np.float32
        assert not (float(rms_after) == pytest.approx(0.0))
        assert rms_after != rms_before

    @pytest.mark.parametrize("color", NOISE_COLOR_DECAYS.keys())
    def test_noise_has_zero_mean_unit_std(self, color):
        np.random.seed(42)
        samples_in = np.random.normal(0, 1, size=16000).astype(np.float32)
        f_decay = NOISE_COLOR_DECAYS[color]
        augmenter = AddColorNoise(
            min_f_decay=f_decay,
            max_f_decay=f_decay,
            min_snr_in_db=0.0,
            max_snr_in_db=0.0,
            p=1.0,
        )
        noise = augmenter(samples=samples_in, sample_rate=16000) - samples_in
        assert noise.dtype == np.float32
        assert np.mean(noise) == pytest.approx(0.0, abs=0.1)
        assert np.std(noise) == pytest.approx(1.0, abs=0.1)

    @pytest.mark.parametrize("color", NOISE_COLOR_DECAYS.keys())
    def test_add_colored_noise_snr(self, color):
        np.random.seed(42)
        samples_in = np.random.normal(0, 1, size=1024).astype(np.float32)

        augmenter = AddColorNoise(min_snr_in_db=5, max_snr_in_db=5, p=1.0)
        rms_in = np.sqrt(np.mean(samples_in**2))
        samples_out = augmenter(samples=samples_in, sample_rate=16000)
        rms_noise = np.sqrt(np.mean((samples_out - samples_in) ** 2))
        snr_db = 20 * np.log10(rms_in / rms_noise)

        assert samples_out.dtype == np.float32
        assert snr_db == pytest.approx(5.0, rel=1e-2)

    @pytest.mark.parametrize("color", NOISE_COLOR_DECAYS.keys())
    def test_add_colored_noise_colors(self, color):
        np.random.seed(42)
        samples_in = np.random.normal(0, 1, size=16000).astype(np.float32)
        f_decay = NOISE_COLOR_DECAYS[color]
        augmenter = AddColorNoise(min_f_decay=f_decay, max_f_decay=f_decay, p=1.0)
        samples_out = augmenter(samples=samples_in, sample_rate=16000)
        noise = samples_out - samples_in

        # Compute decay rate
        decay_rate = calculate_decay_rate(noise)

        # TODO: Decrease the value of abs= below. To do that, we need a better
        #       calculate_decay_rate function.
        assert decay_rate == pytest.approx(10 * np.log10(2) * f_decay, abs=2)
