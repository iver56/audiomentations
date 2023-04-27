import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from audiomentations import Mp3Compression
from audiomentations.core.audio_loading_utils import load_sound_file
from demo.demo import DEMO_DIR


def fast_autocorr(original, delayed, t=1):
    if t == 0:
        return np.corrcoef([original[::4], delayed[::4]])[1, 0]
    elif t < 0:
        return np.corrcoef([original[-t::4], delayed[:t:4]])[1, 0]
    else:
        return np.corrcoef([original[:-t:4], delayed[t::4]])[1, 0]


def calculate_delay(
    original_signal,
    delayed_signal,
    min_lag_in_samples,
    max_lag_in_samples,
    debug=False,
):
    coefs = []
    for lag in range(min_lag_in_samples, max_lag_in_samples):
        correlation_coef = fast_autocorr(original_signal, delayed_signal, t=lag)
        coefs.append(correlation_coef)

    max_coef_index = int(np.argmax(coefs))

    if coefs[max_coef_index] < 0.3:
        print(
            "Warning: Max coefficient is low ({})! Please check if the delay is out of bounds".format(
                coefs[max_coef_index]
            )
        )
    delay = max_coef_index + min_lag_in_samples

    if debug:
        plt.plot(coefs)
        plt.show()

    return delay


def mse(signal1, signal2):
    """Mean squared error"""
    return np.mean((signal1 - signal2) ** 2)


class TestMp3Compression:
    @pytest.mark.parametrize(
        "backend",
        ["pydub", "lameenc"],
    )
    @pytest.mark.parametrize(
        "shape",
        [(44100,), (1, 22049), (2, 10000)],
    )
    def test_apply_mp3_compression(self, backend: str, shape: tuple):
        samples_in = np.random.normal(0, 1, size=shape).astype(np.float32)
        sample_rate = 44100
        augmenter = Mp3Compression(
            p=1.0, min_bitrate=48, max_bitrate=48, backend=backend
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        assert len(shape) == len(samples_out.shape)
        assert samples_out.dtype == np.float32
        assert samples_out.shape[-1] >= shape[-1]
        assert samples_out.shape[-1] < shape[-1] + 3000
        if len(shape) == 2:
            assert samples_out.shape[0] == shape[0]

    @pytest.mark.parametrize(
        "backend",
        ["pydub", "lameenc"],
    )
    @pytest.mark.parametrize(
        "shape",
        [(16000,), (1, 12049), (2, 5000)],
    )
    def test_apply_mp3_compression_low_bitrate(self, backend: str, shape: tuple):
        samples_in = np.random.normal(0, 1, size=shape).astype(np.float32)
        sample_rate = 16000
        augmenter = Mp3Compression(p=1.0, min_bitrate=8, max_bitrate=8, backend=backend)

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        assert len(shape) == len(samples_out.shape)
        assert samples_out.dtype == np.float32
        assert samples_out.shape[-1] >= shape[-1]
        assert samples_out.shape[-1] < shape[-1] + 3100
        if len(shape) == 2:
            assert samples_out.shape[0] == shape[0]

    def test_invalid_argument_combination(self):
        with pytest.raises(AssertionError):
            _ = Mp3Compression(min_bitrate=400, max_bitrate=800)

        with pytest.raises(AssertionError):
            _ = Mp3Compression(min_bitrate=2, max_bitrate=4)

        with pytest.raises(AssertionError):
            _ = Mp3Compression(min_bitrate=64, max_bitrate=8)

    def test_too_loud_input(self):
        """Check that we avoid wrap distortion if input is too loud"""
        samples_in, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "perfect-alley1.ogg"),
            sample_rate=None,
            mono=True,
        )
        samples_in = samples_in[..., 0:196000]
        samples_in *= 10
        augmenter = Mp3Compression(
            min_bitrate=320, max_bitrate=320, backend="lameenc", p=1.0
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        samples_out = samples_out[..., 0 : samples_in.shape[-1]]
        assert samples_out.dtype == np.float32

        # Apply delay compensation, so we can compare output with input sample by sample
        delay = calculate_delay(
            samples_in,
            samples_out,
            min_lag_in_samples=0,
            max_lag_in_samples=3100,
        )
        snippet_length = 42000
        delay_compensated_samples_out = samples_out[..., delay:delay+snippet_length]
        samples_in = samples_in[..., :snippet_length]
        mse_value = mse(samples_in, delay_compensated_samples_out)

        assert mse_value < 0.00001
