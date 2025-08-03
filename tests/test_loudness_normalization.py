import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from audiomentations import LoudnessNormalization


def test_loudness_normalization():
    samples = np.random.uniform(low=-0.2, high=-0.001, size=(8000,)).astype(np.float32)
    sample_rate = 16000

    augment = LoudnessNormalization(min_lufs=-32, max_lufs=-12, p=1.0)
    processed_samples = augment(samples=samples, sample_rate=sample_rate)
    gain_factors = processed_samples / samples
    assert np.amin(gain_factors) == pytest.approx(np.amax(gain_factors))
    assert processed_samples.dtype == np.float32


def test_loudness_normalization_digital_silence():
    samples = np.zeros(8000, dtype=np.float32)
    sample_rate = 16000

    augment = LoudnessNormalization(min_lufs=-32, max_lufs=-12, p=1.0)
    processed_samples = augment(samples=samples, sample_rate=sample_rate)
    assert_almost_equal(processed_samples, np.zeros(8000, dtype=np.float32))
    assert processed_samples.dtype == np.float32


def test_loudness_normalization_too_short_input():
    samples = np.random.uniform(low=-0.2, high=-0.001, size=(800,)).astype(np.float32)
    sample_rate = 16000

    augment = LoudnessNormalization(min_lufs=-32, max_lufs=-12, p=1.0)
    with pytest.raises(ValueError):
        _ = augment(samples=samples, sample_rate=sample_rate)


def test_loudness_normalization_multichannel():
    samples = np.random.uniform(low=-0.2, high=-0.001, size=(3, 8000)).astype(
        np.float32
    )
    sample_rate = 16000

    augment = LoudnessNormalization(min_lufs=-32, max_lufs=-12, p=1.0)
    processed_samples = augment(samples=samples, sample_rate=sample_rate)
    gain_factors = processed_samples / samples
    assert np.amin(gain_factors) == pytest.approx(np.amax(gain_factors))
    assert processed_samples.dtype == np.float32


def test_validation():
    with pytest.raises(ValueError):
        LoudnessNormalization(min_lufs=-12.0, max_lufs=-32.0)
