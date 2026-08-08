import numpy as np
import pytest

from audiomentations import SpectralGating


def test_spectral_gating():
    samples = np.random.random((2048,))
    sample_rate = 16000
    augmenter = SpectralGating(p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == np.float32
    assert samples.shape == (2048,)
    assert np.max(np.abs(samples)) > 0.0
    assert not np.isnan(samples).any()
    assert not np.isinf(samples).any()


def test_spectral_gating_stereo():
    samples = np.random.random((2, 2048,))
    sample_rate = 16000
    augmenter = SpectralGating(p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == np.float32
    assert samples.shape == (2, 2048)
    assert np.max(np.abs(samples)) > 0.0
    assert not np.isnan(samples).any()
    assert not np.isinf(samples).any()


