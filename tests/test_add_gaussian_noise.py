import numpy as np
import pytest

from audiomentations import AddGaussianNoise


def test_gaussian_noise():
    samples = np.zeros((20,), dtype=np.float32)
    sample_rate = 16000
    augmenter = AddGaussianNoise(p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == np.float32
    assert not (float(np.sum(np.abs(samples))) == pytest.approx(0.0))


def test_gaussian_noise_stereo():
    samples = np.zeros((2, 2000), dtype=np.float32)
    sample_rate = 16000
    augmenter = AddGaussianNoise(p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == np.float32
    assert not (float(np.sum(np.abs(samples))) == pytest.approx(0.0))
