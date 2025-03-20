import numpy as np
import pytest

from audiomentations import Aliasing


def test_single_channel():
    samples = np.random.normal(0, 0.1, size=(2048,)).astype(np.float32)
    sample_rate = 16000
    augmenter = Aliasing(min_sample_rate=8000, max_sample_rate=32000, p=1.0)

    distorted_samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == distorted_samples.dtype
    assert samples.shape == distorted_samples.shape
    assert not np.array_equal(samples, distorted_samples)


def test_multichannel():
    num_channels = 3
    samples = np.random.normal(0, 0.1, size=(num_channels, 2048)).astype(np.float32)
    sample_rate = 16000
    augmenter = Aliasing(min_sample_rate=8000, max_sample_rate=32000, p=1.0)

    distorted_samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == distorted_samples.dtype
    assert samples.shape == distorted_samples.shape
    assert not np.array_equal(samples, distorted_samples)


def test_param_range():
    with pytest.raises(ValueError):
        Aliasing(min_sample_rate=0, max_sample_rate=6000, p=1.0)
    with pytest.raises(ValueError):
        Aliasing(min_sample_rate=8000, max_sample_rate=6000, p=1.0)
