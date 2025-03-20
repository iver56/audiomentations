import numpy as np
import pytest

from audiomentations import BitCrush


def test_single_channel():
    samples = np.random.normal(0, 0.1, size=(2048,)).astype(np.float32)
    sample_rate = 16000
    augmenter = BitCrush(min_bit_depth=3, max_bit_depth=6, p=1.0)

    distorted_samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == distorted_samples.dtype
    assert samples.shape == distorted_samples.shape
    assert 2 ** augmenter.parameters["bit_depth"] + 1 > len(
        np.unique(np.round(distorted_samples, 5))
    )


def test_multichannel():
    num_channels = 3
    samples = np.random.normal(0, 0.1, size=(num_channels, 2048)).astype(np.float32)
    sample_rate = 16000
    augmenter = BitCrush(min_bit_depth=3, max_bit_depth=6, p=1.0)

    distorted_samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == distorted_samples.dtype
    assert samples.shape == distorted_samples.shape
    assert 2 ** augmenter.parameters["bit_depth"] + 1 > len(
        np.unique(np.round(distorted_samples, 5))
    )


def test_param_range():
    with pytest.raises(ValueError):
        BitCrush(min_bit_depth=0, max_bit_depth=6, p=1.0)
    with pytest.raises(ValueError):
        BitCrush(min_bit_depth=7, max_bit_depth=64, p=1.0)
    with pytest.raises(ValueError):
        BitCrush(min_bit_depth=8, max_bit_depth=6, p=1.0)
