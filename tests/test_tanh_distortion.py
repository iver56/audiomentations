import numpy as np
import pytest

from audiomentations import TanhDistortion
from audiomentations.core.utils import calculate_rms


def test_single_channel():
    samples = np.random.normal(0, 0.1, size=(20480,)).astype(np.float32)
    sample_rate = 16000
    augmenter = TanhDistortion(min_distortion=0.2, max_distortion=0.6, p=1.0)

    distorted_samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == distorted_samples.dtype
    assert samples.shape == distorted_samples.shape
    assert np.amax(distorted_samples) < np.amax(samples)
    assert calculate_rms(distorted_samples) == pytest.approx(
        calculate_rms(samples), abs=1e-3
    )


def test_multichannel():
    num_channels = 3
    samples = np.random.normal(0, 0.1, size=(num_channels, 5555)).astype(np.float32)
    sample_rate = 16000
    augmenter = TanhDistortion(min_distortion=0.05, max_distortion=0.6, p=1.0)

    distorted_samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == distorted_samples.dtype
    assert samples.shape == distorted_samples.shape
    for i in range(num_channels):
        assert not np.allclose(samples[i], distorted_samples[i])
        assert calculate_rms(distorted_samples[i]) == pytest.approx(
            calculate_rms(samples[i]), abs=1e-3
        )
