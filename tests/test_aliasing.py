import numpy as np
import pytest

from audiomentations import Aliasing


class TestAliasing:
    def test_single_channel(self):
        samples = np.random.normal(0, 0.1, size=(2048,)).astype(np.float32)
        sample_rate = 16000
        augmenter = Aliasing(min_sample_rate=8000, max_sample_rate=32000, p=1.0)

        distorted_samples = augmenter(samples=samples, sample_rate=sample_rate)

        assert samples.dtype == distorted_samples.dtype
        assert samples.shape == distorted_samples.shape
        assert len(distorted_samples) == len(samples)

    def test_multichannel(self):
        num_channels = 3
        samples = np.random.normal(0, 0.1, size=(num_channels, 2048)).astype(np.float32)
        sample_rate = 16000
        augmenter = Aliasing(min_sample_rate=8000, max_sample_rate=32000, p=1.0)

        distorted_samples = augmenter(samples=samples, sample_rate=sample_rate)

        assert samples.dtype == distorted_samples.dtype
        assert samples.shape == distorted_samples.shape
        assert len(distorted_samples) == len(samples)

