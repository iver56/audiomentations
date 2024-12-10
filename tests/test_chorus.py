import numpy as np
import pytest

from audiomentations import Chorus


class TestChorus:
    def test_single_channel(self):
        samples = np.random.normal(0, 0.1, size=(2048,)).astype(np.float32)
        sample_rate = 16000
        augmenter = Chorus(min_chorus_rate=0.1, max_chorus_rate=0.6, p=1.0)

        distorted_samples = augmenter(samples=samples, sample_rate=sample_rate)

        assert samples.dtype == distorted_samples.dtype
        assert samples.shape == distorted_samples.shape
        assert not np.array_equal(samples, distorted_samples)

    def test_multichannel(self):
        num_channels = 3
        samples = np.random.normal(0, 0.1, size=(num_channels, 2048)).astype(np.float32)
        sample_rate = 16000
        augmenter = Chorus(min_chorus_rate=0.1, max_chorus_rate=0.6, p=1.0)

        distorted_samples = augmenter(samples=samples, sample_rate=sample_rate)

        assert samples.dtype == distorted_samples.dtype
        assert samples.shape == distorted_samples.shape
        assert not np.array_equal(samples, distorted_samples)

    def test_param_range(self):
        with pytest.raises(ValueError):
            Chorus(min_chorus_rate=0.6, max_chorus_rate=0.1, p=1.0)
        with pytest.raises(ValueError):
            Chorus(min_chorus_depth_ms=20, max_chorus_depth_ms=10, p=1.0)
        with pytest.raises(ValueError):
            Chorus(min_offset_ms=40, max_offset_ms=20, p=1.0)
