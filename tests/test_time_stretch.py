import numpy as np

from audiomentations import TimeStretch


class TestTimeStretch:
    def test_dynamic_length(self):
        samples = np.zeros((2048,), dtype=np.float32)
        sample_rate = 16000
        augmenter = TimeStretch(
            min_rate=0.8, max_rate=0.9, leave_length_unchanged=False, p=1.0
        )

        samples = augmenter(samples=samples, sample_rate=sample_rate)

        assert samples.dtype == np.float32
        assert len(samples) > 2048

    def test_fixed_length(self):
        samples = np.zeros((2048,), dtype=np.float32)
        sample_rate = 16000
        augmenter = TimeStretch(
            min_rate=0.8, max_rate=0.9, leave_length_unchanged=True, p=1.0
        )

        samples = augmenter(samples=samples, sample_rate=sample_rate)

        assert samples.dtype == np.float32
        assert len(samples) == 2048

    def test_multichannel(self):
        num_channels = 3
        samples = np.random.normal(0, 0.1, size=(num_channels, 5555)).astype(np.float32)
        sample_rate = 16000
        augmenter = TimeStretch(
            min_rate=0.8, max_rate=0.9, leave_length_unchanged=True, p=1.0
        )

        samples_out = augmenter(samples=samples, sample_rate=sample_rate)

        assert samples.dtype == samples_out.dtype
        assert samples.shape == samples_out.shape
        for i in range(num_channels):
            assert not np.allclose(samples[i], samples_out[i])
