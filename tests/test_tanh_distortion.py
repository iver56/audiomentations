import unittest

import numpy as np

from audiomentations.augmentations.transforms import TanhDistortion
from audiomentations.core.composition import Compose
from audiomentations.core.utils import calculate_rms


class TestTanhDistortion(unittest.TestCase):
    def test_single_channel(self):
        samples = np.zeros((2048,), dtype=np.float32)
        sample_rate = 16000
        augmenter = TanhDistortion(
           min_distortion_gain=1., max_distortion_gain=2.0, p=1.0
        )

        distorted_samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, distorted_samples.dtype)
        self.assertEqual(samples.shape, distorted_samples.shape)
        self.assertEqual(calculate_rms(samples), calculate_rms(distorted_samples))

    def test_multichannel(self):
        num_channels = 3
        samples = np.random.normal(0, 0.1, size=(num_channels, 5555)).astype(np.float32)
        sample_rate = 16000
        augmenter = TanhDistortion(
           min_distortion_gain=1., max_distortion_gain=2.0, p=1.0
        )

        distorted_samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, distorted_samples.dtype)
        self.assertEqual(samples.shape, distorted_samples.shape)
        self.assertEqual(calculate_rms(samples), calculate_rms(distorted_samples))
        for i in range(num_channels):
            assert not np.allclose(samples[i], distorted_samples[i])
