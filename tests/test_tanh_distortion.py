import unittest

import numpy as np

from audiomentations.augmentations.transforms import TanhDistortion
from audiomentations.core.composition import Compose


class TestTanhDistortion(unittest.TestCase):
    def test_single_channel(self):
        samples = np.zeros((2048,), dtype=np.float32)
        sample_rate = 16000
        augmenter = TanhDistortion(
            c1=2, c2=2, p=1.0
        )

        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), 2048)

    def test_multichannel(self):
        num_channels = 3
        samples = np.random.normal(0, 0.1, size=(num_channels, 5555)).astype(np.float32)
        sample_rate = 16000
        augmenter = TanhDistortion(
            c1=2, c2=2, p=1.0
        )

        samples_out = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, samples_out.dtype)
        self.assertEqual(samples.shape, samples_out.shape)
        for i in range(num_channels):
            assert not np.allclose(samples[i], samples_out[i])
