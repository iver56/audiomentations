import unittest

import numpy as np

from audiomentations.augmentations.transforms import AddGaussianSNR
from audiomentations.core.composition import Compose


class TestGaussianSNR(unittest.TestCase):
    def test_gaussian_noise(self):
        sample_len = 1024
        # samples = np.ones((1024,), dtype=np.float32)
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([AddGaussianSNR(min_SNR=0.5, max_SNR=1.0, p=1.0)])
        std_in = np.mean(np.abs(samples_in))
        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        std_out = np.mean(np.abs(samples_out))
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertNotAlmostEqual(std_out, 0.0)
        self.assertGreater(std_out, std_in)
