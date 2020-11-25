import unittest

import numpy as np

from audiomentations.augmentations.transforms import AddGaussianNoise
from audiomentations.core.composition import Compose


class TestGaussianNoise(unittest.TestCase):
    def test_gaussian_noise(self):
        samples = np.zeros((20,), dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose([AddGaussianNoise(p=1.0)])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertNotAlmostEqual(float(np.sum(np.abs(samples))), 0.0)

    def test_gaussian_noise_stereo(self):
        samples = np.zeros((2, 2000), dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose([AddGaussianNoise(p=1.0)])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertNotAlmostEqual(float(np.sum(np.abs(samples))), 0.0)
