import unittest

import numpy as np

from audiomentations.augmentations.transforms import Normalize
from audiomentations.core.composition import Compose


class TestNormalize(unittest.TestCase):
    def test_normalize_positive_peak(self):
        samples = np.array([0.5, 0.6, -0.2, 0.0], dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose([Normalize(p=1.0)])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(np.amax(samples), 1.0)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), 4)

    def test_normalize_negative_peak(self):
        samples = np.array([0.5, 0.6, -0.8, 0.0], dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose([Normalize(p=1.0)])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(np.amin(samples), -1.0)
        self.assertEqual(samples[-1], 0.0)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), 4)

    def test_normalize_all_zeros(self):
        samples = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose([Normalize(p=1.0)])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(np.amin(samples), 0.0)
        self.assertEqual(samples[-1], 0.0)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), 4)
