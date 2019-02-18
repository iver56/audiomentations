import unittest
import numpy as np

from audiomentations.augmentations.transforms import AddGaussianNoise, TimeStretch
from audiomentations.core.composition import Compose


class TestCompose(unittest.TestCase):
    def test_compose_single_transform(self):
        samples = np.zeros((20,), dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose([
            AddGaussianNoise(p=1.0)
        ])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertNotAlmostEqual(float(np.sum(np.abs(samples))), 0.0)

    def test_time_stretch(self):
        samples = np.zeros((20,), dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose([
            TimeStretch(min_rate=0.8, max_rate=0.9, p=1.0)
        ])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertGreater(len(samples), 20)

