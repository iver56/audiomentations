import math
import unittest

import numpy as np

from audiomentations.augmentations.transforms import Downsample
from audiomentations.core.composition import Compose


class TestResample(unittest.TestCase):
    def test_downsample(self):
        samples = np.zeros((512,), dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose([Downsample(min_sample_rate=8000, p=1.0)])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertLess(len(samples), 512)
