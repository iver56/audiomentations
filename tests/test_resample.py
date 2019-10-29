import math
import unittest

import numpy as np

from audiomentations.augmentations.transforms import Resample
from audiomentations.core.composition import Compose


class TestResample(unittest.TestCase):
    def test_resample(self):
        samples = np.zeros((512,), dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [Resample(min_sample_rate=8000, max_sample_rate=44100, p=1.0)]
        )
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertLessEqual(
            len(samples), math.ceil(len(samples) * 44100 / sample_rate)
        )
        self.assertGreaterEqual(
            len(samples), math.ceil(len(samples) * 8000 / sample_rate)
        )
