import unittest

import numpy as np

from audiomentations.augmentations.transforms import SmoothFadeTimeMask
from audiomentations.core.composition import Compose


class TestSmoothFadeTimeMask(unittest.TestCase):
    def test_dynamic_length(self):
        sample_len = 1024
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [SmoothFadeTimeMask(min_band_part=0.2, max_band_part=0.5, p=1.0)]
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(len(samples_out), sample_len)

        std_in = np.mean(np.abs(samples_in))
        std_out = np.mean(np.abs(samples_out))
        self.assertLess(std_out, std_in)
