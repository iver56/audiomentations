import unittest

import numpy as np

from audiomentations.augmentations.transforms import ClippingDistortion
from audiomentations.core.composition import Compose


class TestDistortion(unittest.TestCase):
    def test_distort(self):
        sample_len = 1024
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([ClippingDistortion(percentile_cut_off=40, p=1.0)])

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(len(samples_out), sample_len)
        self.assertLessEqual(sum(abs(samples_out)), sum(abs(samples_in)))
