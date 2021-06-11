import unittest

import numpy as np

from audiomentations.augmentations.transforms import TanhDistortion
from audiomentations.core.composition import Compose


class TestTanhDistortion(unittest.TestCase):
    def test_tanh_distortion(self):
        sample_len = 1024
        samples1 = np.zeros((sample_len,), dtype=np.float32)
        samples2 = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([TanhDistortion(c1=2, c2=2)])
        samples_in = np.hstack((samples1, samples2))
        self.assertEqual(len(samples_in), sample_len * 2)
        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)

        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(len(samples_out), sample_len)
