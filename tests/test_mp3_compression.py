import unittest

import numpy as np

from audiomentations.augmentations.transforms import Mp3Compression
from audiomentations.core.composition import Compose


class TestMp3Compression(unittest.TestCase):
    def test_apply_mp3_compression(self):
        sample_len = 44100
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 44100
        augmenter = Compose([Mp3Compression(p=1.0)])

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(len(samples_out), sample_len)
