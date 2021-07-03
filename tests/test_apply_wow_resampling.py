import unittest

import numpy as np

from audiomentations import ApplyWowResampling


class TestApplyWowResampling(unittest.TestCase):
    def test_single_channel(self):
        samples = np.array([0.5, 0.6, -0.2, 0.0], dtype=np.float32)
        sample_rate = 16000
        augmenter = ApplyWowResampling(min_am=0.5, max_am=1.5, min_fm=0.5, max_fm=1.5, p=1.0)
        samples = augmenter(samples=samples, sample_rate=sample_rate)
        
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), 4)

    def test_multichannel(self):
        samples = np.array(
            [[0.9, 0.5, -0.25, -0.125, 0.0], [0.95, 0.5, -0.25, -0.125, 0.0]],
            dtype=np.float32,
        )
        sample_rate = 16000
        augmenter = ApplyWowResampling(min_am=0.5, max_am=1.5, min_fm=0.5, max_fm=1.5, p=1.0)
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
