  
import unittest

import numpy as np

from audiomentations import HighPassFilter


class TestHighPassFilter(unittest.TestCase):
    def test_high_pass_filter(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125, 0.0], [0.65, 0.5, -0.25, -0.125, 0.0]],
                [[0.3, 0.5, -0.25, -0.125, 0.0], [0.9, 0.5, -0.25, -0.125, 0.0]],
                [[0.9, 0.5, -0.25, -1.06, 0.0], [0.9, 0.5, -0.25, -1.12, 0.0]],
            ],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = HighPassFilter(p=1.0)
        processed_samples = augment(
            samples, sample_rate=sample_rate
        ).numpy()
        self.assertEqual(processed_samples.shape, samples.shape)
        self.assertEqual(processed_samples.dtype, np.float32)
        
