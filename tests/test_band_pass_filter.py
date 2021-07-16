import unittest

import numpy as np
from numpy.testing import assert_array_equal

from audiomentations.augmentations.transforms import BandPassFilter


class TestBandPassFilter(unittest.TestCase):
    def test_band_pass_filter(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125, 0.0], [0.65, 0.5, -0.25, -0.125, 0.0]],
                [[0.3, 0.5, -0.25, -0.125, 0.0], [0.9, 0.5, -0.25, -0.125, 0.0]],
                [[0.9, 0.5, -0.25, -1.06, 0.0], [0.9, 0.5, -0.25, -1.12, 0.0]],
            ],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = BandPassFilter(min_center_freq=100, 
                                 max_center_freq=5000,
                                 min_q=1.0,
                                 max_q=2.0,
                                 p=1.0)
        processed_samples = augment(
            samples=samples, sample_rate=sample_rate
        )
        self.assertEqual(processed_samples.shape, samples.shape)
        self.assertEqual(processed_samples.dtype, np.float32)
        assert_raises(AssertionError, assert_array_equal, processed_samples, samples)
