import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from audiomentations.augmentations.transforms import Reverse


class TestReverse(unittest.TestCase):
    def test_single_channel(self):
        samples = np.array([0.5, 0.6, -0.2, 0.0], dtype=np.float32)
        sample_rate = 16000
        augmenter = Reverse(p=1.0)
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), 4)

    def test_multichannel(self):
        samples = np.array(
            [[0.9, 0.5, -0.25, -0.125, 0.0], [0.95, 0.5, -0.25, -0.125, 0.0]],
            dtype=np.float32,
        )
        sample_rate = 16000
        augmenter = Reverse(p=1.0)
        reversed_samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        assert_array_almost_equal(
            reversed_samples,
            np.array(
                [[0.0, -0.125, -0.25, 0.5, 0.9], [0.0, -0.125, -0.25, 0.5, 0.95]],
                dtype=np.float32,
            ),
        )
