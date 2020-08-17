import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from audiomentations.augmentations.transforms import Gain
from audiomentations.core.composition import Compose


class TestGain(unittest.TestCase):
    def test_gain(self):
        samples = np.array([1.0, 0.5, -0.25, -0.125, 0.0], dtype=np.float32)
        sample_rate = 16000

        augment = Compose([Gain(min_gain_in_db=-6, max_gain_in_db=-6, p=1.0)])
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        assert_almost_equal(
            processed_samples,
            np.array(
                [0.5011872, 0.2505936, -0.1252968, -0.0626484, 0.0], dtype=np.float32
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)
