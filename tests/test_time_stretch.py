import unittest

import numpy as np

from audiomentations.augmentations.transforms import TimeStretch
from audiomentations.core.composition import Compose


class TestTimeStretch(unittest.TestCase):
    def test_dynamic_length(self):
        samples = np.zeros((20,), dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [
                TimeStretch(
                    min_rate=0.8, max_rate=0.9, leave_length_unchanged=False, p=1.0
                )
            ]
        )
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertGreater(len(samples), 20)

    def test_fixed_length(self):
        samples = np.zeros((20,), dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [
                TimeStretch(
                    min_rate=0.8, max_rate=0.9, leave_length_unchanged=True, p=1.0
                )
            ]
        )
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), 20)
