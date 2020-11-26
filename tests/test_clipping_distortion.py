import unittest

import numpy as np

from audiomentations.augmentations.transforms import ClippingDistortion
from audiomentations.core.composition import Compose


class TestClippingDistortion(unittest.TestCase):
    def test_distort(self):
        sample_len = 1024
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [
                ClippingDistortion(
                    min_percentile_threshold=20, max_percentile_threshold=40, p=1.0
                )
            ]
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(len(samples_out), sample_len)
        self.assertLess(sum(abs(samples_out)), sum(abs(samples_in)))

    def test_distort_multichannel(self):
        sample_len = 32000
        samples_in = np.random.normal(0, 1, size=(2, sample_len)).astype(np.float32)
        sample_rate = 16000
        augmenter = ClippingDistortion(
            min_percentile_threshold=20, max_percentile_threshold=40, p=1.0
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(samples_out.shape, samples_in.shape)
        self.assertLess(np.sum(np.abs(samples_out)), np.sum(np.abs(samples_in)))
        self.assertAlmostEqual(np.amax(samples_out[0, :]), np.amax(samples_out[1, :]))
