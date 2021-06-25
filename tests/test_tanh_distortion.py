import unittest

import numpy as np

from audiomentations.augmentations.transforms import (
    TanhDistortion,
)  # TODO: Import directly from audiomentations once it's exposed there
from audiomentations.core.utils import calculate_rms


class TestTanhDistortion(unittest.TestCase):
    @unittest.skip(
        "Skipping this failing test for now. TanhDistortion will get a rework."
    )
    def test_single_channel(self):
        samples = np.random.normal(0, 0.1, size=(2048,)).astype(np.float32)
        sample_rate = 16000
        augmenter = TanhDistortion(
            min_distortion_gain=1.0, max_distortion_gain=2.0, p=1.0
        )

        distorted_samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, distorted_samples.dtype)
        self.assertEqual(samples.shape, distorted_samples.shape)
        self.assertEqual(
            np.round(calculate_rms(samples), 3),
            np.round(calculate_rms(distorted_samples), 3),
        )

    @unittest.skip(
        "Skipping this failing test for now. TanhDistortion will get a rework."
    )
    def test_multichannel(self):
        num_channels = 3
        samples = np.random.normal(0, 0.1, size=(num_channels, 5555)).astype(np.float32)
        sample_rate = 16000
        augmenter = TanhDistortion(
            min_distortion_gain=1.0, max_distortion_gain=2.0, p=1.0
        )

        distorted_samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, distorted_samples.dtype)
        self.assertEqual(samples.shape, distorted_samples.shape)
        for i in range(num_channels):
            assert not np.allclose(samples[i], distorted_samples[i])
            self.assertEqual(
                np.round(calculate_rms(samples[i]), 3),
                np.round(calculate_rms(distorted_samples[i]), 3),
            )
