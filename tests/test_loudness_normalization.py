import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from audiomentations import LoudnessNormalization


class TestLoudnessNormalization(unittest.TestCase):
    def test_loudness_normalization(self):
        samples = np.random.uniform(low=-0.2, high=-0.001, size=(8000,)).astype(
            np.float32
        )
        sample_rate = 16000

        augment = LoudnessNormalization(min_lufs_in_db=-32, max_lufs_in_db=-12, p=1.0)
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        gain_factors = processed_samples / samples
        self.assertAlmostEqual(np.amin(gain_factors), np.amax(gain_factors), places=6)
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_loudness_normalization_digital_silence(self):
        samples = np.zeros(8000, dtype=np.float32)
        sample_rate = 16000

        augment = LoudnessNormalization(min_lufs_in_db=-32, max_lufs_in_db=-12, p=1.0)
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        assert_almost_equal(processed_samples, np.zeros(8000, dtype=np.float32))
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_loudness_normalization_too_short_input(self):
        samples = np.random.uniform(low=-0.2, high=-0.001, size=(800,)).astype(
            np.float32
        )
        sample_rate = 16000

        augment = LoudnessNormalization(min_lufs_in_db=-32, max_lufs_in_db=-12, p=1.0)
        with self.assertRaises(ValueError):
            _ = augment(samples=samples, sample_rate=sample_rate)

    def test_loudness_normalization_multichannel(self):
        samples = np.random.uniform(low=-0.2, high=-0.001, size=(3, 8000)).astype(
            np.float32
        )
        sample_rate = 16000

        augment = LoudnessNormalization(min_lufs_in_db=-32, max_lufs_in_db=-12, p=1.0)
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        gain_factors = processed_samples / samples
        self.assertAlmostEqual(np.amin(gain_factors), np.amax(gain_factors), places=6)
        self.assertEqual(processed_samples.dtype, np.float32)
