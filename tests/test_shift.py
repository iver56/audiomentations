import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from audiomentations.augmentations.transforms import Shift
from audiomentations.core.composition import Compose


class TestShift(unittest.TestCase):
    def test_shift(self):
        samples = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32)
        sample_rate = 16000

        forward_augmenter = Compose([Shift(min_fraction=0.5, max_fraction=0.5, p=1.0)])
        forward_shifted_samples = forward_augmenter(
            samples=samples, sample_rate=sample_rate
        )
        assert_almost_equal(
            forward_shifted_samples, np.array([0.25, 0.125, 1.0, 0.5], dtype=np.float32)
        )
        self.assertEqual(forward_shifted_samples.dtype, np.float32)
        self.assertEqual(len(forward_shifted_samples), 4)

        backward_augmenter = Compose(
            [Shift(min_fraction=-0.25, max_fraction=-0.25, p=1.0)]
        )
        backward_shifted_samples = backward_augmenter(
            samples=samples, sample_rate=sample_rate
        )
        assert_almost_equal(
            backward_shifted_samples,
            np.array([0.5, 0.25, 0.125, 1.0], dtype=np.float32),
        )
        self.assertEqual(backward_shifted_samples.dtype, np.float32)
        self.assertEqual(len(backward_shifted_samples), 4)

    def test_shift_without_rollover(self):
        samples = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32)
        sample_rate = 16000

        forward_augmenter = Compose(
            [Shift(min_fraction=0.5, max_fraction=0.5, rollover=False, p=1.0)]
        )
        forward_shifted_samples = forward_augmenter(
            samples=samples, sample_rate=sample_rate
        )
        assert_almost_equal(
            forward_shifted_samples, np.array([0.0, 0.0, 1.0, 0.5], dtype=np.float32)
        )
        self.assertEqual(forward_shifted_samples.dtype, np.float32)
        self.assertEqual(len(forward_shifted_samples), 4)

        backward_augmenter = Compose(
            [Shift(min_fraction=-0.25, max_fraction=-0.25, rollover=False, p=1.0)]
        )
        backward_shifted_samples = backward_augmenter(
            samples=samples, sample_rate=sample_rate
        )
        assert_almost_equal(
            backward_shifted_samples,
            np.array([0.5, 0.25, 0.125, 0.0], dtype=np.float32),
        )
        self.assertEqual(backward_shifted_samples.dtype, np.float32)
        self.assertEqual(len(backward_shifted_samples), 4)
