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

    def test_shift_multichannel(self):
        samples = np.array(
            [[0.75, 0.5, -0.25, -0.125], [0.9, 0.5, -0.25, -0.125]], dtype=np.float32
        )
        sample_rate = 4000

        augment = Shift(min_fraction=0.5, max_fraction=0.5, p=1.0)
        processed_samples = augment(samples=samples, sample_rate=sample_rate)

        assert_almost_equal(
            processed_samples,
            np.array(
                [[-0.25, -0.125, 0.75, 0.5], [-0.25, -0.125, 0.9, 0.5]],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_shift_without_rollover_multichannel(self):
        samples = np.array(
            [[0.75, 0.5, -0.25, -0.125], [0.9, 0.5, -0.25, -0.125]], dtype=np.float32
        )
        sample_rate = 4000

        augment = Shift(min_fraction=0.5, max_fraction=0.5, rollover=False, p=1.0)
        processed_samples = augment(samples=samples, sample_rate=sample_rate)

        assert_almost_equal(
            processed_samples,
            np.array([[0.0, 0.0, 0.75, 0.5], [0.0, 0.0, 0.9, 0.5]], dtype=np.float32),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_shift_fade(self):
        samples = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, -2.0, -3.0, -4.0, -5.0]],
            dtype=np.float32,
        )
        sample_rate = 4000

        augment = Shift(
            min_fraction=0.5,
            max_fraction=0.5,
            rollover=False,
            fade=True,
            fade_duration=3,
            p=1.0,
        )
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        assert_almost_equal(
            processed_samples,
            np.array(
                [[0.0, 0.0, 0.5, 2.0, 3.0], [0.0, 0.0, -0.5, -2.0, -3.0]],
                dtype=np.float32,
            ),
        )

    def test_shift_fade_sample_rate(self):
        samples = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, -2.0, -3.0, -4.0, -5.0]],
            dtype=np.float32,
        )
        sample_rate = 4000

        augment = Shift(
            min_fraction=0.5,
            max_fraction=0.5,
            rollover=False,
            fade=True,
            fade_duration=0.00075,  # 0.00075 * 4000 = 3
            p=1.0,
        )
        processed_samples = augment(samples=samples, sample_rate=sample_rate)

        assert_almost_equal(
            processed_samples,
            np.array(
                [[0.0, 0.0, 0.5, 2.0, 3.0], [0.0, 0.0, -0.5, -2.0, -3.0]],
                dtype=np.float32,
            ),
        )
