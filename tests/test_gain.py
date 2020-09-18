import unittest
import warnings

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

    def test_gain_multichannel(self):
        samples = np.array(
            [[1.0, 0.5, -0.25, -0.125, 0.0], [1.0, 0.5, -0.25, -0.125, 0.0]],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = Compose([Gain(min_gain_in_db=-6, max_gain_in_db=-6, p=1.0)])
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        assert_almost_equal(
            processed_samples,
            np.array(
                [
                    [0.5011872, 0.2505936, -0.1252968, -0.0626484, 0.0],
                    [0.5011872, 0.2505936, -0.1252968, -0.0626484, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_gain_multichannel_with_wrong_dimension_ordering(self):
        samples = np.array(
            [[1.0, 0.5, -0.25, -0.125, 0.0], [1.0, 0.5, -0.25, -0.125, 0.0]],
            dtype=np.float32,
        ).T
        print(samples.shape)
        sample_rate = 16000

        augment = Compose([Gain(min_gain_in_db=-6, max_gain_in_db=-6, p=1.0)])

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            processed_samples = augment(samples=samples, sample_rate=sample_rate)

            assert len(w) == 1
            assert "Multichannel audio must have channels first" in str(w[-1].message)
