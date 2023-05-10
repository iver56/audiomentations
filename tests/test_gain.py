import warnings

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from audiomentations import Gain
from audiomentations.core.transforms_interface import WrongMultichannelAudioShape


class TestGain:
    def test_gain(self):
        samples = np.array([1.0, 0.5, -0.25, -0.125, 0.0], dtype=np.float32)
        sample_rate = 16000

        augment = Gain(min_gain_in_db=-6, max_gain_in_db=-6, p=1.0)
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        assert_almost_equal(
            processed_samples,
            np.array(
                [0.5011872, 0.2505936, -0.1252968, -0.0626484, 0.0], dtype=np.float32
            ),
        )
        assert processed_samples.dtype == np.float32

    def test_gain_multichannel(self):
        samples = np.array(
            [[1.0, 0.5, -0.25, -0.125, 0.0], [1.0, 0.5, -0.25, -0.125, 0.0]],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = Gain(min_gain_in_db=-6, max_gain_in_db=-6, p=1.0)
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
        assert processed_samples.dtype == np.float32

    def test_gain_multichannel_with_wrong_dimension_ordering(self):
        samples = np.array(
            [[1.0, 0.5, -0.25, -0.125, 0.0], [1.0, 0.5, -0.25, -0.125, 0.0]],
            dtype=np.float32,
        ).T
        print(samples.shape)
        sample_rate = 16000

        augment = Gain(min_gain_in_db=-6, max_gain_in_db=-6, p=1.0)

        with pytest.raises(WrongMultichannelAudioShape):
            processed_samples = augment(samples=samples, sample_rate=sample_rate)
