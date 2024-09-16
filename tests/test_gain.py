import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from audiomentations import Gain
from audiomentations.core.transforms_interface import WrongMultichannelAudioShape


def test_gain():
    samples = np.array([1.0, 0.5, -0.25, -0.125, 0.0], dtype=np.float32)
    sample_rate = 16000

    augment = Gain(min_gain_db=-6, max_gain_db=-6, p=1.0)
    processed_samples = augment(samples=samples, sample_rate=sample_rate)
    assert_almost_equal(
        processed_samples,
        np.array([0.5011872, 0.2505936, -0.1252968, -0.0626484, 0.0], dtype=np.float32),
    )
    assert processed_samples.dtype == np.float32


def test_gain_multichannel():
    samples = np.array(
        [[1.0, 0.5, -0.25, -0.125, 0.0], [1.0, 0.5, -0.25, -0.125, 0.0]],
        dtype=np.float32,
    )
    sample_rate = 16000

    augment = Gain(min_gain_db=-6, max_gain_db=-6, p=1.0)
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


def test_gain_multichannel_with_wrong_dimension_ordering():
    samples = np.random.uniform(low=-0.5, high=0.5, size=(2000, 2)).astype(np.float32)

    augment = Gain(min_gain_db=-6, max_gain_db=-6, p=1.0)

    with pytest.raises(WrongMultichannelAudioShape):
        augment(samples=samples, sample_rate=16000)


def test_gain_min_greater_than_max():
    with pytest.raises(ValueError):
        Gain(min_gain_db=60, max_gain_db=10, p=1.0)
