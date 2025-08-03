import numpy as np
from numpy.testing import assert_array_almost_equal

from audiomentations import Reverse


def test_single_channel():
    samples = np.array([0.5, 0.6, -0.2, 0.0], dtype=np.float32)
    sample_rate = 16000
    augmenter = Reverse(p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == np.float32
    assert samples.shape[-1] == 4


def test_multichannel():
    samples = np.array(
        [[0.9, 0.5, -0.25, -0.125, 0.0], [0.95, 0.5, -0.25, -0.125, 0.0]],
        dtype=np.float32,
    )
    sample_rate = 16000
    augmenter = Reverse(p=1.0)
    reversed_samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert reversed_samples.dtype == np.float32
    assert_array_almost_equal(
        reversed_samples,
        np.array(
            [[0.0, -0.125, -0.25, 0.5, 0.9], [0.0, -0.125, -0.25, 0.5, 0.95]],
            dtype=np.float32,
        ),
    )
