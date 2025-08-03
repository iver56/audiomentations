import numpy as np
from numpy.testing import assert_almost_equal

from audiomentations import PolarityInversion, Compose


def test_polarity_inversion():
    samples = np.array([1.0, 0.5, -0.25, -0.125, 0.0], dtype=np.float32)
    sample_rate = 16000

    augment = Compose([PolarityInversion(p=1.0)])
    inverted_samples = augment(samples=samples, sample_rate=sample_rate)
    assert_almost_equal(
        inverted_samples, np.array([-1.0, -0.5, 0.25, 0.125, 0.0], dtype=np.float32)
    )
    assert inverted_samples.dtype == np.float32


def test_polarity_inversion_multichannel():
    samples = np.array(
        [[1.0, 0.5, -0.25, -0.125, 0.0], [1.0, 0.5, -0.25, -0.125, 0.0]],
        dtype=np.float32,
    )
    sample_rate = 16000

    augment = Compose([PolarityInversion(p=1.0)])
    inverted_samples = augment(samples=samples, sample_rate=sample_rate)
    assert_almost_equal(
        inverted_samples,
        np.array(
            [[-1.0, -0.5, 0.25, 0.125, 0.0], [-1.0, -0.5, 0.25, 0.125, 0.0]],
            dtype=np.float32,
        ),
    )
    assert inverted_samples.dtype == np.float32
