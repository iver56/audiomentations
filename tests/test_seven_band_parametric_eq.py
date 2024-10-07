import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from audiomentations.augmentations.seven_band_parametric_eq import SevenBandParametricEQ


@pytest.mark.parametrize("shape", [(44100,), (1, 22049), (2, 10000)])
def test_apply_eq(shape: tuple):
    samples_in = np.random.normal(0.0, 0.5, size=shape).astype(np.float32)
    sample_rate = 44100
    augmenter = SevenBandParametricEQ(p=1.0)

    samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
    assert samples_out.dtype == np.float32
    assert samples_out.shape == shape

    with np.testing.assert_raises(AssertionError):
        assert_array_almost_equal(samples_out, samples_in)
