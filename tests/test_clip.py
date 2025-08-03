import numpy as np
import pytest

from audiomentations import Clip


def test_single_channel():
    samples = np.array([0.5, 0.6, -0.2, 0.0], dtype=np.float32)
    sample_rate = 16000
    augmenter = Clip(a_min=-0.1, a_max=0.1, p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert np.amin(samples) == pytest.approx(-0.1)
    assert np.amax(samples) == pytest.approx(0.1)
    assert samples.dtype == np.float32
    assert samples.shape[-1] == 4


def test_multichannel():
    samples = np.array(
        [[0.9, 0.5, -0.25, -0.125, 0.0], [0.95, 0.5, -0.25, -0.125, 0.0]],
        dtype=np.float32,
    )
    sample_rate = 16000
    augmenter = Clip(a_min=-0.1, a_max=0.1, p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert np.amin(samples) == pytest.approx(-0.1)
    assert np.amax(samples) == pytest.approx(0.1)
    assert samples.dtype == np.float32
