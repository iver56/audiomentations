import numpy as np
from numpy.testing import assert_array_equal

from audiomentations import Normalize


def test_normalize_positive_peak():
    samples = np.array([0.5, 0.6, -0.2, 0.0], dtype=np.float32)
    sample_rate = 16000
    augmenter = Normalize(p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert np.amax(samples) == 1.0
    assert samples.dtype == np.float32
    assert samples.shape[-1] == 4


def test_normalize_negative_peak():
    samples = np.array([0.5, 0.6, -0.8, 0.0], dtype=np.float32)
    sample_rate = 16000
    augmenter = Normalize(p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert np.amin(samples) == -1.0
    assert samples[-1] == 0.0
    assert samples.dtype == np.float32
    assert samples.shape[-1] == 4


def test_normalize_all_zeros():
    samples = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sample_rate = 16000
    augmenter = Normalize(p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert np.amin(samples) == 0.0
    assert samples[-1] == 0.0
    assert samples.dtype == np.float32
    assert samples.shape[-1] == 4


def test_normalize_multichannel():
    samples = np.array(
        [[0.9, 0.5, -0.25, -0.125, 0.0], [0.95, 0.5, -0.25, -0.125, 0.0]],
        dtype=np.float32,
    )
    sample_rate = 16000
    augmenter = Normalize(p=1.0)
    processed_samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert_array_equal(processed_samples, samples / 0.95)
    assert processed_samples.dtype == np.float32


def test_normalize_multichannel_conditionally():
    sample_rate = 16000
    augmenter = Normalize(apply_to="only_too_loud_sounds", p=1.0)

    samples = np.array(
        [[0.9, 0.5, -0.25, -0.125, 0.0], [0.95, 0.5, -0.25, -0.125, 0.0]],
        dtype=np.float32,
    )
    processed_samples = augmenter(samples=samples, sample_rate=sample_rate)
    assert_array_equal(processed_samples, samples)
    assert processed_samples.dtype == np.float32

    samples_too_loud = np.array(
        [[0.9, 0.5, -0.25, -0.125, 0.0], [1.2, 0.5, -0.25, -0.125, 0.0]],
        dtype=np.float32,
    )
    processed_samples = augmenter(samples=samples_too_loud, sample_rate=sample_rate)
    assert_array_equal(processed_samples, samples_too_loud / 1.2)
    assert processed_samples.dtype == np.float32
