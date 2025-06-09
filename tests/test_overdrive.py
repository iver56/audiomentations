import numpy as np
import pytest

from audiomentations import Overdrive


def test_overdrive():
    samples = np.random.normal(0, 1, size=(2, 1000)).astype(np.float32)
    sample_rate = 44100

    augment = Overdrive(min_gain=10, max_gain=60, min_colour=0, max_colour=100, p=1.0)
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)

    assert augmented_samples.shape == samples.shape
    assert augmented_samples.dtype == np.float32
    assert not np.array_equal(augmented_samples, samples)
    assert np.min(augmented_samples) >= -1.0
    assert np.max(augmented_samples) <= 1.0


def test_overdrive_validation():
    with pytest.raises(AssertionError):
        Overdrive(min_gain=60, max_gain=10)  # min_gain > max_gain

    with pytest.raises(AssertionError):
        Overdrive(min_colour=100, max_colour=0)  # min_colour > max_colour


def test_overdrive_multichannel():
    samples = np.random.normal(0, 1, size=(2, 1000)).astype(np.float32)
    sample_rate = 44100

    augment = Overdrive(min_gain=10, max_gain=60, min_colour=0, max_colour=100, p=1.0)
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)

    assert augmented_samples.shape == samples.shape
    assert augmented_samples.dtype == np.float32
    assert not np.array_equal(augmented_samples, samples)
    assert np.min(augmented_samples) >= -1.0
    assert np.max(augmented_samples) <= 1.0


def test_overdrive_mono():
    samples = np.random.normal(0, 1, size=1000).astype(np.float32)
    sample_rate = 44100

    augment = Overdrive(min_gain=10, max_gain=60, min_colour=0, max_colour=100, p=1.0)
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)

    assert augmented_samples.shape == samples.shape
    assert augmented_samples.dtype == np.float32
    assert not np.array_equal(augmented_samples, samples)
    assert np.min(augmented_samples) >= -1.0
    assert np.max(augmented_samples) <= 1.0 