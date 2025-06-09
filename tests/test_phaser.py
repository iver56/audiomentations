import numpy as np
import pytest

from audiomentations import Phaser


def test_phaser():
    samples = np.random.normal(0, 1, size=(2, 1000)).astype(np.float32)
    sample_rate = 44100

    augment = Phaser(
        min_gain=0.5,
        max_gain=1.0,
        min_speed=0.1,
        max_speed=2.0,
        modulation_types=['sinusoidal', 'triangular'],
        p=1.0
    )
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)

    assert augmented_samples.shape == samples.shape
    assert augmented_samples.dtype == np.float32
    assert not np.array_equal(augmented_samples, samples)
    assert np.min(augmented_samples) >= -1.0
    assert np.max(augmented_samples) <= 1.0


def test_phaser_validation():
    with pytest.raises(AssertionError):
        Phaser(min_gain=1.0, max_gain=0.5)  # min_gain > max_gain

    with pytest.raises(AssertionError):
        Phaser(min_speed=2.0, max_speed=0.1)  # min_speed > max_speed

    with pytest.raises(AssertionError):
        Phaser(modulation_types=['invalid'])  # invalid modulation type


def test_phaser_multichannel():
    samples = np.random.normal(0, 1, size=(2, 1000)).astype(np.float32)
    sample_rate = 44100

    augment = Phaser(
        min_gain=0.5,
        max_gain=1.0,
        min_speed=0.1,
        max_speed=2.0,
        modulation_types=['sinusoidal', 'triangular'],
        p=1.0
    )
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)

    assert augmented_samples.shape == samples.shape
    assert augmented_samples.dtype == np.float32
    assert not np.array_equal(augmented_samples, samples)
    assert np.min(augmented_samples) >= -1.0
    assert np.max(augmented_samples) <= 1.0


def test_phaser_mono():
    samples = np.random.normal(0, 1, size=1000).astype(np.float32)
    sample_rate = 44100

    augment = Phaser(
        min_gain=0.5,
        max_gain=1.0,
        min_speed=0.1,
        max_speed=2.0,
        modulation_types=['sinusoidal', 'triangular'],
        p=1.0
    )
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)

    assert augmented_samples.shape == samples.shape
    assert augmented_samples.dtype == np.float32
    assert not np.array_equal(augmented_samples, samples)
    assert np.min(augmented_samples) >= -1.0
    assert np.max(augmented_samples) <= 1.0 