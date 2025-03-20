import random

import numpy as np
import pytest

from audiomentations import GainTransition


@pytest.mark.parametrize(
    "samples",
    [
        # Test both mono and stereo
        np.random.uniform(low=-0.5, high=0.5, size=(1234,)).astype(np.float32),
        np.random.uniform(low=-0.5, high=0.5, size=(2, 5678)).astype(np.float32),
    ],
)
def test_gain_transition_fraction(samples):
    np.random.seed(42)
    random.seed(42)
    sample_rate = 8000

    augment = GainTransition(p=1.0)
    processed_samples = augment(samples=samples, sample_rate=sample_rate)
    assert not np.allclose(samples, processed_samples)
    assert processed_samples.shape == samples.shape
    assert processed_samples.dtype == np.float32


def test_gain_transition_seconds():
    np.random.seed(40)
    random.seed(40)
    samples = np.random.uniform(low=-0.5, high=0.5, size=(2345,)).astype(np.float32)
    sample_rate = 16000

    augment = GainTransition(
        min_duration=0.2, max_duration=0.3, duration_unit="seconds", p=1.0
    )
    processed_samples = augment(samples=samples, sample_rate=sample_rate)
    assert not np.allclose(samples, processed_samples)
    assert processed_samples.shape == samples.shape
    assert processed_samples.dtype == np.float32


def test_gain_transition_samples():
    np.random.seed(1337)
    random.seed(1337)
    samples = np.random.uniform(low=-0.5, high=0.5, size=(3456,)).astype(np.float32)
    sample_rate = 32000

    augment = GainTransition(
        min_duration=1, max_duration=5000, duration_unit="samples", p=1.0
    )
    processed_samples = augment(samples=samples, sample_rate=sample_rate)
    assert not np.allclose(samples, processed_samples)
    assert processed_samples.shape == samples.shape
    assert processed_samples.dtype == np.float32


def test_invalid_params():
    with pytest.raises(ValueError):
        GainTransition(
            min_duration=-12, max_duration=324, duration_unit="samples", p=1.0
        )

    with pytest.raises(ValueError):
        GainTransition(min_duration=45, max_duration=10, duration_unit="samples", p=1.0)

    with pytest.raises(ValueError):
        GainTransition(min_gain_db=6.0, max_gain_db=-24.0, p=1.0)

    augment = GainTransition(
        min_duration=45, max_duration=45, duration_unit="lightyears", p=1.0
    )
    with pytest.raises(ValueError):
        augment(np.zeros(40, dtype=np.float32), 16000)
