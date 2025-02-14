from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from audiomentations import PitchShift


@pytest.mark.parametrize("method", ["librosa_phase_vocoder", "signalsmith_stretch"])
def test_apply_pitch_shift_1dim(method):
    samples = np.zeros((2048,), dtype=np.float32)
    sample_rate = 16000
    augmenter = PitchShift(min_semitones=-2, max_semitones=-1, method=method, p=1.0)
    samples_out = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples_out.dtype == np.float32
    assert samples_out.shape == (2048,)


@pytest.mark.parametrize("method", ["librosa_phase_vocoder", "signalsmith_stretch"])
def test_apply_pitch_shift_multichannel(method):
    num_channels = 3
    samples = np.random.normal(0, 0.1, size=(num_channels, 5555)).astype(np.float32)
    sample_rate = 16000
    augmenter = PitchShift(min_semitones=1, max_semitones=2, method=method, p=1.0)
    samples_out = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples_out.dtype == np.float32
    assert samples_out.shape == samples.shape
    for i in range(num_channels):
        assert not np.allclose(samples[i], samples_out[i])


def test_freeze_parameters():
    """
    Test that the transform can freeze its parameters, e.g. to apply the effect with the
    same parameters to multiple sounds.
    """
    samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 8000)).astype(np.float32)
    sample_rate = 16000
    augmenter = PitchShift(min_semitones=1, max_semitones=12, p=1.0)

    first_samples = augmenter(samples=samples, sample_rate=sample_rate)
    first_parameters = deepcopy(augmenter.parameters)

    augmenter.min_semitones = -12
    augmenter.max_semitones = -1
    augmenter.are_parameters_frozen = True
    second_samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert first_parameters == augmenter.parameters
    assert_array_equal(first_samples, second_samples)


def test_validate_parameters():
    with pytest.raises(ValueError):
        PitchShift(min_semitones=-40)
    with pytest.raises(ValueError):
        PitchShift(max_semitones=40)
    with pytest.raises(ValueError):
        PitchShift(min_semitones=5, max_semitones=2)
    with pytest.raises(ValueError):
        PitchShift(method="invalidvalue")
