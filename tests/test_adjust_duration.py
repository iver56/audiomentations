import random

import numpy as np
import pytest

from audiomentations import AdjustDuration


@pytest.mark.parametrize("mode", ["silence", "wrap", "reflect"])
@pytest.mark.parametrize("pad_section", ["start", "end"])
@pytest.mark.parametrize("sample_len", [3, 4, 5])
@pytest.mark.parametrize("ndim", [None, 1, 2])
def test_padding(mode, pad_section, sample_len, ndim):
    random.seed(546)
    samples = np.ones((ndim, 4) if ndim else 4, dtype=np.float32)
    sample_rate = 16000
    input_shape = samples.shape
    target_shape = list(input_shape)
    target_shape[-1] = sample_len
    target_shape = tuple(target_shape)
    augmenter = AdjustDuration(
        duration_samples=sample_len,
        padding_mode=mode,
        padding_position=pad_section,
        p=1.0,
    )
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == np.float32
    assert samples.shape == target_shape


@pytest.mark.parametrize("mode", ["silence", "wrap", "reflect"])
@pytest.mark.parametrize("pad_section", ["start", "end"])
@pytest.mark.parametrize("second", [0.4, 0.5, 0.6])
@pytest.mark.parametrize("ndim", [None, 1, 2])
def test_padding_second(mode, pad_section, second, ndim):
    random.seed(546)
    sample_rate = 80
    samples = np.ones((ndim, 40) if ndim else 40, dtype=np.float32)
    input_shape = samples.shape
    target_shape = list(input_shape)
    target_shape[-1] = int(second * sample_rate)
    target_shape = tuple(target_shape)
    augmenter = AdjustDuration(
        duration_seconds=second, padding_mode=mode, padding_position=pad_section, p=1.0
    )
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == np.float32
    assert samples.shape == target_shape


def test_adjust_duration_raises_with_both_samples_and_seconds():
    with pytest.raises(ValueError) as exc_info:
        _ = AdjustDuration(duration_samples=100, duration_seconds=1.0)
    assert (
        "You must specify either duration_samples or duration_seconds, but not both."
        in str(exc_info.value)
    )


def test_adjust_duration_raises_with_non_positive_seconds():
    with pytest.raises(ValueError) as exc_info:
        _ = AdjustDuration(duration_seconds=0.0)
    assert "duration_seconds must be a positive float" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        _ = AdjustDuration(duration_seconds=-1.0)
    assert "duration_seconds must be a positive float" in str(exc_info.value)


def test_adjust_duration_raises_with_non_positive_samples():
    with pytest.raises(ValueError) as exc_info:
        _ = AdjustDuration(duration_samples=0)
    assert "duration_samples must be a positive int" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        _ = AdjustDuration(duration_samples=-1)
    assert "duration_samples must be a positive int" in str(exc_info.value)
