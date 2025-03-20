import numpy as np
import pytest

from audiomentations import TimeMask, Compose


def test_apply_time_mask():
    sample_len = 1024
    samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
    sample_rate = 16000
    augmenter = TimeMask(min_band_part=0.2, max_band_part=0.5, p=1.0)

    samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
    assert samples_out.dtype == np.float32
    assert len(samples_out) == sample_len

    std_in = np.mean(np.abs(samples_in))
    std_out = np.mean(np.abs(samples_out))
    assert std_out < std_in


def test_invalid_params():
    with pytest.raises(ValueError):
        TimeMask(min_band_part=0.5, max_band_part=1.5)

    with pytest.raises(ValueError):
        TimeMask(min_band_part=-0.5, max_band_part=0.5)

    with pytest.raises(ValueError):
        TimeMask(min_band_part=0.6, max_band_part=0.5)


def test_apply_time_mask_multichannel():
    sample_len = 1024
    samples_in = np.random.normal(0, 1, size=(2, sample_len)).astype(np.float32)
    sample_rate = 16000
    augmenter = TimeMask(min_band_part=0.2, max_band_part=0.5, p=1.0)

    samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
    assert samples_out.dtype == np.float32
    assert samples_out.shape == samples_in.shape

    std_in = np.mean(np.abs(samples_in))
    std_out = np.mean(np.abs(samples_out))
    assert std_out < std_in


def test_apply_time_mask_with_fade():
    sample_len = 1024
    samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
    sample_rate = 16000
    augmenter = TimeMask(min_band_part=0.2, max_band_part=0.5, fade=True, p=1.0)

    samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
    assert samples_out.dtype == np.float32
    assert len(samples_out) == sample_len

    std_in = np.mean(np.abs(samples_in))
    std_out = np.mean(np.abs(samples_out))
    assert std_out < std_in


def test_apply_time_mask_with_fade_short_signal():
    sample_len = 100
    samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
    sample_rate = 16000
    augmenter = TimeMask(min_band_part=0.2, max_band_part=0.5, fade=True, p=1.0)

    samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
    assert samples_out.dtype == np.float32
    assert len(samples_out) == sample_len

    std_in = np.mean(np.abs(samples_in))
    std_out = np.mean(np.abs(samples_out))
    assert std_out < std_in
