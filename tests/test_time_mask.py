import numpy as np
import pytest

from audiomentations import TimeMask


def test_apply_time_mask():
    sample_len = 1024
    samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
    sample_rate = 16000
    augmenter = TimeMask(min_band_part=0.2, max_band_part=0.5, fade_duration=0.0, p=1.0)

    samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
    assert samples_out.dtype == np.float32
    assert len(samples_out) == sample_len

    std_in = np.mean(np.abs(samples_in))
    std_out = np.mean(np.abs(samples_out))
    assert std_out < std_in


def test_apply_time_mask_start():
    sample_len = 1000
    samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
    sample_rate = 8000
    augmenter = TimeMask(min_band_part=0.1, max_band_part=0.1, fade_duration=0.0, mask_location="start", p=1.0)

    samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
    assert samples_out.dtype == np.float32
    assert len(samples_out) == sample_len
    assert not np.any(samples_out[0:100])
    assert samples_out[101] == samples_in[101]

def test_apply_time_mask_end():
    sample_len = 1000
    samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
    sample_rate = 8000
    augmenter = TimeMask(min_band_part=0.1, max_band_part=0.1, fade_duration=0.0, mask_location="end", p=1.0)

    samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
    assert samples_out.dtype == np.float32
    assert len(samples_out) == sample_len
    assert not np.any(samples_out[-100:])
    assert samples_out[-101] == samples_in[-101]


def test_invalid_params():
    with pytest.raises(ValueError):
        TimeMask(min_band_part=0.5, max_band_part=1.5)

    with pytest.raises(ValueError):
        TimeMask(min_band_part=-0.5, max_band_part=0.5)

    with pytest.raises(ValueError):
        TimeMask(min_band_part=0.6, max_band_part=0.5)

    with pytest.raises(ValueError):
        TimeMask(fade_duration=-0.1)

    with pytest.raises(ValueError):
        TimeMask(fade_duration=0.00001)

    with pytest.raises(ValueError):
        TimeMask(mask_location="beyond_infinity")

    with pytest.raises(TypeError):
        TimeMask(fade=True)


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
    augmenter = TimeMask(min_band_part=0.2, max_band_part=0.5, p=1.0)

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
    augmenter = TimeMask(min_band_part=0.2, max_band_part=0.5, p=1.0)

    samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
    assert samples_out.dtype == np.float32
    assert len(samples_out) == sample_len

    std_in = np.mean(np.abs(samples_in))
    std_out = np.mean(np.abs(samples_out))
    assert std_out < std_in
