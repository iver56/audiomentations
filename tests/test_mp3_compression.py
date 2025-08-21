import os

import numpy as np
import pytest

from audiomentations import Mp3Compression
from audiomentations.core.audio_loading_utils import load_sound_file
from demo.demo import DEMO_DIR
from tests.utils import find_best_alignment_offset_with_corr_coef


def mse(signal1, signal2):
    """Mean squared error"""
    return np.mean((signal1 - signal2) ** 2)


@pytest.mark.parametrize(
    "params",
    [
        {"backend": "fast-mp3-augment", "preserve_delay": False},
        {"backend": "pydub", "preserve_delay": False},
        {"backend": "lameenc", "preserve_delay": True},
    ],
)
@pytest.mark.parametrize(
    "shape",
    [(44100,), (1, 22049), (2, 10000)],
)
def test_apply_mp3_compression(params: dict, shape: tuple):
    samples_in = np.random.normal(0, 1, size=shape).astype(np.float32)
    sample_rate = 44100
    augmenter = Mp3Compression(
        p=1.0,
        min_bitrate=48,
        max_bitrate=48,
        backend=params["backend"],
        preserve_delay=params["preserve_delay"],
    )

    samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
    assert len(shape) == len(samples_out.shape)
    assert samples_out.dtype == np.float32
    assert samples_out.shape[-1] >= shape[-1]
    assert samples_out.shape[-1] < shape[-1] + 3000
    if len(shape) == 2:
        assert samples_out.shape[0] == shape[0]


@pytest.mark.parametrize(
    "params",
    [
        {"backend": "fast-mp3-augment", "preserve_delay": False},
        {"backend": "pydub", "preserve_delay": False},
        {"backend": "lameenc", "preserve_delay": True},
    ],
)
@pytest.mark.parametrize(
    "shape",
    [(16000,), (1, 12049), (2, 5000)],
)
def test_apply_mp3_compression_low_bitrate(params: dict, shape: tuple):
    samples_in = np.random.normal(0, 1, size=shape).astype(np.float32)
    sample_rate = 16000
    augmenter = Mp3Compression(
        p=1.0,
        min_bitrate=8,
        max_bitrate=8,
        backend=params["backend"],
        preserve_delay=params["preserve_delay"],
    )

    samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
    assert len(shape) == len(samples_out.shape)
    assert samples_out.dtype == np.float32
    assert samples_out.shape[-1] >= shape[-1]
    assert samples_out.shape[-1] < shape[-1] + 3100
    if len(shape) == 2:
        assert samples_out.shape[0] == shape[0]


def test_non_contiguous_2d_array():
    samples_in = np.random.normal(0, 1, size=(5000, 2)).astype(np.float32).T
    augmenter = Mp3Compression(p=1.0)
    samples_out = augmenter(samples=samples_in, sample_rate=44100)
    assert samples_out.shape == samples_in.shape
    assert samples_out.dtype == np.float32


def test_invalid_argument_combination():
    with pytest.raises(ValueError):
        _ = Mp3Compression(min_bitrate=400, max_bitrate=800)

    with pytest.raises(ValueError):
        _ = Mp3Compression(min_bitrate=2, max_bitrate=128)

    with pytest.raises(ValueError):
        _ = Mp3Compression(min_bitrate=64, max_bitrate=32)

    with pytest.raises(ValueError):
        _ = Mp3Compression(min_bitrate=66, max_bitrate=67)

    with pytest.raises(ValueError):
        _ = Mp3Compression(backend="both")

    with pytest.raises(ValueError):
        _ = Mp3Compression(backend="pydub", preserve_delay=True)

    with pytest.raises(ValueError):
        _ = Mp3Compression(backend="lameenc", preserve_delay=False)


@pytest.mark.parametrize(
    "params",
    [
        {"backend": "fast-mp3-augment", "preserve_delay": False},
        {"backend": "pydub", "preserve_delay": False},
        {"backend": "lameenc", "preserve_delay": True},
    ],
)
def test_too_loud_input(params):
    """Check that we avoid wrap distortion if input is too loud"""
    samples_in, sample_rate = load_sound_file(
        os.path.join(DEMO_DIR, "perfect-alley1.ogg"),
        sample_rate=None,
        mono=True,
    )
    samples_in = samples_in[..., 0:196000]
    samples_in *= 10
    augmenter = Mp3Compression(
        min_bitrate=320,
        max_bitrate=320,
        backend=params["backend"],
        preserve_delay=params["preserve_delay"],
        p=1.0,
    )

    samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
    samples_out = samples_out[..., 0 : samples_in.shape[-1]]
    assert samples_out.dtype == np.float32

    # Apply delay compensation, so we can compare output with input sample by sample
    offset, _ = find_best_alignment_offset_with_corr_coef(
        reference_signal=samples_in,
        delayed_signal=samples_out,
        min_offset_samples=-3100,
        max_offset_samples=3100,
    )

    snippet_length = 42000
    delay_compensated_samples_out = samples_out[..., offset : offset + snippet_length]
    samples_in = samples_in[..., :snippet_length]
    mse_value = mse(samples_in, delay_compensated_samples_out)

    assert mse_value < 0.00001
