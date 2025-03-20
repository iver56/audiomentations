import json
import random

import fast_align_audio
import numpy as np
import pytest

from audiomentations import Limiter


@pytest.mark.parametrize(
    "samples_in",
    [
        np.random.normal(0, 1, size=1000).astype(np.float32),
        np.random.normal(0, 0.001, size=(1, 250)).astype(np.float32),
        np.random.normal(0, 0.1, size=(3, 8888)).astype(np.float32),
    ],
    ids=["normal_1d_1000", "normal_2d_1_250", "normal_2d_3_8888"],
)
def test_limiter(samples_in):
    augmenter = Limiter(p=1.0, min_attack=0.0025, max_attack=0.0025)
    std_in = np.mean(np.abs(samples_in))
    samples_out = augmenter(samples=samples_in, sample_rate=16000)
    std_out = np.mean(np.abs(samples_out))
    length = samples_in.shape[-1]

    samples_in_mono = samples_in
    samples_out_mono = samples_out
    if samples_in_mono.ndim > 1:
        samples_in_mono = samples_in_mono[0]
        samples_out_mono = samples_out_mono[0]
    offset, _ = fast_align_audio.find_best_alignment_offset(
        reference_signal=samples_in_mono,
        delayed_signal=samples_out_mono,
        max_offset_samples=length // 2,
        lookahead_samples=length // 2,
        method="corr",
    )
    # Check that the output is aligned with the input, i.e. no delay was introduced
    assert offset == 0

    assert samples_out.dtype == np.float32
    assert samples_out.shape == samples_in.shape
    assert std_out < std_in


def test_limiter_validation():
    with pytest.raises(AssertionError):
        Limiter(min_attack=-0.5)


def test_serialize_parameters():
    random.seed(42)
    transform = Limiter(p=1.0)
    samples = np.random.normal(0, 1, size=1024).astype(np.float32)
    transform.randomize_parameters(samples, sample_rate=16000)
    json.dumps(transform.serialize_parameters())


def test_digital_silence():
    samples_in = np.zeros((1024,), np.float32)
    augmenter = Limiter(p=1.0)
    std_in = np.mean(np.abs(samples_in))
    samples_out = augmenter(samples=samples_in, sample_rate=16000)
    std_out = np.mean(np.abs(samples_out))
    assert samples_out.dtype == np.float32
    assert samples_out.shape == samples_in.shape
    assert std_out == std_in == 0.0
