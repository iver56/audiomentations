import random

import numpy as np
import pytest
from numpy import array_equal

from audiomentations import Padding


@pytest.mark.parametrize("mode", ["silence", "wrap", "reflect"])
@pytest.mark.parametrize("pad_section", ["start", "end"])
def test_padding_mono_1d(mode, pad_section):
    random.seed(546)
    samples = np.array([0.5, 0.6, -0.2, 1.0], dtype=np.float32)
    sample_rate = 16000
    input_shape = samples.shape
    augmenter = Padding(mode=mode, pad_section=pad_section, p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == np.float32
    assert samples.shape == input_shape


def test_padding_mono_2d():
    samples = np.array(
        [[0.9, 0.5, -0.25, -0.125, 0.0]],
        dtype=np.float32,
    )
    sample_rate = 16000
    input_shape = samples.shape

    augmenter = Padding(p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == np.float32
    assert samples.shape == input_shape


@pytest.mark.parametrize("mode", ["silence", "wrap", "reflect"])
@pytest.mark.parametrize("pad_section", ["start", "end"])
def test_padding_multichannel(mode, pad_section):
    samples = np.array(
        [
            [0.9, 0.5, -0.25, -0.125, 0.0],
            [0.95, 0.5, -0.25, -0.125, 0.0],
            [0.95, 0.5, -0.25, -0.125, 0.0],
        ],
        dtype=np.float32,
    )
    sample_rate = 16000
    input_shape = samples.shape

    augmenter = Padding(mode=mode, pad_section=pad_section, p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == np.float32
    assert samples.shape == input_shape


def test_padding_reflect_start():
    samples = np.array([0.5, 0.6, 0.9, -0.2, 1.0], dtype=np.float32)
    sample_rate = 16000
    augmenter = Padding(
        mode="reflect",
        pad_section="start",
        min_fraction=0.4,
        max_fraction=0.4,
        p=1.0,
    )
    samples = augmenter(samples=samples, sample_rate=sample_rate)
    assert array_equal(samples, np.array([1.0, -0.2, 0.9, -0.2, 1.0], dtype=np.float32))


def test_padding_reflect_end():
    samples = np.array([0.5, 0.6, 0.9, -0.2, 1.0], dtype=np.float32)
    sample_rate = 16000
    augmenter = Padding(
        mode="reflect",
        pad_section="end",
        min_fraction=0.4,
        max_fraction=0.4,
        p=1.0,
    )
    samples = augmenter(samples=samples, sample_rate=sample_rate)
    assert array_equal(samples, np.array([0.5, 0.6, 0.9, 0.6, 0.5], dtype=np.float32))


def test_pad_nothing():
    samples = np.array([0.5, 0.6, -0.2, 0.1], dtype=np.float32)
    sample_rate = 16000
    input_shape = samples.shape
    augmenter = Padding(min_fraction=0.0, max_fraction=0.0, p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert array_equal(samples, np.array([0.5, 0.6, -0.2, 0.1], dtype=np.float32))
    assert samples.dtype == np.float32
    assert samples.shape == input_shape


def test_pad_everything():
    samples = np.array([0.5, 0.6, -0.2, 0.7], dtype=np.float32)
    sample_rate = 16000
    input_shape = samples.shape
    augmenter = Padding(min_fraction=1.0, max_fraction=1.0, p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert not np.any(samples)
    assert samples.dtype == np.float32
    assert samples.shape == input_shape
