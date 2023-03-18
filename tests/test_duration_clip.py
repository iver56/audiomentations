import random

import numpy as np
import pytest

from audiomentations import DurationClip


class TestDurationClip:
    @pytest.mark.parametrize("mode", ["constant", "wrap", "reflect"])
    @pytest.mark.parametrize("pad_section", ["start", "end"])
    @pytest.mark.parametrize("sample_len", [3, 4, 5])
    @pytest.mark.parametrize("ndim", [None, 1, 2])
    def test_padding(self, mode, pad_section, sample_len, ndim):
        random.seed(546)
        samples = np.ones((ndim, 4) if ndim else 4, dtype=np.float32)
        sample_rate = 16000
        input_shape = samples.shape
        target_shape = list(input_shape)
        target_shape[-1] = sample_len
        target_shape = tuple(target_shape)
        augmenter = DurationClip(
            duration_samples=sample_len, pad_mode=mode, pad_section=pad_section, p=1.0
        )
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        assert samples.dtype == np.float32
        assert samples.shape == target_shape

    @pytest.mark.parametrize("mode", ["constant", "wrap", "reflect"])
    @pytest.mark.parametrize("pad_section", ["start", "end"])
    @pytest.mark.parametrize("second", [0.4, 0.5, 0.6])
    @pytest.mark.parametrize("ndim", [None, 1, 2])
    def test_padding_second(self, mode, pad_section, second, ndim):
        random.seed(546)
        sample_rate = 80
        samples = np.ones((ndim, 40) if ndim else 40, dtype=np.float32)
        input_shape = samples.shape
        target_shape = list(input_shape)
        target_shape[-1] = int(second * sample_rate)
        target_shape = tuple(target_shape)
        augmenter = DurationClip(
            duration_seconds=second, pad_mode=mode, pad_section=pad_section, p=1.0
        )
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        assert samples.dtype == np.float32
        assert samples.shape == target_shape
