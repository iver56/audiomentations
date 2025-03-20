import math

import numpy as np
import pytest

from audiomentations import Resample


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros((512,), dtype=np.float32),
        np.zeros(
            (
                2,
                2512,
            ),
            dtype=np.float32,
        ),
    ],
)
def test_resample(samples):
    sample_rate = 16000
    augmenter = Resample(min_sample_rate=8000, max_sample_rate=44100, p=1.0)
    samples = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == np.float32
    assert samples.shape[-1] <= math.ceil(samples.shape[-1] * 44100 / sample_rate)
    assert samples.shape[-1] >= math.ceil(samples.shape[-1] * 8000 / sample_rate)
