import json
import random

import numpy as np
import pytest

from audiomentations import Limiter


class TestLimiter:
    @pytest.mark.parametrize("samples_in", [
        np.random.normal(0, 1, size=1024).astype(np.float32),
        np.random.normal(0, 0.001, size=(1, 50)).astype(np.float32),
        np.random.normal(0, 0.1, size=(3, 8888)).astype(np.float32)
    ])
    def test_limiter(self, samples_in):
        random.seed(42)
        augmenter = Limiter(p=1.0)
        std_in = np.mean(np.abs(samples_in))
        samples_out = augmenter(samples=samples_in, sample_rate=16000)
        std_out = np.mean(np.abs(samples_out))
        assert samples_out.dtype == np.float32
        assert samples_out.shape == samples_in.shape
        assert std_out < std_in

    def test_serialize_parameters(self):
        random.seed(42)
        transform = Limiter(p=1.0)
        samples = np.random.normal(0, 1, size=1024).astype(np.float32)
        transform.randomize_parameters(samples, sample_rate=16000)
        json.dumps(transform.serialize_parameters())
