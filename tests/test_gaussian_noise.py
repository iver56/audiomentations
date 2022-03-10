import unittest

import numpy as np
import pytest

from audiomentations import AddGaussianNoise, Compose


class TestGaussianNoise(unittest.TestCase):
    def test_gaussian_noise(self):
        samples = np.zeros((20,), dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose([AddGaussianNoise(p=1.0)])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        assert samples.dtype == np.float32
        assert not (float(np.sum(np.abs(samples))) == pytest.approx(0.0))

    def test_gaussian_noise_stereo(self):
        samples = np.zeros((2, 2000), dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose([AddGaussianNoise(p=1.0)])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        assert samples.dtype == np.float32
        assert not (float(np.sum(np.abs(samples))) == pytest.approx(0.0))
