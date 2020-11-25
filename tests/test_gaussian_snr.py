import json
import unittest

import numpy as np

from audiomentations.augmentations.transforms import AddGaussianSNR
from audiomentations.core.composition import Compose


class TestGaussianSNR(unittest.TestCase):
    def test_gaussian_noise_snr(self):
        sample_len = 1024
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([AddGaussianSNR(min_SNR=0.5, max_SNR=1.0, p=1.0)])
        std_in = np.mean(np.abs(samples_in))
        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        std_out = np.mean(np.abs(samples_out))
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertNotAlmostEqual(float(std_out), 0.0)
        self.assertGreater(std_out, std_in)

    def test_serialize_parameters(self):
        transform = AddGaussianSNR(min_SNR=0.5, max_SNR=1.0, p=1.0)
        samples = np.random.normal(0, 1, size=1024).astype(np.float32)
        transform.randomize_parameters(samples, sample_rate=16000)
        json.dumps(transform.serialize_parameters())

    def test_gaussian_noise_snr_multichannel(self):
        samples = np.random.normal(0, 0.1, size=(3, 8888)).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([AddGaussianSNR(p=1.0)])
        samples_out = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples_out.dtype, np.float32)
        self.assertGreater(
            float(np.sum(np.abs(samples_out))), float(np.sum(np.abs(samples)))
        )
