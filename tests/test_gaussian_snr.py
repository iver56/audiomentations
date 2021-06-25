import json
import math
import unittest

import numpy as np
import random

from audiomentations.core.utils import calculate_rms

from audiomentations.augmentations.transforms import AddGaussianSNR


class TestGaussianSNR(unittest.TestCase):
    def test_gaussian_noise_snr_defaults(self):
        np.random.seed(42)
        samples_in = np.random.normal(0, 1, size=1024).astype(np.float32)
        augmenter = AddGaussianSNR(p=1.0)
        std_in = np.mean(np.abs(samples_in))
        with self.assertWarns(UserWarning):
            samples_out = augmenter(samples=samples_in, sample_rate=16000)
        std_out = np.mean(np.abs(samples_out))
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertNotAlmostEqual(float(std_out), 0.0)
        self.assertGreater(std_out, std_in)

    def test_gaussian_noise_snr_legacy_positional_parameter(self):
        np.random.seed(42)
        samples_in = np.random.normal(0, 1, size=1024).astype(np.float32)
        augmenter = AddGaussianSNR(0.001, 1.0, p=1.0)
        std_in = np.mean(np.abs(samples_in))
        with self.assertWarns(UserWarning):
            samples_out = augmenter(samples=samples_in, sample_rate=16000)
        std_out = np.mean(np.abs(samples_out))
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertNotAlmostEqual(float(std_out), 0.0)
        self.assertGreater(std_out, std_in)

    def test_gaussian_noise_snr_legacy_keyword_parameter(self):
        np.random.seed(42)
        samples_in = np.random.normal(0, 1, size=1024).astype(np.float32)
        augmenter = AddGaussianSNR(min_SNR=0.001, max_SNR=1.0, p=1.0)
        std_in = np.mean(np.abs(samples_in))
        with self.assertWarns(UserWarning):
            samples_out = augmenter(samples=samples_in, sample_rate=16000)
        std_out = np.mean(np.abs(samples_out))
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertNotAlmostEqual(float(std_out), 0.0)
        self.assertGreater(std_out, std_in)

    def test_gaussian_noise_snr_specify_both_new_and_legacy_params(self):
        # Trying to specify both legacy and new parameters. This should raise an exception.
        with self.assertRaises(Exception):
            augmenter = AddGaussianSNR(
                min_SNR=0.001, max_SNR=1.0, min_snr_in_db=15, max_snr_in_db=35, p=1.0
            )

    def test_gaussian_noise_snr(self):
        np.random.seed(42)
        samples_in = np.random.normal(0, 1, size=1024).astype(np.float32)
        augmenter = AddGaussianSNR(min_snr_in_db=15, max_snr_in_db=35, p=1.0)
        std_in = np.mean(np.abs(samples_in))
        samples_out = augmenter(samples=samples_in, sample_rate=16000)
        std_out = np.mean(np.abs(samples_out))
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertNotAlmostEqual(float(std_out), 0.0)
        self.assertGreater(std_out, std_in)

    def test_serialize_parameters(self):
        np.random.seed(42)
        transform = AddGaussianSNR(min_snr_in_db=15, max_snr_in_db=35, p=1.0)
        samples = np.random.normal(0, 1, size=1024).astype(np.float32)
        transform.randomize_parameters(samples, sample_rate=16000)
        json.dumps(transform.serialize_parameters())

    def test_gaussian_noise_snr_multichannel(self):
        np.random.seed(42)
        samples = np.random.normal(0, 0.1, size=(3, 8888)).astype(np.float32)
        augmenter = AddGaussianSNR(min_snr_in_db=15, max_snr_in_db=35, p=1.0)
        samples_out = augmenter(samples=samples, sample_rate=16000)

        self.assertEqual(samples_out.dtype, np.float32)
        self.assertGreater(
            float(np.sum(np.abs(samples_out))), float(np.sum(np.abs(samples)))
        )

    def test_convert_old_parameters_to_new_parameters(self):
        np.random.seed(42)
        samples_in = np.random.normal(0, 1, size=1024).astype(np.float32)
        min_SNR = 0.5
        max_SNR = 0.5
        min_snr_in_db = -20 * math.log10(min_SNR)
        max_snr_in_db = -20 * math.log10(max_SNR)

        self.assertAlmostEqual(min_snr_in_db, 6.0205999)
        self.assertAlmostEqual(max_snr_in_db, 6.0205999)


        legacy_augmenter = AddGaussianSNR(min_SNR=min_SNR, max_SNR=max_SNR, p=1.0)
        new_augmenter = AddGaussianSNR(
            min_snr_in_db=min_snr_in_db, max_snr_in_db=max_snr_in_db, p=1.0
        )

        np.random.seed(42)
        random.seed(42)
        with self.assertWarns(UserWarning):
            samples_out_legacy = legacy_augmenter(samples=samples_in, sample_rate=16000)
        np.random.seed(42)
        random.seed(42)
        samples_out_new = new_augmenter(samples=samples_in, sample_rate=16000)

        legacy_rms = calculate_rms(samples_out_legacy)
        new_rms = calculate_rms(samples_out_new)

        self.assertAlmostEqual(legacy_rms, new_rms, places=3)
