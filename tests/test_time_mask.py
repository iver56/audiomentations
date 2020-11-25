import unittest

import numpy as np

from audiomentations.augmentations.transforms import TimeMask
from audiomentations.core.composition import Compose


class TestTimeMask(unittest.TestCase):
    def test_apply_time_mask(self):
        sample_len = 1024
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([TimeMask(min_band_part=0.2, max_band_part=0.5, p=1.0)])

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(len(samples_out), sample_len)

        std_in = np.mean(np.abs(samples_in))
        std_out = np.mean(np.abs(samples_out))
        self.assertLess(std_out, std_in)

    def test_apply_time_mask_multichannel(self):
        sample_len = 1024
        samples_in = np.random.normal(0, 1, size=(2, sample_len)).astype(np.float32)
        sample_rate = 16000
        augmenter = TimeMask(min_band_part=0.2, max_band_part=0.5, p=1.0)

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(samples_out.shape, samples_in.shape)

        std_in = np.mean(np.abs(samples_in))
        std_out = np.mean(np.abs(samples_out))
        self.assertLess(std_out, std_in)

    def test_apply_time_mask_with_fade(self):
        sample_len = 1024
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [TimeMask(min_band_part=0.2, max_band_part=0.5, fade=True, p=1.0)]
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(len(samples_out), sample_len)

        std_in = np.mean(np.abs(samples_in))
        std_out = np.mean(np.abs(samples_out))
        self.assertLess(std_out, std_in)

    def test_apply_time_mask_with_fade_short_signal(self):
        sample_len = 100
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [TimeMask(min_band_part=0.2, max_band_part=0.5, fade=True, p=1.0)]
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(len(samples_out), sample_len)

        std_in = np.mean(np.abs(samples_in))
        std_out = np.mean(np.abs(samples_out))
        self.assertLess(std_out, std_in)
